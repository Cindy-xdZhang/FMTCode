"""Append a capacity-matched PointNet++ SSG to the frozen Task4-c comparison."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from FMT_Utils.Task4C_PointNetPlusPlus_1_1 import PARAMETERS, CACHE_SHAPES, make_model, build_hierarchy, canonical_order, farthest_points, ball_query, index_points
from experiments import Task4C_GeometricBaselines_1_1 as geometry
from experiments import Task4C_BottomDensity_1_2 as source
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine

CONFIG = 'config/Ablation_Task4C_PointNetPlusPlus_1.1.json'
SPLITS = engine.SPLITS
write, sha, metrics, append_locked, deterministic = engine.write, engine.sha, engine.metrics, engine.append_locked, engine.deterministic


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    assert set(definition)=={'version','user_version','execution_revision','output','base_config','base_config_sha256',
        'source_output','source_scientific_commit','dataset_policy','methods','pointnetplusplus'}
    assert sha(definition['base_config']) == definition['base_config_sha256']
    spec = source.load_spec(definition['base_config']); spec.update(definition)
    assert spec['methods'] == ['pointnetplusplus_ssg_small']
    assert spec['pointnetplusplus']['parameters'] == PARAMETERS
    assert spec['pointnetplusplus']['centers'] == [512,128]
    assert spec['pointnetplusplus']['radii'] == [.2,.4] and spec['pointnetplusplus']['neighbors'] == [32,64]
    assert spec['pointnetplusplus']['mlps'] == [[16,16,32],[32,32,64],[64,128,256]]
    assert spec['pointnetplusplus']['classifier_hidden'] == [80,45]
    assert spec['pointnetplusplus']['dropout'] == spec['training']['dropout'] == .15
    assert spec['dataset_policy'] == 'reuse_all_source_geometry_seeds_metadata_without_sampling'
    return spec


def identity(config):
    result = geometry.identity(config)
    for file in ('FMT_Utils/Task4C_PointNetPlusPlus_1_1.py', 'experiments/Task4C_PointNetPlusPlus_1_1.py',
                 'ibex_bash/task4c_pointnetplusplus_1p1.sh', 'FMT_Utils/Task4C_PointNet_1_1.py',
                 'third_party/pointnet2/LICENSE'):
        result['sources'][file] = sha(file)
    return result


def run_engine(name, *args):
    fn = getattr(engine, name)
    return FunctionType(fn.__code__, dict(engine.__dict__, identity=identity), argdefs=fn.__defaults__)(*args)


def reuse(spec, config):
    fn = geometry.reuse
    FunctionType(fn.__code__, dict(geometry.__dict__, identity=identity), argdefs=fn.__defaults__)(spec, config)


def load_parts(spec, role, method):
    assert method == 'pointnetplusplus_ssg_small'
    dataset=geometry.load_parts(spec,role,'baseline2')
    for part,flow in zip(dataset[0],spec['flows']):
        cache=Path(spec['output'])/'encoded'/flow['name']/role
        manifest=json.loads((cache/'manifest.json').read_text())
        part['hierarchy']={}
        for key in CACHE_SHAPES:
            file=cache/(key+'.npy'); assert sha(file)==manifest['files'][file.name]
            part['hierarchy'][key]=np.load(file,mmap_mode='r')
    return dataset


def get_batch(parts, ids, method, norm, device='cuda'):
    assert method == 'pointnetplusplus_ssg_small' and norm is None
    g,c=geometry.get_batch(parts,ids,'baseline2',None,device)
    h={key:torch.as_tensor(np.stack([parts[p]['hierarchy'][key][i] for p,i in ids]).astype(np.int64),device=device) for key in CACHE_SHAPES}
    return g,c,h


def predict(model, dataset, method, norm, batch):
    fn = engine.predict
    return FunctionType(fn.__code__, dict(engine.__dict__, get_batch=get_batch), argdefs=fn.__defaults__)(model, dataset, method, norm, batch)


# PointNet++ validation and index-cache preparation are defined below.


def train(spec,config,index):
    deterministic('cuda');seeds=spec['training']['seeds'];method=spec['methods'][index//len(seeds)];seed=seeds[index%len(seeds)]
    torch.manual_seed(seed);rng=np.random.default_rng(seed);options=spec['training'];batch=options['batch_size']
    root=Path(spec['output']);folder=root/'runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    training=load_parts(spec,'train',method);validation=load_parts(spec,'validation',method)
    parts,ids,y,_,_=training;assert len(y)==spec['expected_counts']['train']
    norm=None
    model=make_model(options['dropout']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=options['lr_patience'],threshold=1e-4,min_lr=1e-6)
    best=(-1.,-1.);best_epoch=0;state=None;history=[];started=time.perf_counter()
    for epoch in range(1,options['epochs']+1):
        order=rng.permutation(len(y));model.train();total=0.;lr=optimizer.param_groups[0]['lr']
        for first in range(0,len(y),batch):
            chosen=order[first:first+batch];xx=get_batch(parts,ids[chosen],method,norm)
            yy=torch.as_tensor(y[chosen],device='cuda',dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(xx),yy)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(chosen)
        probability,val_loss=predict(model,validation,method,norm,batch);score=metrics(validation[2],probability)
        rank=(score['f1'],score['average_precision'])
        row=dict(epoch=epoch,training_loss=total/len(y),validation_loss=val_loss,validation_f1=score['f1'],
            validation_average_precision=score['average_precision'],learning_rate=lr,samples=len(y),
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),seconds=time.perf_counter()-started)
        history.append(row);append_locked(folder/'history.jsonl',json.dumps(row)+'\n')
        if rank>best:best=rank;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        scheduler.step(val_loss)
        if epoch==1 or epoch%10==0:print(method,seed,row,flush=True)
        if epoch-best_epoch>=options['patience']:break
    model.load_state_dict(state);vp,_=predict(model,validation,method,norm,batch)
    assert abs(metrics(validation[2],vp)['f1']-best[0])<1e-12
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    write(folder/'selection.lock.json',dict(identity=identity(config),selected_epoch=best_epoch,threshold=.5,test_loaded=False))
    outputs={}
    for role,dataset in [('train',training),('validation',validation),('test',None)]:
        if role=='test':dataset=load_parts(spec,'test',method)
        p,_=predict(model,dataset,method,norm,batch);_,rowids,yy,flows,owners=dataset
        assert len(yy)==spec['expected_counts'][role]
        np.savez_compressed(folder/f'{role}_predictions.npz',labels=yy,probability=p,flow_index=flows,
            instance=owners,row_in_split=rowids[:,1],threshold=.5)
        outputs[role]=dict(combined=metrics(yy,p),per_flow={f['name']:metrics(yy[flows==i],p[flows==i]) for i,f in enumerate(spec['flows'])})
    result=dict(complete=True,version=spec['version'],execution_revision=spec['execution_revision'],identity=identity(config),
        method=method,seed=seed,parameters=sum(p.numel() for p in model.parameters()),epochs=len(history),selected_epoch=best_epoch,
        training_seconds=elapsed,metrics=outputs,threshold=.5,selection=options['selection'],history=history,
        normalization=None if norm is None else dict(mean=norm[0].tolist(),std=norm[1].tolist()),
        predictions={role:sha(folder/f'{role}_predictions.npz') for role in SPLITS},gpu=torch.cuda.get_device_name())
    write(folder/'result.json',result);print(json.dumps(outputs),flush=True)

def merge(spec, config):
    fn = geometry.reuse_source.merge
    FunctionType(fn.__code__, dict(geometry.reuse_source.__dict__, identity=identity, run_engine=run_engine), argdefs=fn.__defaults__)(spec, config)
    file = Path(spec['output'])/'viewer_package/manifest.json'; manifest = json.loads(file.read_text())
    manifest['evidence_note'] = 'PointNet++ single-scale grouping with reduced widths and frozen geometry-only indices; same 193000/3000/10000 bundles; seed 96611.'
    write(file, manifest)


def submit(spec, config):
    root = Path(spec['output']).resolve(); assert shutil.disk_usage(Path.cwd()).free > 20*2**30
    root.mkdir(parents=True, exist_ok=False); (root/'logs').mkdir(); write(root/'config.frozen.json', spec)
    dependency = None
    phases = [('preflight', None, True, '00:20:00'), ('reuse', None, False, '00:30:00'),
              ('pilot', None, True, '00:30:00'), ('encode', '0-1%2', True, '04:00:00'), ('train', '0-2%3', True, '2-00:00:00'), ('merge', None, False, '01:00:00')]
    for phase, array, gpu, limit in phases:
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=32G', '--time='+limit,
                   '--job-name=t4c-pointnetpp-'+phase, f'--output={root}/logs/{phase}_%A_%a.out', f'--error={root}/logs/{phase}_%A_%a.err']
        if array: command += ['--array='+array]
        if gpu: command += ['--gres=gpu:1', '--constraint=v100']
        if dependency: command += ['--dependency=afterok:'+dependency, '--kill-on-invalid-dep=yes']
        command += ['ibex_bash/task4c_pointnetplusplus_1p1.sh', phase, config]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        row = dict(job_id=job, phase=phase, submitted_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                   dependency=dependency, expected_device='V100' if gpu else 'CPU', identity=identity(config))
        append_locked(root/'submissions.jsonl', json.dumps(row)+'\n'); print(json.dumps(row), flush=True); dependency = job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'reuse', 'pilot', 'encode', 'train', 'merge', 'submit', 'runtime'))
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--runtime-phase')
    parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = load_spec(args.config)
    if args.phase == 'preflight': preflight(spec, args.config, args.device)
    elif args.phase in ('encode','train'): globals()[args.phase](spec, args.config, args.index)
    elif args.phase == 'runtime': run_engine('runtime', spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else: globals()[args.phase](spec, args.config)


def preflight(spec, config, device):
    deterministic(device); torch.manual_seed(91721)
    small=torch.randn((2,16,3),device=device)*.3
    nc=torch.tensor([16,9],device=device); sampled=farthest_points(small,nc,8)
    xyz=small.double().cpu().numpy()
    for b,n in enumerate(nc.tolist()):
        chosen=[0]
        while len(chosen)<8:
            candidates=[i for i in range(n) if i not in chosen]
            chosen.append(max(candidates,key=lambda i:(min(np.sum((xyz[b,i]-xyz[b,j])**2) for j in chosen),-i)))
        assert sampled[b].tolist()==chosen
    centers=index_points(small,sampled)
    groups=ball_query(small,centers,nc,torch.full_like(nc,8),.4,4)
    for b,n in enumerate(nc.tolist()):
        for i,center in enumerate(centers.double().cpu().numpy()[b]):
            expected=np.flatnonzero(np.sum((xyz[b,:n]-center)**2,axis=1)<.4**2)[:4].tolist()
            expected+= [expected[0]]*(4-len(expected))
            assert groups[b,i].tolist()==expected
    counts=torch.tensor([10,27],device=device)
    g=torch.randn((2,27,32,3),device=device)*.15
    valid=torch.arange(27,device=device)[None]<counts[:,None]
    changed=g.clone();changed[~valid]=12345
    shuffled=g.clone()
    for b,n in enumerate(counts.tolist()):
        cloud=g[b,:n].reshape(-1,3)
        shuffled[b,:n]=cloud[torch.randperm(len(cloud),device=device)].reshape(n,32,3)
    h=build_hierarchy(g,counts); hc=build_hierarchy(changed,counts); hs=build_hierarchy(shuffled,counts)
    for key in h:
        assert torch.equal(h[key],hc[key]), key
        cached=h[key].cpu().numpy().astype(np.uint16).astype(np.int64)
        assert np.array_equal(cached,h[key].cpu().numpy())
    assert torch.equal(index_points(g.reshape(2,-1,3),h['order'])[:1,:320],
                       index_points(shuffled.reshape(2,-1,3),hs['order'])[:1,:320])
    model=make_model(.15).to(device).eval()
    with torch.no_grad():
        expected=model((g,counts,h))
        assert torch.equal(expected,model((changed,counts,hc)))
        assert torch.allclose(expected,model((shuffled,counts,hs)),atol=2e-6,rtol=2e-6)
        padded=F.pad(g,(0,0,0,0,0,3)); hp=build_hierarchy(padded,counts)
        assert torch.equal(expected,model((padded,counts,hp)))
    a=copy.deepcopy(model).train();b=copy.deepcopy(model).train()
    torch.manual_seed(91722); left=a((g,counts,h))
    torch.manual_seed(91722); right=b((changed,counts,hc))
    assert torch.equal(left,right)
    for key,value in a.state_dict().items():assert torch.equal(value,b.state_dict()[key]),key
    del a,b
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model((g,counts,h)),torch.tensor([0,1],device=device))
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    model.eval()
    with torch.no_grad():
        out=model((g,counts,h))
        assert torch.allclose(out,model((shuffled,counts,hs)),atol=2e-6,rtol=2e-6)
        assert torch.allclose(out[:1],model((g[:1],counts[:1],{k:v[:1] for k,v in h.items()})),atol=2e-6,rtol=2e-6)
    result=dict(complete=True,identity=identity(config),device=device,parameters=PARAMETERS,
        independent_fps_and_ball_reference=True,padding_excluded_in_sampling_and_training_batchnorm=True,
        point_permutation_invariance=True,uint16_cache_lossless=True,finite_forward_backward=True)
    write(Path(spec['output'])/f'preflight_{device}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='identity'}),flush=True)


def pilot(spec, config):
    deterministic('cuda');torch.manual_seed(91723);root=Path(spec['output'])
    geometries=[];counts=[];labels=[]
    for flow in spec['flows']:
        folder=root/'physical'/flow['name']/'train';g=np.load(folder/'geometry.npy',mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            rows=np.concatenate([np.flatnonzero(z['labels']==cls)[:8] for cls in (0,1)])
            geometries.append(np.array(g[rows]));counts.append(z['counts'][rows]);labels.append(z['labels'][rows])
    g=torch.as_tensor(np.concatenate(geometries),device='cuda')
    c=torch.as_tensor(np.concatenate(counts).astype(np.int64),device='cuda'); y=np.concatenate(labels)
    torch.cuda.synchronize();start=time.perf_counter();h=build_hierarchy(g,c);torch.cuda.synchronize()
    encode_seconds=time.perf_counter()-start
    # Cached grouping must be independent of unrelated samples in the encoding batch.
    single=build_hierarchy(g[:1],c[:1])
    assert all(torch.equal(single[k],v[:1]) for k,v in h.items())
    model=make_model(.15).cuda().eval();xx=(g,c,h);yy=torch.as_tensor(y,device='cuda',dtype=torch.long)
    with torch.no_grad():initial=float(F.cross_entropy(model(xx),yy))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);start=time.perf_counter()
    for _ in range(100):
        model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(xx),yy)
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    model.eval()
    with torch.no_grad():
        logits=model(xx);final=float(F.cross_entropy(logits,yy));prob=logits.softmax(-1)[:,1].cpu().numpy()
    assert final<initial,(initial,final)
    big=(g.repeat(4,1,1,1),c.repeat(4),{k:v.repeat(4,*([1]*(v.ndim-1))) for k,v in h.items()})
    elapsed=time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(3):
        model.train();optimizer.zero_grad(set_to_none=True);F.cross_entropy(model(big),yy.repeat(4)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    torch.cuda.synchronize();step_seconds=(time.perf_counter()-start)/3
    result=dict(complete=True,identity=identity(config),train_only=True,samples=32,parameters=PARAMETERS,
        initial_loss=initial,final_loss=final,training_metrics=metrics(y,prob),fit100_seconds=elapsed,
        hierarchy_seconds=encode_seconds,cache_batch_independent=True,batch128_forward_backward=True,
        batch128_step_seconds=step_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        approximate_training_only_epoch_seconds=step_seconds*np.ceil(spec['expected_counts']['train']/128))
    write(root/'pilot.json',result);print(json.dumps({k:v for k,v in result.items() if k!='identity'}),flush=True)


def encode(spec, config, index):
    deterministic('cuda');root=Path(spec['output']);name=spec['flows'][index]['name']
    assert json.loads((root/'pilot.json').read_text())['complete']
    report=engine.physical_reports(spec)[name];records={};started=time.perf_counter()
    for split in SPLITS:
        folder=root/'physical'/name/split
        for file,digest in report['splits'][split]['files'].items():assert sha(folder/file)==digest
        g=np.load(folder/'geometry.npy',mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:counts=z['counts'].astype(np.int64)
        needed=sum(int(np.prod(shape))*2*len(g) for shape in CACHE_SHAPES.values())
        assert shutil.disk_usage(root).free>needed+4*2**30
        dest=root/'encoded'/name/split;dest.mkdir(parents=True,exist_ok=False)
        arrays={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype='uint16',shape=(len(g),*shape)) for k,shape in CACHE_SHAPES.items()}
        batch=spec['pointnetplusplus']['encoding_batch_size']
        for first in range(0,len(g),batch):
            sl=slice(first,first+batch);gg=torch.as_tensor(np.array(g[sl]),device='cuda');cc=torch.as_tensor(counts[sl],device='cuda')
            h=build_hierarchy(gg,cc)
            for key,value in h.items():
                assert value.min()>=0 and value.max()<864
                arrays[key][sl]=value.cpu().numpy().astype(np.uint16)
            if first%4992==0:print('hierarchy',name,split,first,len(g),'seconds',time.perf_counter()-started,flush=True)
        for value in arrays.values():value.flush()
        del arrays
        records[split]=dict(samples=len(g),files={f.name:sha(f) for f in dest.glob('*.npy')},index_dtype='uint16')
        write(dest/'manifest.json',records[split])
    write(root/'encoding'/f'{index}.json',dict(complete=True,identity=identity(config),splits=records,seconds=time.perf_counter()-started))


if __name__=='__main__':main()
