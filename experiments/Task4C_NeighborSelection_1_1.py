"""Nearest/FPS neighbor ablation on frozen physical bundles and p35 training."""
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
from FMT_Utils.Task4C_NeighborSelection_1_1 import STRATEGIES, encode as encode_tokens, neighbor_indices, selection_statistics
from FMT_Utils.FMT_P35_NormFrequency_3_1 import encode as frozen_encode
from FMT_Utils.FMT_V8_Search_2_1 import task4_model
from experiments import Task4C_BottomDensity_1_2 as source
from experiments import Task4C_BottomDensity_1_3 as reuse_source
from experiments import Task4C_BottomDensity_1_1 as merge_source
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine

CONFIG = 'config/Ablation_Task4C_NeighborSelection_1.1.json'
SPLITS = engine.SPLITS
write, sha, append_locked, metrics, deterministic = engine.write, engine.sha, engine.append_locked, engine.metrics, engine.deterministic
normalization = engine.normalization


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    assert sha(definition['base_config']) == definition['base_config_sha256']
    spec = source.load_spec(definition['base_config'])
    allowed = {'version', 'user_version', 'execution_revision', 'output', 'base_config', 'base_config_sha256',
        'source_output', 'source_scientific_commit', 'dataset_policy', 'methods', 'nearest_reference',
        'selection', 'model_parameters', 'encoding_batch_size', 'data_offset_audit_tolerance'}
    assert set(definition) == allowed
    spec.update(definition)
    assert spec['methods'] == list(STRATEGIES[1:])
    assert spec['model_parameters']['p35'] == 76738 and spec['encoding_batch_size'] == 32
    assert spec['dataset_policy'] == 'reuse_all_source_geometry_seeds_metadata_without_sampling'
    assert spec['selection']['neighbors'] == 6 and spec['selection']['same_for_train_validation_test']
    return spec


def identity(config):
    result = source.identity(config)
    for name in ('experiments/Task4C_NeighborSelection_1_1.py', 'FMT_Utils/Task4C_NeighborSelection_1_1.py',
                 'ibex_bash/task4c_neighbor_selection_1p1.sh', 'experiments/Task4C_BottomDensity_1_3.py',
                 'config/Ablation_Task4C_BottomDensity_1.2.json'):
        result['sources'][name] = sha(name)
    return result


def run_engine(name, *args):
    fn = getattr(engine, name)
    return FunctionType(fn.__code__, dict(engine.__dict__, identity=identity), argdefs=fn.__defaults__)(*args)


def reuse(spec, config):
    fn = reuse_source.reuse
    FunctionType(fn.__code__, dict(reuse_source.__dict__, identity=identity), argdefs=fn.__defaults__)(spec, config)
    root = Path(spec['output']); results = {}
    for flow in spec['flows']:
        for split in SPLITS:
            folder = root/'physical'/flow['name']/split
            seeds = np.load(folder/'seeds.npy', mmap_mode='r')
            with np.load(folder/'metadata.npz') as z: m = {k:z[k] for k in z.files}
            counts = m['counts'].astype(np.int64)
            assert counts.min() >= 10 and counts.max() <= 27
            center_survived = 0; maximum_error = 0.
            for first in range(0, len(seeds), 2048):
                sl = slice(first, first+2048); c = counts[sl]
                valid = np.arange(27)[None] < c[:, None]
                world = np.asarray(seeds[sl], np.float64)*m['radius'][sl,None,None]+m['centroid'][sl,None,:]
                offset = (world-m['center'][sl,None,:])/m['neighbor_distance'][sl,None,None]
                integer = np.rint(offset).astype(np.int64)
                maximum_error = max(maximum_error, float(np.abs(offset-integer)[valid].max()))
                assert np.abs(integer[valid]).max() <= 1
                codes = (integer[...,0]+1)*9+(integer[...,1]+1)*3+integer[...,2]+1
                for row, n in enumerate(c): assert len(np.unique(codes[row,:n])) == n
                center_survived += int(((integer == 0).all(-1) & valid).any(-1).sum())
            assert maximum_error < spec['data_offset_audit_tolerance'], maximum_error
            results[flow['name']+'/'+split] = dict(samples=len(counts),
                valid_line_count_min=int(counts.min()), valid_line_count_max=int(counts.max()),
                valid_line_count_mean=float(counts.mean()), line_count_histogram=np.bincount(counts,minlength=28).tolist(),
                original_center_line_survived=center_survived, stencil_max_error=maximum_error,
                class_counts=np.bincount(m['labels'],minlength=2).tolist(), scale_counts=np.bincount(m['scale_id']).tolist())
    write(root/'seed_stencil_audit.json', dict(complete=True, identity=identity(config), splits=results,
        scope='All valid cached seeds in all splits are unique members of the same 3x3x3 stencil.'))
    print(json.dumps(results), flush=True)


def preflight(spec, config, device):
    deterministic(device); torch.manual_seed(91711)
    axis = torch.arange(-1, 2, device=device, dtype=torch.float32)
    seeds = torch.cartesian_prod(axis, axis, axis)[None].repeat(2,1,1)
    counts = torch.tensor([27,10], device=device)
    t = torch.linspace(-1,1,32,device=device)
    geometry = seeds[:,:,None,:]+torch.stack((t, .15*t*t, .2*torch.sin(3*t)), -1)[None,None]
    definition = spec['encoding']['fmt_definition']
    reference = frozen_encode(geometry, definition, seeds=seeds, counts=counts)
    assert torch.equal(encode_tokens(geometry,seeds,counts,definition,'nearest6'),reference)
    central_width = 4*definition['frequencies']-1+12*definition['frequencies']
    for strategy in STRATEGIES:
        selected = neighbor_indices(seeds,counts,strategy)
        assert torch.equal(selected,neighbor_indices(seeds,counts,strategy))
        # Independent scalar greedy implementation, including anchor initialization and ties.
        xyz = seeds.cpu().numpy()
        for b,n in enumerate(counts.cpu().tolist()):
            for anchor in range(n):
                distance = np.linalg.norm(xyz[b,:n,None]-xyz[b,None,:n],axis=-1)
                candidates = [i for i in range(n) if i != anchor]
                near = sorted(candidates,key=lambda i:(distance[anchor,i],i))
                expected = near[:6] if strategy == 'nearest6' else (near[:3] if strategy == 'nearest3_fps3' else [])
                while len(expected) < 6:
                    pool = [i for i in candidates if i not in expected]
                    expected.append(max(pool,key=lambda i:(min(distance[i,j] for j in [anchor]+expected),-i)))
                assert selected[b,anchor].cpu().tolist() == expected
        features = encode_tokens(geometry,seeds,counts,definition,strategy)
        assert torch.equal(features[...,:central_width],reference[...,:central_width])
        changed = seeds.clone(); changed[1,10:] = 1000
        assert torch.equal(selected[1,:10],neighbor_indices(changed,counts,strategy)[1,:10])
        assert torch.isfinite(features).all()
    near = neighbor_indices(seeds,counts,'nearest6')[0,13]
    far = neighbor_indices(seeds,counts,'fps6')[0,13]
    assert torch.all(seeds[0,near].norm(dim=-1) == 1)
    assert float(seeds[0,far].norm(dim=-1).max()) > 1.7
    model = task4_model(141,'h0').to(device)
    mask = torch.arange(27,device=device)[None] < counts[:,None]
    loss = F.cross_entropy(model(torch.cat((features,mask[...,None].float()),-1)),torch.tensor([0,1],device=device))
    loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    parameters = sum(p.numel() for p in model.parameters()); assert parameters == 76738
    result = dict(complete=True,identity=identity(config),device=device,parameters=parameters,
        independent_greedy_reference=True,padding_excluded=True,six_unique_nonself_neighbors=True,
        frozen_nearest_tokens_exact=True,central95_features_unchanged=True,finite_backward=True)
    write(Path(spec['output'])/f'preflight_{device}.json',result); print(json.dumps(result),flush=True)


def pilot(spec, config):
    deterministic('cuda'); root = Path(spec['output']); results = {}
    assert json.loads((root/'seed_stencil_audit.json').read_text())['complete']
    for flow in spec['flows']:
        folder = root/'physical'/flow['name']/'train'
        g = torch.as_tensor(np.array(np.load(folder/'geometry.npy',mmap_mode='r')[:32]),device='cuda')
        s = torch.as_tensor(np.array(np.load(folder/'seeds.npy',mmap_mode='r')[:32]),device='cuda')
        with np.load(folder/'metadata.npz') as z: c = torch.as_tensor(z['counts'][:32].astype(np.int64),device='cuda')
        cache = Path(spec['source_output'])/'encoded'/flow['name']/'train'
        expected = json.loads((cache/'manifest.json').read_text())['files']['p35.npy']
        assert sha(cache/'p35.npy') == expected
        stored = np.load(cache/'p35.npy',mmap_mode='r')[:32]
        mask = torch.arange(27,device='cuda')[None] < c[:,None]
        nearest = encode_tokens(g,s,c,spec['encoding']['fmt_definition'],'nearest6')
        assert np.array_equal(torch.cat((nearest,mask[...,None].float()),-1).cpu().numpy(),stored)
        result = dict(frozen_nearest_cache_byte_equal=True,source_cache_sha256=expected,methods={})
        for strategy in STRATEGIES:
            features = encode_tokens(g,s,c,spec['encoding']['fmt_definition'],strategy)
            assert torch.equal(features[...,:95],nearest[...,:95]) and torch.isfinite(features).all()
            result['methods'][strategy] = selection_statistics(s,c,neighbor_indices(s,c,strategy))
        results[flow['name']] = result
    write(root/'pilot.json',dict(complete=True,identity=identity(config),train_only=True,samples_per_flow=32,flows=results))
    print(json.dumps(results),flush=True)


def encode(spec, config, index):
    deterministic('cuda'); root = Path(spec['output']); name = spec['flows'][index]['name']
    assert json.loads((root/'pilot.json').read_text())['complete']
    report = engine.physical_reports(spec)[name]; records = {}; started = time.perf_counter()
    for split in SPLITS:
        folder = root/'physical'/name/split
        for filename,digest in report['splits'][split]['files'].items(): assert sha(folder/filename) == digest
        g = np.load(folder/'geometry.npy',mmap_mode='r'); s = np.load(folder/'seeds.npy',mmap_mode='r')
        with np.load(folder/'metadata.npz') as z: counts = z['counts'].astype(np.int64)
        dest = root/'encoded'/name/split; dest.mkdir(parents=True,exist_ok=False)
        tokens = {method:np.lib.format.open_memmap(dest/f'{method}.npy',mode='w+',dtype='float32',shape=(len(g),27,142)) for method in spec['methods']}
        stats = {method:{} for method in STRATEGIES}
        for first in range(0,len(g),spec['encoding_batch_size']):
            sl = slice(first,first+spec['encoding_batch_size'])
            gg = torch.as_tensor(np.array(g[sl]),device='cuda'); ss = torch.as_tensor(np.array(s[sl]),device='cuda')
            cc = torch.as_tensor(counts[sl],device='cuda'); mask = torch.arange(27,device='cuda')[None] < cc[:,None]
            for method in STRATEGIES:
                if method in tokens:
                    x = encode_tokens(gg,ss,cc,spec['encoding']['fmt_definition'],method)
                    tokens[method][sl] = torch.cat((x,mask[...,None].float()),-1).cpu().numpy()
                stats_batch = selection_statistics(ss,cc,neighbor_indices(ss,cc,method))
                for key,value in stats_batch.items(): stats[method][key] = stats[method].get(key,0)+value
            if first%10016 == 0: print('encoded',name,split,first,len(g),flush=True)
        for value in tokens.values(): value.flush()
        del tokens
        records[split] = dict(samples=len(g),files={f.name:sha(f) for f in dest.glob('*.npy')},neighbor_statistics=stats)
        write(dest/'manifest.json',records[split])
    write(root/'encoding'/f'{index}.json',dict(complete=True,identity=identity(config),splits=records,seconds=time.perf_counter()-started))


def load_parts(spec, role, method):
    root = Path(spec['output']); parts=[]; ids=[]; labels=[]; flows=[]; instances=[]
    assert method in spec['methods']
    for fi,flow in enumerate(spec['flows']):
        src = root/'physical'/flow['name']/role; cache = root/'encoded'/flow['name']/role
        with np.load(src/'metadata.npz') as z: y=z['labels'].astype(np.int64); owner=z['instance'].copy()
        manifest=json.loads((cache/'manifest.json').read_text()); file=cache/f'{method}.npy'
        assert sha(file)==manifest['files'][file.name]
        parts.append(dict(tokens=np.load(file,mmap_mode='r')))
        ids.extend((fi,i) for i in range(len(y))); labels.append(y); instances.append(owner); flows.extend([fi]*len(y))
    return parts,np.asarray(ids,np.int64),np.concatenate(labels),np.asarray(flows),np.concatenate(instances)


def get_batch(parts, ids, method, norm, device='cuda'):
    return engine.get_batch(parts,ids,'p35',norm,device)


def predict(model, dataset, method, norm, batch):
    fn=engine.predict
    return FunctionType(fn.__code__,dict(engine.__dict__,get_batch=get_batch),argdefs=fn.__defaults__)(model,dataset,method,norm,batch)


def merge(spec, config):
    fn=merge_source.merge
    FunctionType(fn.__code__,dict(merge_source.__dict__,identity=identity,run_engine=run_engine),argdefs=fn.__defaults__)(spec,config)
    root=Path(spec['output']); reference=[]
    for seed in spec['training']['seeds']:
        folder=Path(spec['source_output'])/'runs/p35'/f'seed{seed}'
        result=json.loads((folder/'result.json').read_text())
        assert result['complete'] and result['parameters']==76738
        assert result['identity']['git_commit']==spec['source_scientific_commit']
        assert result['identity']['config_sha256']==spec['base_config_sha256']
        for role in SPLITS:
            file=folder/f'{role}_predictions.npz'; assert sha(file)==result['predictions'][role]
            with np.load(file) as z:
                assert metrics(z['labels'],z['probability'])==result['metrics'][role]['combined']
                for fi,flow in enumerate(spec['flows']):
                    with np.load(root/'physical'/flow['name']/role/'metadata.npz') as m:
                        selected=z['flow_index']==fi; ids=z['row_in_split'][selected]
                        assert np.array_equal(np.sort(ids),np.arange(len(m['labels'])))
                        assert np.array_equal(z['labels'][selected],m['labels'][ids])
        reference.append(result)
    summary=json.loads((root/'summary.json').read_text()); values=[r['metrics']['test']['combined']['f1'] for r in reference]
    summary['nearest_reference']=dict(method='nearest6',parameters=76738,test_f1_mean=float(np.mean(values)),
        test_f1_std=float(np.std(values,ddof=1)),source_commit=spec['source_scientific_commit'],
        predictions=[r['predictions'] for r in reference])
    assert all(r['parameters']==76738 for r in summary['methods'])
    write(root/'summary.json',summary)
    file=root/'viewer_package/manifest.json'; manifest=json.loads(file.read_text())
    manifest['evidence_note']='Only six-neighbor selection differs; all frozen fivefold bundles, splits and training rules are unchanged. Seed 96611.'
    write(file,manifest)


def submit(spec, config):
    root=Path(spec['output']).resolve(); assert shutil.disk_usage(Path.cwd()).free>20*2**30
    root.mkdir(parents=True,exist_ok=False); (root/'logs').mkdir(); write(root/'config.frozen.json',spec)
    dependency=None
    phases=[('preflight',None,True,'00:20:00'),('reuse',None,False,'00:45:00'),('pilot',None,True,'00:20:00'),
        ('encode','0-1%2',True,'02:00:00'),('train','0-5%6',True,'08:00:00'),('merge',None,False,'01:00:00')]
    for phase,array,gpu,limit in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time='+limit,
            '--job-name=t4c-neighbors-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array: command+=['--array='+array]
        if gpu: command+=['--gres=gpu:1','--constraint=v100']
        if dependency: command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_neighbor_selection_1p1.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),command=command,
            dependency=dependency,expected_device='V100' if gpu else 'CPU',identity=identity(config))
        append_locked(root/'submissions.jsonl',json.dumps(row)+'\n'); print(json.dumps(row),flush=True); dependency=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','reuse','pilot','encode','train','merge','submit','runtime'))
    parser.add_argument('--config',default=CONFIG); parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--runtime-phase')
    parser.add_argument('--state'); parser.add_argument('--exit-code',type=int)
    args=parser.parse_args(); spec=load_spec(args.config)
    if args.phase=='preflight': preflight(spec,args.config,args.device)
    elif args.phase in ('encode','train'): globals()[args.phase](spec,args.config,args.index)
    elif args.phase=='runtime': run_engine('runtime',spec,args.config,args.runtime_phase,args.state,args.exit_code)
    else: globals()[args.phase](spec,args.config)


# The training loop below is copied from the frozen 1.2 engine with only the two
# method dispatch lines replaced: every new method uses the same p35 model and normalization.


def train(spec,config,index):
    deterministic('cuda');seeds=spec['training']['seeds'];method=spec['methods'][index//len(seeds)];seed=seeds[index%len(seeds)]
    torch.manual_seed(seed);rng=np.random.default_rng(seed);options=spec['training'];batch=options['batch_size']
    root=Path(spec['output']);folder=root/'runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    training=load_parts(spec,'train',method);validation=load_parts(spec,'validation',method)
    parts,ids,y,_,_=training;assert len(y)==spec['expected_counts']['train']
    norm=normalization(parts)
    model=task4_model(141,'h0').cuda()
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


if __name__=='__main__':main()
