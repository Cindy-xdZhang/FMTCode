"""Corrected task4-c-v9.15_v2: restore 4.14 labels, local splits and validation selection."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, shutil, socket, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from FMT_Utils.Task4C_InstanceCoverage_9_15_v2 import write, sparse_voxels, dense_sparse_batch, validate_spec, prepare as prepare_physical
from FMT_Utils.FMT_V8_Search_2_1 import task4_model
from FMT_Utils.FMT_P35_NormFrequency_3_1 import candidates, encode as encode_p35, apply_normalizer
from experiments.Task4C_HairpinBinary_2_1 import append_locked
CONFIG='config/mainExp_Task4C_InstanceCoverage_9.15_v2.json'
SPLITS=('train','validation','test')


def load_spec(config):
    spec=json.loads(Path(config).read_text(encoding='utf-8-sig'));validate_spec(spec)
    reference=json.loads(Path(spec['reference_config']).read_text())
    base=json.loads(Path(reference['base_config']).read_text())
    for key in ('flows','labels','bundles','input_root'):
        assert spec[key]==base[key], f'Frozen 4.14 {key} changed'
    assert spec['local_split']==reference['local_split']
    for key,value in reference['sampling'].items():
        if key!='per_flow_counts':assert spec['sampling'][key]==value
    for key,value in reference['training'].items():assert spec['training'][key]==value
    assert spec['encoding']['fmt_definition']==next(c for c in candidates() if c['id']=='n0_k06')
    return spec


def prepare(spec,config,index,pilot=False,input_root=None):
    prepare_physical(spec,config,index,identity,pilot,input_root)


def physical_reports(spec,pilot=False):
    root=Path(spec['output'])/('pilot' if pilot else '')/'physical'
    return {flow['name']:json.loads((root/flow['name']/'preparation.json').read_text()) for flow in spec['flows']}


def audit(spec,config):
    root=Path(spec['output']);reports=physical_reports(spec);totals={s:0 for s in SPLITS};summary={}
    for flow in spec['flows']:
        name=flow['name'];report=reports[name]
        assert report['identity']['config_sha256']==sha(config)
        assert report['spatial_cell_and_physical_length_audit_passed']
        catalog={r['head_component']:r for r in report['catalog']['eligible_labels']}
        heads={};instances={};data={};summary[name]={}
        for split in SPLITS:
            src=root/'physical'/name/split;entry=report['splits'][split]
            for filename,digest in entry['files'].items():assert sha(src/filename)==digest
            with np.load(src/'metadata.npz') as z:m={k:z[k] for k in z.files}
            data[split]=m;g=np.load(src/'geometry.npy',mmap_mode='r')
            assert len(g)==spec['sampling']['per_flow_counts'][split]
            assert np.all(m['labels']==[catalog[int(h)]['label'] for h in m['head_component']])
            assert np.all(m['instance']==[catalog[int(h)]['instance'] for h in m['head_component']])
            n=m['counts'];valid=np.arange(27)[None]<n[:,None];assert np.all((n>=10)&(n<=27))
            assert np.isfinite(g).all() and np.all(g[~valid]==0)
            lo,hi=spec['integration']['length_ranges'][name];eps=hi*2e-6
            half=m['half_arc_lengths'][valid]
            assert np.all((half>=lo-eps)&(half<=hi+eps))
            for start in range(0,len(g),512):
                sl=slice(start,start+512);x=np.asarray(g[sl],np.float64);mask=valid[sl]
                assert np.allclose(x.sum((1,2))/(n[sl,None]*32),0,atol=2e-6)
                assert np.allclose(np.linalg.norm(x,axis=-1).max((1,2)),1,atol=2e-6)
                arcs=np.linalg.norm(np.diff(x,axis=2),axis=-1).sum(2)*m['radius'][sl,None]
                assert np.all((arcs[mask]>=2*lo-eps)&(arcs[mask]<=2*hi+eps))
            heads[split]=set(m['head_component'].tolist());instances[split]=set(m['instance'][m['instance']>=0].tolist())
            totals[split]+=len(g)
            summary[name][split]=dict(samples=len(g),class_counts=np.bincount(m['labels'],minlength=2).tolist(),
                positive_instances=sorted(instances[split]),head_count=len(heads[split]),
                actual_half_length_minmax=[float(half.min()),float(half.max())])
        for split in ('validation','test'):
            assert heads[split]<=heads['train'] and instances[split]<=instances['train']
        summary[name]['test_only_instances']=sorted(instances['test']-instances['train'])
    assert totals=={s:spec['expected_counts'][s] for s in SPLITS}
    result=dict(complete=True,identity=identity(config),counts=totals,whole_instance_holdout_count=0,
        labels='frozen_4.14_whole_head',training_selection='frozen_4.14_validation',flows=summary)
    write(root/'data_audit.json',result);print(json.dumps(result),flush=True)


def capacity(spec,config):
    deterministic('cuda');root=Path(spec['output']);total_bytes=samples=0
    for name,report in physical_reports(spec,pilot=True).items():
        for split in SPLITS:
            src=root/'pilot'/'physical'/name/split
            g=np.load(src/'geometry.npy',mmap_mode='r')
            with np.load(src/'metadata.npz') as m:counts=m['counts'].astype(np.int64)
            for first in range(0,len(g),32):
                sl=slice(first,first+32);gg=torch.as_tensor(np.array(g[sl]),device='cuda');cc=torch.as_tensor(counts[sl],device='cuda')
                for resolution in spec['encoding']['voxel_resolutions']:
                    total_bytes+=sum(a.nbytes for a in sparse_voxels(gg,cc,resolution))
                samples+=len(gg)
    estimated=total_bytes/samples*spec['expected_counts']['total']
    required=estimated*2+spec['expected_counts']['total']*(27*32*3*4*2+27*142*4+12000)+10*2**30
    available=shutil.disk_usage(root).free
    result=dict(complete=available>=required,identity=identity(config),pilot_samples=samples,
        estimated_sparse_bytes=estimated,required_bytes=required,available_bytes=available)
    write(root/'capacity.json',result)
    assert result['complete'],'Insufficient storage; no existing data removed'
    print(json.dumps(result),flush=True)


def encode(spec,config,index):
    deterministic('cuda');root=Path(spec['output']);name=spec['flows'][index]['name']
    report=physical_reports(spec)[name];a=json.loads((root/'data_audit.json').read_text());assert a['complete']
    records={}
    for split in SPLITS:
        assert shutil.disk_usage(root).free>10*2**30
        src=root/'physical'/name/split
        for filename,digest in report['splits'][split]['files'].items():assert sha(src/filename)==digest
        g=np.load(src/'geometry.npy',mmap_mode='r');seeds=np.load(src/'seeds.npy',mmap_mode='r')
        with np.load(src/'metadata.npz') as z:counts=z['counts'].astype(np.int64)
        dest=root/'encoded'/name/split;dest.mkdir(parents=True,exist_ok=False)
        tokens=np.lib.format.open_memmap(dest/'p35.npy',mode='w+',dtype=np.float32,shape=(len(g),27,142))
        sparse={r:dict(offsets=[np.array([0],np.int64)],indices=[],values=[]) for r in spec['encoding']['voxel_resolutions']}
        for first in range(0,len(g),32):
            sl=slice(first,first+32);gg=torch.as_tensor(np.array(g[sl]),device='cuda')
            ss=torch.as_tensor(np.array(seeds[sl]),device='cuda');cc=torch.as_tensor(counts[sl],device='cuda')
            x=encode_p35(gg,spec['encoding']['fmt_definition'],seeds=ss,counts=cc)
            mask=(torch.arange(27,device='cuda')[None]<cc[:,None]).to(x.dtype)
            tokens[sl]=torch.cat((x,mask[...,None]),-1).cpu().numpy()
            for resolution,p in sparse.items():
                off,ids,values=sparse_voxels(gg,cc,resolution)
                p['offsets'].append(off[1:]+p['offsets'][-1][-1]);p['indices'].append(ids);p['values'].append(values)
        tokens.flush();del tokens
        for resolution,p in sparse.items():
            dst=dest/f'r{resolution}';dst.mkdir()
            for key,arrays in p.items():np.save(dst/f'{key}.npy',np.concatenate(arrays))
        records[split]=dict(samples=len(g),files={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*.npy')})
        write(dest/'manifest.json',records[split]);print('encoded',name,split,len(g),flush=True)
    write(root/'encoding'/f'{index}.json',dict(complete=True,identity=identity(config),splits=records))


def load_parts(spec,role,method):
    root=Path(spec['output']);parts=[];ids=[];labels=[];flows=[];instances=[]
    for fi,flow in enumerate(spec['flows']):
        src=root/'physical'/flow['name']/role;cache=root/'encoded'/flow['name']/role
        with np.load(src/'metadata.npz') as z:y=z['labels'].astype(np.int64);owner=z['instance'].copy()
        manifest=json.loads((cache/'manifest.json').read_text())
        if method=='p35':
            assert sha(cache/'p35.npy')==manifest['files']['p35.npy']
            part=dict(tokens=np.load(cache/'p35.npy',mmap_mode='r'))
        else:
            folder=cache/f'r{int(method[4:])}';part={}
            for key in ('offsets','indices','values'):
                f=folder/f'{key}.npy';assert sha(f)==manifest['files'][str(f.relative_to(cache))]
                part[key]=np.load(f,mmap_mode='r')
        ids.extend((fi,i) for i in range(len(y)));labels.append(y);instances.append(owner)
        flows.extend([fi]*len(y));parts.append(part)
    return parts,np.asarray(ids,np.int64),np.concatenate(labels),np.asarray(flows),np.concatenate(instances)


def predict(model,dataset,method,norm,batch):
    parts,ids,y,_,_=dataset;prob=[];loss=0.;model.eval()
    with torch.no_grad():
        for start in range(0,len(y),batch):
            sl=slice(start,start+batch);logits=model(get_batch(parts,ids[sl],method,norm))
            yy=torch.as_tensor(y[sl],device='cuda',dtype=torch.long)
            loss+=float(F.cross_entropy(logits,yy,reduction='sum'));prob.append(logits.softmax(-1)[:,1].cpu().numpy())
    return np.concatenate(prob),loss/len(y)


def train(spec,config,index):
    deterministic('cuda');seeds=spec['training']['seeds'];method=spec['methods'][index//len(seeds)];seed=seeds[index%len(seeds)]
    torch.manual_seed(seed);rng=np.random.default_rng(seed);options=spec['training'];batch=options['batch_size']
    root=Path(spec['output']);folder=root/'runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    training=load_parts(spec,'train',method);validation=load_parts(spec,'validation',method)
    parts,ids,y,_,_=training;assert len(y)==spec['expected_counts']['train']
    norm=normalization(parts) if method=='p35' else None
    model=(task4_model(141,'h0') if method=='p35' else Conv915(options['dropout'])).cuda()
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


def merge(spec,config):
    root=Path(spec['output']);rows=[]
    for method in spec['methods']:
        for seed in spec['training']['seeds']:
            folder=root/'runs'/method/f'seed{seed}';r=json.loads((folder/'result.json').read_text());assert r['complete']
            assert r['execution_revision']=='restored_4.14' and r['identity']['config_sha256']==sha(config)
            for role in SPLITS:
                f=folder/f'{role}_predictions.npz';assert sha(f)==r['predictions'][role]
                with np.load(f) as z:
                    assert metrics(z['labels'],z['probability'])==r['metrics'][role]['combined']
                    from sklearn.metrics import f1_score
                    assert abs(f1_score(z['labels'],z['probability']>=.5)-r['metrics'][role]['combined']['f1'])<1e-12
                    for fi,flow in enumerate(spec['flows']):
                        with np.load(root/'physical'/flow['name']/role/'metadata.npz') as m:
                            chosen=z['flow_index']==fi;ids=z['row_in_split'][chosen]
                            assert len(ids)==len(m['labels']) and np.array_equal(np.sort(ids),np.arange(len(ids)))
                            assert np.array_equal(z['labels'][chosen],m['labels'][ids])
            rows.append(r)
    summary=[]
    for method in spec['methods']:
        subset=[r for r in rows if r['method']==method];values=[r['metrics']['test']['combined']['f1'] for r in subset]
        summary.append(dict(method=method,parameters=subset[0]['parameters'],test_f1_mean=float(np.mean(values)),
            test_f1_std=float(np.std(values,ddof=1)),mean_training_minutes=float(np.mean([r['training_seconds'] for r in subset])/60)))
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth'))
    export_viewer_package(spec,config,rows)
    write(root/'summary.json',dict(complete=True,identity=identity(config),execution_revision='restored_4.14',methods=summary))
    print(json.dumps(summary),flush=True)


def export_viewer_package(spec,config,rows):
    root=Path(spec['output']);dest=root/'viewer_package';dest.mkdir(parents=True,exist_ok=True)
    seed=spec['training']['seeds'][0]
    models=[dict(id=r['method'],label=r['method'],seed=seed,parameters=r['parameters'],
        version=spec['version'],metrics=r['metrics']) for r in rows if r['seed']==seed]
    manifest=dict(version=spec['version'],execution_revision=spec['execution_revision'],identity=identity(config),
        config=dict(splits=list(SPLITS),display_bundles=100,geometry_chunk_size=128,
            default_hairpin_count=100,default_nonhairpin_count=100),models=models,pending=[],flows=[],splits={},
        evidence_note='Corrected v2: frozen 4.14 whole-head labels and shared-instance spatial blocks; seed 96611.')
    for fi,flow in enumerate(spec['flows']):
        record=dict(flow)
        record.update(flow_sha256=sha(Path(spec['input_root'])/flow['flow']),gt_sha256=sha(Path(spec['input_root'])/flow['gt']))
        manifest['flows'].append(record)
        for role in SPLITS:
            src=root/'physical'/flow['name']/role
            with np.load(src/'metadata.npz') as z:m={k:z[k] for k in z.files}
            g=np.load(src/'geometry.npy');n=len(g)
            world=g*m['radius'][:,None,None,None]+m['centroid'][:,None,None,:]
            world[np.arange(27)[None]>=m['counts'][:,None]]=0
            pack={key:m[key] for key in ('counts','labels','center','head_component','instance','scale_id',
                'radius','neighbor_distance','ds','maxiteration','requested_half_length')}
            pack.update(geometry=world.astype(np.float32),row_ids=np.arange(n),requested_total_length=2*m['requested_half_length'])
            for model in models:
                with np.load(root/'runs'/model['id']/f'seed{seed}'/f'{role}_predictions.npz') as z:
                    selected=z['flow_index']==fi;ids=z['row_in_split'][selected]
                    p=np.empty(n,np.float32);p[ids]=z['probability'][selected];pack['p_'+model['id']]=p
            file=dest/f'{flow["name"]}_{role}.npz';np.savez_compressed(file,**pack)
            manifest['splits'][flow['name']+'/'+role]=dict(file=file.name,sha256=sha(file),selected=n,population=n,
                selection_uses_prediction_scores=False,class_counts=np.bincount(m['labels'],minlength=2).tolist())
    write(dest/'manifest.json',manifest)


def submit(spec,config):
    root=Path(spec['output']).resolve();assert shutil.disk_usage(Path.cwd()).free>10*2**30
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();write(root/'config.frozen.json',spec);previous=None
    phases=[('preflight',None,True,'00:20:00'),('pilot','0-1%2',False,'01:00:00'),('capacity',None,True,'00:30:00'),
        ('prepare','0-1%2',False,'12:00:00'),('audit',None,False,'01:00:00'),('encode','0-1%2',True,'03:00:00'),
        ('train','0-11%6',True,'2-00:00:00'),('merge',None,False,'00:30:00')]
    for phase,array,gpu,limit in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time='+limit,
            '--job-name=t4cv2-restored-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if array:command+=['--array='+array]
        if previous:command+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_instance_9p15_v2.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),dependency=previous,
            expected_device='V100' if gpu else 'CPU',command=command,identity=identity(config),execution_revision='restored_4.14')
        append_locked(root/'submissions.jsonl',json.dumps(row)+'\n');append_locked('docs/ibex_run_registry.md','\n- Task4C v2 restored '+json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);previous=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('pilot','prepare','audit','capacity','encode','train','merge','preflight','runtime','submit'))
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--input-root');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=load_spec(args.config)
    if args.phase in ('pilot','prepare'):prepare(spec,args.config,args.index,args.phase=='pilot',args.input_root)
    elif args.phase=='preflight':preflight(spec,args.config,args.device)
    elif args.phase=='runtime':runtime(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    elif args.phase in ('encode','train'):globals()[args.phase](spec,args.config,args.index)
    else:globals()[args.phase](spec,args.config)




def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(config):
    paths = [__file__, 'FMT_Utils/Task4C_InstanceCoverage_9_15_v2.py', 'FMT_Utils/Task4C_InstanceCoverage_9_15.py',
        'FMT_Utils/FMT_P35_NormFrequency_3_1.py', 'FMT_Utils/FMT_V8_Search_2_1.py',
        'FMT_Utils/Task4C_Encoders_4_15.py', 'FMT_Utils/DFT_FMT_3D.py',
        'FMT_Utils/Task4C_PaperBundles_3_1.py', 'FMT_Utils/Task4C_Multiscale_4_1.py',
        'FMT_Utils/Task4C_HairpinBinary_2_1.py', 'experiments/Task4C_PhysicalLength_4_14.py',
        'experiments/Task4C_ConvResolution_4_22.py',
        'experiments/Visualize_Task4C_Bundles_3D.py', 'experiments/templates/task4c_bundles.html',
        'ibex_bash/task4c_instance_9p15_v2.sh']
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        config_sha256=sha(config), sources={str(p): sha(p) for p in paths},
        hostname=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
        array_index=os.environ.get('SLURM_ARRAY_TASK_ID'))

def deterministic(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        if not torch.cuda.is_available() or 'V100' not in torch.cuda.get_device_name():
            raise ValueError('The frozen p35 nearest-neighbor encoding requires a V100')

class DeterministicPool2(nn.Module):
    """Same 2x2x2 adaptive averaging bins; no CUDA atomic backward reduction."""
    def forward(self, x):
        import itertools
        shape=x.shape[-3:]
        bins=[]
        for position in itertools.product(range(2),repeat=3):
            slices=tuple(slice(i*n//2,((i+1)*n+1)//2) for i,n in zip(position,shape))
            bins.append(x[(slice(None),slice(None),*slices)].mean(dim=(-3,-2,-1)))
        return torch.stack(bins,-1).reshape(*x.shape[:2],2,2,2)

class DeterministicMaxPool2(nn.Module):
    """Nonoverlapping 2x2x2 max bins, preserving the first-index tie rule."""
    def forward(self,x):
        d,h,w=(n//2 for n in x.shape[-3:])
        cells=x[...,:2*d,:2*h,:2*w].reshape(*x.shape[:2],d,2,h,2,w,2)
        return cells.permute(0,1,2,4,6,3,5,7).flatten(5).max(-1).values

class Conv915(nn.Module):
    """Same spatial blocks; 70-wide head keeps 90-95% of current p35 parameters."""
    def __init__(self, dropout=.15):
        super().__init__()
        layers = []
        for i, (cin, cout) in enumerate(((4, 12), (12, 24), (24, 48))):
            layers.extend((nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
                nn.Dropout3d(dropout/2), DeterministicMaxPool2() if i < 2 else DeterministicPool2()))
        self.network = nn.Sequential(*layers, nn.Flatten(), nn.Linear(384, 70), nn.LayerNorm(70),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(70, 64), nn.Linear(64, 2))

    def forward(self, x):
        return self.network(x)

def metrics(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, bool); pred = np.asarray(p) >= .5
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    return dict(f1=2*tp/max(1, 2*tp+fp+fn), precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
        accuracy=(tp+tn)/len(y), average_precision=float(average_precision_score(y, p)),
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn, samples=len(y))

def normalization(parts):
    count = 0; mean = np.zeros(141, np.float64); m2 = mean.copy()
    for part in parts:
        x = np.asarray(part['tokens']); valid = x[..., -1] > .5
        values = x[..., :-1][valid]
        local_mean = values.mean(0, dtype=np.float64)
        local_m2 = ((values-local_mean)**2).sum(0, dtype=np.float64)
        delta = local_mean-mean; n = len(values)
        m2 += local_m2+delta*delta*count*n/(count+n)
        mean += delta*n/(count+n); count += n
    std = np.sqrt(m2/count); std[std < 1e-8] = 1
    return mean, std

def get_batch(parts, ids, method, norm, device='cuda'):
    if method != 'p35':
        return dense_sparse_batch(parts, ids, int(method[4:]), device)
    x = np.stack([parts[p]['tokens'][i] for p, i in ids])
    value = apply_normalizer(x[..., :-1], dict(kind='zscore', mean=norm[0], std=norm[1]), x[..., -1:])
    return torch.as_tensor(np.concatenate((value, x[..., -1:]), -1), device=device)

def preflight(spec, config, device):
    deterministic(device); torch.manual_seed(91500)
    pool_checks={}
    for size in (8,9,12):
        values=torch.randn(2,3,size,size,size,dtype=torch.float64)
        reference=values.clone().requires_grad_(True)
        expected=F.adaptive_avg_pool3d(reference,2)
        weight=torch.randn_like(expected);(expected*weight).sum().backward()
        actual=values.to(device).requires_grad_(True)
        output=DeterministicPool2()(actual);(output*weight.to(device)).sum().backward()
        assert torch.allclose(output.cpu(),expected.detach(),atol=1e-12,rtol=1e-12)
        assert torch.allclose(actual.grad.cpu(),reference.grad,atol=1e-12,rtol=1e-12)
        pool_checks[str(size)]=dict(forward_max_error=float((output.cpu()-expected.detach()).abs().max().detach()),
            backward_max_error=float((actual.grad.cpu()-reference.grad).abs().max()))
        # Rounded values deliberately exercise equal maxima and odd-size cropping.
        values=torch.randn(2,3,size,size,size,dtype=torch.float64).round()
        reference=values.clone().requires_grad_(True);expected=F.max_pool3d(reference,2)
        weight=torch.randn_like(expected);(expected*weight).sum().backward()
        actual=values.to(device).requires_grad_(True);output=DeterministicMaxPool2()(actual)
        (output*weight.to(device)).sum().backward()
        assert torch.equal(output.cpu(),expected.detach()) and torch.equal(actual.grad.cpu(),reference.grad)
        pool_checks[str(size)]['max_forward_and_backward_exact_including_ties']=True
    from experiments.Task4C_ConvResolution_4_22 import reference_voxel
    from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_voxels
    t = torch.linspace(-1, 1, 32, device=device); geometry = torch.zeros((2, 27, 32, 3), device=device)
    for i in range(27):
        geometry[:, i, :, 0] = .7*t
        geometry[:, i, :, 1] = .17*torch.sin(3*t+i*.03)+i*.008
        geometry[:, i, :, 2] = .18*torch.cos(2*t+i*.04)-i*.006
    counts = torch.tensor([27, 19], device=device); geometry[1, 19:] = 0
    seeds = geometry[:, :, 16].clone()
    tokens = encode_p35(geometry, spec['encoding']['fmt_definition'], seeds=seeds, counts=counts)
    mask = (torch.arange(27, device=device)[None] < counts[:, None]).to(geometry.dtype)
    packed = torch.cat((tokens, mask[..., None]), -1).cpu().numpy()
    parts = [dict(tokens=packed)]
    norm = normalization(parts)
    from FMT_Utils.FMT_P35_NormFrequency_3_1 import fit_normalizer
    reference = fit_normalizer(packed[..., :-1], packed[..., -1:], 'zscore')
    assert np.allclose(norm[0], reference['mean'], atol=1e-12)
    assert np.allclose(norm[1], reference['std'], atol=1e-12)
    fmt_input = get_batch(parts, [(0,0),(0,1)], 'p35', norm, device)
    expected = apply_normalizer(packed[..., :-1], reference, packed[..., -1:])
    assert np.allclose(fmt_input[..., :-1].cpu().numpy(), expected, atol=1e-6)
    fmt_model = task4_model(141, 'h0').to(device)
    fmt_optimizer = torch.optim.AdamW(fmt_model.parameters(), lr=.001)
    for _ in range(2):
        fmt_optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(fmt_model(fmt_input), torch.tensor([0, 1], device=device))
        assert torch.isfinite(loss); loss.backward(); fmt_optimizer.step()
    fmt_model.eval(); changed = fmt_input.clone(); changed[1, 19:, :-1] = 1234.
    with torch.no_grad():
        assert torch.equal(fmt_model(changed), fmt_model(fmt_input))
    del fmt_model, fmt_optimizer
    parameters = dict(p35=sum(p.numel() for p in task4_model(141, 'h0').parameters()),
                      conv=sum(p.numel() for p in Conv915().parameters()))
    assert parameters['p35'] == 76738 and .90 <= parameters['conv']/parameters['p35'] <= .95
    report = dict(parameters=parameters, resolutions={}, deterministic_pool_checks=pool_checks,
        fmt_definition=spec['encoding']['fmt_definition'], train_only_zscore_reference=True,
        device=device, identity=identity(config))
    for r in spec['encoding']['voxel_resolutions']:
        offsets, indices, values = sparse_voxels(geometry, counts, r)
        recovered = dense_sparse_batch([dict(offsets=offsets, indices=indices, values=values)], [(0, 0), (0, 1)], r, device)
        expected = bundle_voxels(geometry, counts, r).half().float()
        assert torch.equal(expected, recovered)
        independent = reference_voxel(geometry[0].cpu().numpy(), 27, r)
        assert np.allclose(expected[0].cpu().numpy(), independent, atol=.0005, rtol=.002)
        model = Conv915().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        xb = recovered[:1].expand(spec['training']['batch_size'] if device == 'cuda' else 2, -1, -1, -1, -1).contiguous()
        start = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(xb), torch.zeros(len(xb), device=device, dtype=torch.long))
            loss.backward(); optimizer.step()
        if device == 'cuda': torch.cuda.synchronize()
        report['resolutions'][str(r)] = dict(roundtrip_exact=True, loss=float(loss.detach()),
            two_steps_seconds=time.perf_counter()-start, max_voxel_error=float(np.max(np.abs(expected[0].cpu().numpy()-independent))))
        del model, optimizer, xb, recovered, expected
    report['complete'] = True
    write(Path(spec['output'])/f'preflight_{device}.json', report)
    print(json.dumps(report), flush=True)

def runtime(spec, config, phase, state, code=None):
    row = dict(version=spec['version'], phase=phase, state=state, exit_code=code, time_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity(config))
    if torch.cuda.is_available(): row['gpu'] = torch.cuda.get_device_name()
    root = Path(spec['output']); root.mkdir(parents=True, exist_ok=True)
    append_locked(root/'runtime.jsonl', json.dumps(row)+'\n')

if __name__=='__main__':main()
