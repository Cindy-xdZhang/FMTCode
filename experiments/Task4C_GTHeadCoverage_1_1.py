"""Append paired GT-head bundles and evaluate the already-selected c156 model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from types import FunctionType
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_GTHeadCoverage_1_1 as data

CONFIG='config/mainExp_Task4C_GTHeadCoverage_1.1.json'
SPLITS=('train','validation','test')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')


def identity(config):
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=data.sha(config),sources={p:data.sha(p) for p in (
            'experiments/Task4C_GTHeadCoverage_1_1.py','FMT_Utils/Task4C_GTHeadCoverage_1_1.py','FMT_Utils/Task4C_FPSAugmentSearch_1_1.py',
            'experiments/Task4C_FPSAugmentSearch_1_1.py','experiments/Task4C_PhysicalLength_4_14.py')})


def load_spec(config):
    spec=json.loads(Path(config).read_text())
    assert data.sha(spec['source_config'])==spec['source_config_sha256']
    assert spec['candidate']==dict(id='c156',pool='p35',profile='h0',architecture='wide_residual',
        points=48,sampling='uniform',augmentation='none',parameters=992386)
    assert spec['final']['seeds']==[96721] and spec['training']['threshold']==.5
    return spec


def metadata(folder):
    with np.load(Path(folder)/'metadata.npz') as z:return {k:z[k] for k in z.files}


def prepare(spec,config,index,pilot=False,input_root=None,source_root=None):
    out=Path(spec['output'])/('pilot' if pilot else '')
    report=data.prepare(spec,index,source_root or (Path(spec['source_output'])/'physical'),input_root,out,pilot)
    report['identity']=identity(config)
    write(out/'physical'/spec['flows'][index]['name']/'preparation.json',report)


def audit(spec,config,pilot=False,input_root=None,source_root=None):
    from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
    from scipy.interpolate import RegularGridInterpolator
    root=Path(spec['output'])/('pilot' if pilot else '')
    source=Path(source_root or (Path(spec['source_output'])/'physical'))
    counts={s:0 for s in SPLITS};reports={}
    for fi,flow in enumerate(spec['flows']):
        name=flow['name'];folder=root/'physical'/name
        old=data.read_source(source,name)
        added={r:metadata(folder/('added_'+r)) for r in ('train','test')}
        scene=data.load_scene(spec,fi,input_root);gt=scene['gt']
        axes=tuple(scene['axes'][::-1])
        for role,m in added.items():
            ids,_=sample_gt(gt,m['center']);assert np.array_equal(ids,m['instance'])
            query=m['center'][:,::-1]
            velocity=RegularGridInterpolator(axes,scene['velocity'])(query)
            omega=RegularGridInterpolator(axes,scene['omega'])(query)
            cosine=np.abs(np.sum(velocity*omega,axis=1))/(np.linalg.norm(velocity,axis=1)*np.linalg.norm(omega,axis=1))
            assert np.isfinite(cosine).all() and np.all(cosine*cosine<.5)
            assert np.all(RegularGridInterpolator(axes,scene['lambda2'])(query)<flow['lambda2_threshold'])
            assert np.all(RegularGridInterpolator(axes,scene['oyf'])(query)>0)
            assert np.all(m['labels']==1) and np.all((m['counts']>=10)&(m['counts']<=27))
            unique,n=np.unique(m['instance'],return_counts=True)
            q=(1 if pilot else spec['head_coverage']['test_bundles_per_instance'])*(spec['head_coverage']['train_bundles_per_test'] if role=='train' else 1)
            assert len(unique)==spec['head_coverage']['instances'][name] and np.all(n==q)
            g=np.load(folder/('added_'+role)/'geometry.npy',mmap_mode='r')
            mask=np.arange(27)[None]<m['counts'][:,None]
            assert np.isfinite(g).all() and np.all(g[~mask]==0)
            assert np.allclose(np.asarray(g,dtype=float).sum((1,2))/(m['counts'][:,None]*32),0,atol=2e-6)
            assert np.allclose(np.linalg.norm(g,axis=-1).max((1,2)),1,atol=2e-6)
            limit=np.broadcast_to(m['requested_half_length'][:,None,None],m['half_arc_lengths'].shape)[mask]
            arc=m['half_arc_lengths'][mask]
            assert np.all(arc>=.95*limit-1e-10) and np.all(arc<=1.002*limit+1e-10)
            assert len(np.unique(m['center'],axis=0))==len(m['center'])
        ep=np.concatenate([old[r]['center'] for r in ('validation','test')]+[added['test']['center']])
        eh=np.concatenate([old[r]['local_grid_scale'] for r in ('validation','test')]+[added['test']['local_grid_scale']])
        assert data.clearance(added['train']['center'],cKDTree(ep),ep,eh).all()
        train=np.concatenate((old['train']['center'],added['train']['center']))
        minimum=cKDTree(train).query(added['test']['center'])[0]
        assert np.all(minimum>=added['test']['local_grid_scale']-1e-12)
        assert data.clearance(added['test']['center'],cKDTree(old['validation']['center']),old['validation']['center'],old['validation']['local_grid_scale'],.5).all()
        for instance in np.unique(added['test']['instance']):
            t=added['test']['instance']==instance;a=added['train']['instance']==instance
            distance=cKDTree(added['train']['center'][a]).query(added['test']['center'][t])[0]
            assert np.all(distance<=4*added['test']['local_grid_scale'][t]+1e-12)
        if not pilot:
            original_manifest=json.loads((source/name/'preparation.json').read_text())
            new_manifest=json.loads((folder/'preparation.json').read_text())
            for role in SPLITS:
                merged=metadata(folder/role);n=len(old[role]['labels']);counts[role]+=len(merged['labels'])
                for filename,digest in original_manifest['splits'][role]['files'].items():assert data.sha(source/name/role/filename)==digest
                for filename,digest in new_manifest['splits'][role]['files'].items():assert data.sha(folder/role/filename)==digest
                for key in old[role]:assert np.array_equal(old[role][key],merged[key][:n],equal_nan=True),key
                for filename in ('geometry.npy','seeds.npy'):
                    a=np.load(source/name/role/filename,mmap_mode='r');b=np.load(folder/role/filename,mmap_mode='r')
                    for first in range(0,n,512):assert np.array_equal(a[first:first+512],b[first:min(first+512,n)])
                if role=='validation':
                    for filename,digest in original_manifest['splits'][role]['files'].items():assert data.sha(folder/role/filename)==digest
                else:
                    for key in added[role]:assert np.array_equal(added[role][key],merged[key][n:],equal_nan=True)
        reports[name]=dict(instances=len(np.unique(added['test']['instance'])),
            added={r:len(m['labels']) for r,m in added.items()},minimum_new_test_train_over_h=float((minimum/added['test']['local_grid_scale']).min()))
        del scene,gt
    if not pilot:assert counts==spec['expected_counts']
    report=dict(complete=True,identity=identity(config),pilot=pilot,flows=reports,counts=counts,
        original_prefix_unchanged=None if pilot else True,old_validation_unchanged=None if pilot else True,all_GT_instances_covered=True)
    write(root/'data_audit.json',report)
    if not pilot:write(root/'selection.lock.json',dict(complete=True,selected=spec['candidate'],threshold=.5,seed=96721,
        selected_by='user_fixed_previously_selected_c156_before_new_data',identity=identity(config)))
    print(json.dumps(report),flush=True)


def train(spec,config):
    from experiments import Task4C_FPSAugmentSearch_1_1 as frozen
    root=Path(spec['output']);assert json.loads((root/'data_audit.json').read_text())['complete']
    folder=root/'final/c156/seed96721'
    def predict(model,dataset,norm,batch):
        result=frozen.predict(model,dataset,norm,batch)
        if dataset.role=='test':
            training=frozen.Dataset(spec,'train',spec['candidate']);training.encode(spec['candidate'],batch)
            probability,_=frozen.predict(model,training,norm,batch)
            frozen.save_predictions(folder,'train',training,probability)
            del training
        return result
    fn=frozen.train
    FunctionType(fn.__code__,dict(frozen.__dict__,identity=identity,predict=predict),argdefs=fn.__defaults__)(spec,config,0,'final')
    result=json.loads((folder/'result.json').read_text())
    result['predictions']['train']=data.sha(folder/'train_predictions.npz')
    write(folder/'result.json',result)


def gpu_check(spec,config):
    import torch
    from experiments import Task4C_FPSAugmentSearch_1_1 as frozen
    frozen.engine.deterministic('cuda');torch.manual_seed(97011)
    dataset=frozen.Dataset(spec,'train',spec['candidate'],limit=32);dataset.encode(spec['candidate'])
    norm=frozen.fit_normalizer(dataset,spec['candidate'])
    model=frozen.method.FourierClassifier('p35','wide_residual','h0').cuda()
    assert sum(p.numel() for p in model.parameters())==992386
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    tokens=frozen.standardize(dataset.clean,norm);history=[]
    for _ in range(120):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(tokens,dataset.neighbors),dataset.targets)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        optimizer.step();history.append(float(loss.detach()))
    p,_=frozen.predict(model,dataset,norm,128);score=frozen.metrics(dataset.labels,p)
    assert score['f1']>=.95 and history[-1]<history[0]
    write(Path(spec['output'])/'gpu_check.json',dict(complete=True,identity=identity(config),samples=32,
        parameters=992386,gpu=torch.cuda.get_device_name(),pilot_f1=score['f1'],initial_loss=history[0],final_loss=history[-1],scientific_metric=False))


def submit(spec,config):
    root=Path(spec['output']);root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    def launch(phase,dependency=None,array=None,gpu=False):
        args=['sbatch','--parsable','--job-name=GThead_'+phase,'--time=08:00:00','--cpus-per-task=4','--mem=64G',
            '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
        if dependency:args+=['--dependency=afterok:'+str(dependency)]
        if array:args+=['--array='+array]
        if gpu:args+=['--gres=gpu:1','--constraint=v100']
        args+=['ibex_bash/task4c_gt_head_coverage_1p1.sh',phase,config]
        return subprocess.check_output(args,text=True).strip().split(';')[0]
    jobs={};jobs['prepare']=launch('prepare',array='0-1');jobs['audit']=launch('audit',jobs['prepare'])
    jobs['gpu_check']=launch('gpu-check',jobs['audit'],gpu=True)
    jobs['train']=launch('train',jobs['gpu_check'],gpu=True)
    jobs['export']=launch('export',jobs['train'])
    write(root/'submission.json',dict(identity=identity(config),submitted_at_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs))
    print(json.dumps(jobs),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['coordinates','pilot','pilot-audit','prepare','audit','gpu-check','train','export','submit','runtime'])
    p.add_argument('--config',default=CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--source-root');p.add_argument('--input-root');p.add_argument('--state');p.add_argument('--runtime-phase');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=load_spec(a.config)
    if a.phase=='coordinates':data.coordinate_pilot(spec,a.index,a.source_root,a.input_root,Path(spec['output'])/f'coordinates_{a.index}.json')
    elif a.phase in ('pilot','prepare'):prepare(spec,a.config,a.index,a.phase=='pilot',a.input_root,a.source_root)
    elif a.phase in ('pilot-audit','audit'):audit(spec,a.config,a.phase=='pilot-audit',a.input_root,a.source_root)
    elif a.phase=='gpu-check':gpu_check(spec,a.config)
    elif a.phase=='train':train(spec,a.config)
    elif a.phase=='export':
        from experiments.Export_Task4C_GTHeadCoverage_1_1 import export
        export(spec,a.config,identity)
    elif a.phase=='submit':submit(spec,a.config)
    elif a.phase=='runtime':
        from experiments.Task4C_HairpinBinary_2_1 import append_locked
        import socket
        append_locked(Path(spec['output'])/'runtime.jsonl',json.dumps(dict(identity=identity(a.config),phase=a.runtime_phase,
            state=a.state,exit_code=a.exit_code,hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
            array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


if __name__=='__main__':main()
