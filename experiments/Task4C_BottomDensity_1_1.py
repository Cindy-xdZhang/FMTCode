"""Only append bottom-region training geometry; reuse frozen p35 and small Conv3D."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from types import FunctionType
import numpy as np
from scipy.spatial import cKDTree
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine
from experiments import Task4C_PhysicalLength_4_14 as original
from FMT_Utils import Task4C_BottomDensity_1_1 as data

CONFIG='config/Ablation_Task4C_BottomDensity_1.1.json'
SPLITS=('train','validation','test')
write=engine.write
sha=engine.sha


def load_spec(config):
    spec=json.loads(Path(config).read_text());ref=original.load_spec(spec['reference_config'])
    for key in ('flows','input_root','labels','bundles','integration','sampling','local_split'):
        assert spec[key]==ref[key],f'Frozen 4.14 setting changed: {key}'
    for key,value in ref['training'].items():assert spec['training'][key]==value
    assert spec['density']['split']=='train' and spec['density']['additional_bundles_per_existing_training_instance']==100
    assert spec['methods']==['p35','conv8','conv24','conv32']
    assert spec['encoding']['voxel_resolutions']==[8,24,32]
    assert spec['expected_counts']==dict(train=38600,validation=3000,test=10000,total=51600)
    assert spec['encoding']['fmt_definition']==next(c for c in engine.candidates() if c['id']=='n0_k06')
    return spec


def identity(config):
    result=engine.identity(config)
    for name in (__file__,'FMT_Utils/Task4C_BottomDensity_1_1.py','ibex_bash/task4c_bottom_density_1p1.sh',
                 'config/mainExp_Task4C_PhysicalLength_4.14.json','config/mainExp_Task4C_Multiscale_4.1.json'):
        result['sources'][name]=sha(name)
    return result


def run_engine(name,*args):
    """Bind run identity without editing or monkeypatching frozen model/training code."""
    fn=getattr(engine,name)
    return FunctionType(fn.__code__,dict(engine.__dict__,identity=identity),argdefs=fn.__defaults__)(*args)


def prepare(spec,config,index,pilot=False,input_root=None,source_root=None):
    data.prepare(spec,config,index,identity,write,sha,pilot,input_root,source_root)


def audit(spec,config):
    root=Path(spec['output']);counts={s:0 for s in SPLITS};reports={}
    for flow in spec['flows']:
        name=flow['name'];folder=root/'physical'/name;source=Path(spec['source_output'])/'physical'/name
        report=json.loads((folder/'preparation.json').read_text());assert report['identity']['config_sha256']==sha(config)
        catalog={r['head_component']:r for r in report['original_catalog']['labels']}
        for split in SPLITS:
            entry=report['splits'][split]
            for file,digest in entry['files'].items():assert sha(folder/split/file)==digest
            before=data.read_metadata(source/split);after=data.read_metadata(folder/split);n=len(before['labels'])
            counts[split]+=len(after['labels'])
            for key in before:assert np.array_equal(before[key],after[key][:n],equal_nan=True),key
            for file in ('geometry.npy','seeds.npy'):
                old=np.load(source/split/file,mmap_mode='r');new=np.load(folder/split/file,mmap_mode='r')
                assert np.array_equal(old,new[:n])
            if split!='train':
                for file,digest in report['source_hashes'][split].items():assert sha(folder/split/file)==digest
        training=data.read_metadata(folder/'train');n=spec['sampling']['per_flow_counts']['train']
        extra={k:v[n:] for k,v in training.items()};mask=np.arange(27)[None]<extra['counts'][:,None]
        assert np.all(extra['labels']==1) and np.all((extra['counts']>=10)&(extra['counts']<=27))
        ids,num=np.unique(extra['instance'],return_counts=True)
        assert np.all(num==spec['density']['additional_bundles_per_existing_training_instance'])
        assert len(ids)==spec['density']['expected_original_training_instances'][name]
        assert np.array_equal(ids,np.unique(training['instance'][:n][training['labels'][:n]==1]))
        for h,i in zip(extra['head_component'],extra['instance']):assert catalog[int(h)]['label']==1 and catalog[int(h)]['instance']==i
        with np.load(source/'cell_split.npz') as cells:assert np.isin(extra['source_cell'],cells['train']).all()
        plan=original.scale_plan(spec,name)
        assert np.array_equal(extra['ds'],[plan[s]['ds'] for s in extra['scale_id']])
        assert np.array_equal(extra['maxiteration'],[plan[s]['maxiteration'] for s in extra['scale_id']])
        ratios=(extra['half_arc_lengths']/extra['requested_half_length'][:,None,None])[mask]
        assert ratios.min()>=spec['integration']['minimum_actual_half_arc_fraction']
        assert ratios.max()<=spec['integration']['maximum_actual_half_arc_fraction']
        assert not np.any((extra['half_step_counts']>extra['maxiteration'][:,None,None])[mask])
        g=np.load(folder/'train'/'geometry.npy',mmap_mode='r')[n:]
        assert np.isfinite(g).all() and np.all(g[~mask]==0)
        assert np.allclose(g.sum((1,2))/(extra['counts'][:,None]*32),0,atol=2e-6)
        assert np.allclose(np.linalg.norm(g,axis=-1).max((1,2)),1,atol=2e-6)
        gt=data.read_dataset(Path(spec['input_root'])/flow['gt']);membership,_=data.sample_gt(gt,extra['center'])
        assert np.array_equal(membership,extra['instance'])
        centers=cKDTree(training['center'])
        minimum={}
        for split in ('validation','test'):
            m=data.read_metadata(folder/split);distances=centers.query(m['center'])[0]/m['local_grid_scale']
            assert distances.min()>=1-1e-12;minimum[split]=float(distances.min())
        reports[name]=dict(instances=len(ids),extra_train=len(extra['labels']),negative_rows_unchanged=True,
            validation_test_files_byte_identical=True,original_geometry_prefix_identical=True,minimum_eval_distance_over_h=minimum)
    assert counts=={s:spec['expected_counts'][s] for s in SPLITS}
    write(root/'data_audit.json',dict(complete=True,identity=identity(config),counts=counts,flows=reports))
    print(json.dumps(reports),flush=True)


def merge(spec,config):
    from sklearn.metrics import f1_score
    root=Path(spec['output']);rows=[]
    for method in spec['methods']:
        for seed in spec['training']['seeds']:
            folder=root/'runs'/method/f'seed{seed}';r=json.loads((folder/'result.json').read_text())
            assert r['complete'] and r['identity']['config_sha256']==sha(config)
            assert r['execution_revision']==spec['execution_revision']
            for split in SPLITS:
                p=folder/f'{split}_predictions.npz';assert sha(p)==r['predictions'][split]
                with np.load(p) as z:
                    assert abs(f1_score(z['labels'],z['probability']>=.5)-r['metrics'][split]['combined']['f1'])<1e-12
                    for fi,flow in enumerate(spec['flows']):
                        m=data.read_metadata(root/'physical'/flow['name']/split);take=z['flow_index']==fi;ids=z['row_in_split'][take]
                        assert np.array_equal(np.sort(ids),np.arange(len(m['labels'])))
                        assert np.array_equal(z['labels'][take],m['labels'][ids])
            rows.append(r)
    summary=[]
    for method in spec['methods']:
        group=[r for r in rows if r['method']==method];values=[r['metrics']['test']['combined']['f1'] for r in group]
        summary.append(dict(method=method,parameters=group[0]['parameters'],test_f1_mean=float(np.mean(values)),
            test_f1_std=float(np.std(values,ddof=1)),mean_training_minutes=float(np.mean([r['training_seconds'] for r in group])/60)))
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth'))
    run_engine('export_viewer_package',spec,config,rows)
    manifest=json.loads((root/'viewer_package/manifest.json').read_text())
    manifest['evidence_note']='Original 4.14 evaluation samples; only bottom-region training bundles appended; seed 96611.'
    write(root/'viewer_package/manifest.json',manifest)
    write(root/'summary.json',dict(complete=True,identity=identity(config),methods=summary))
    print(json.dumps(summary),flush=True)


def submit(spec,config):
    root=Path(spec['output']).resolve();assert shutil.disk_usage(Path.cwd()).free>20*2**30
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();write(root/'config.frozen.json',spec);previous=None
    phases=[('preflight',None,True,'00:20:00'),('pilot','0-1%2',False,'01:00:00'),('prepare','0-1%2',False,'03:00:00'),
        ('audit',None,False,'00:30:00'),('encode','0-1%2',True,'01:00:00'),('train','0-11%6',True,'12:00:00'),('merge',None,False,'00:30:00')]
    for phase,array,gpu,limit in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time='+limit,
            '--job-name=t4c-density-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if previous:command+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_bottom_density_1p1.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),command=command,
            dependency=previous,expected_device='V100' if gpu else 'CPU',identity=identity(config))
        engine.append_locked(root/'submissions.jsonl',json.dumps(row)+'\n');print(json.dumps(row),flush=True);previous=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','pilot','prepare','audit','encode','train','merge','submit','runtime'))
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',default='cuda');parser.add_argument('--input-root');parser.add_argument('--source-root')
    parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=load_spec(args.config)
    if args.phase in ('pilot','prepare'):prepare(spec,args.config,args.index,args.phase=='pilot',args.input_root,args.source_root)
    elif args.phase=='preflight':run_engine('preflight',spec,args.config,args.device)
    elif args.phase in ('encode','train'):run_engine(args.phase,spec,args.config,args.index)
    elif args.phase=='runtime':run_engine('runtime',spec,args.config,args.runtime_phase,args.state,args.exit_code)
    else:globals()[args.phase](spec,args.config)


if __name__=='__main__':main()
