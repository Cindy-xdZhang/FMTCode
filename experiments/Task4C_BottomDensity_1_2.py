"""Fivefold training bundles; frozen validation/test, p35 and small Conv3D."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
import numpy as np
from scipy.spatial import cKDTree
from experiments import Task4C_BottomDensity_1_1 as previous
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine
from experiments import Task4C_PhysicalLength_4_14 as original
from FMT_Utils import Task4C_BottomDensity_1_2 as data

CONFIG='config/Ablation_Task4C_BottomDensity_1.2.json'
SPLITS=previous.SPLITS
write=engine.write
sha=engine.sha


def load_spec(config):
    spec=json.loads(Path(config).read_text())
    assert sha(spec['source_config'])==spec['source_config_sha256']
    old=previous.load_spec(spec['source_config'])
    for key in ('flows','input_root','labels','bundles','integration','sampling','local_split','training',
                'density','model_parameters','whole_instance_holdout_count','checkpoint_policy'):
        assert spec[key]==old[key],f'Frozen 1.1 setting changed: {key}'
    for key,value in old['encoding'].items():
        if key!='voxel_resolutions':assert spec['encoding'][key]==value
    assert spec['expansion']['factor']==5 and spec['expansion']['new_centers_per_source_bundle']==4
    assert spec['expansion']['source_rows_per_flow']==dict(channel=19700,tbl=18900)
    assert spec['methods']==['p35','conv16','conv24'] and spec['encoding']['voxel_resolutions']==[16,24]
    assert spec['expected_counts']==dict(train=193000,validation=3000,test=10000,total=206000)
    return spec


def identity(config):
    result=engine.identity(config)
    for name in ('experiments/Task4C_BottomDensity_1_2.py','FMT_Utils/Task4C_BottomDensity_1_2.py',
                 'ibex_bash/task4c_bottom_density_1p2.sh','experiments/Task4C_BottomDensity_1_1.py',
                 'FMT_Utils/Task4C_BottomDensity_1_1.py','config/Ablation_Task4C_BottomDensity_1.1.json',
                 'config/mainExp_Task4C_PhysicalLength_4.14.json','config/mainExp_Task4C_Multiscale_4.1.json'):
        result['sources'][name]=sha(name)
    return result


def run_engine(name,*args):
    fn=getattr(engine,name)
    return FunctionType(fn.__code__,dict(engine.__dict__,identity=identity),argdefs=fn.__defaults__)(*args)


def prepare(spec,config,index,pilot=False,input_root=None,source_root=None):
    data.prepare(spec,config,index,identity,write,sha,pilot,input_root,source_root)


def capacity(spec,config):
    import torch
    engine.deterministic('cuda');root=Path(spec['output']);byte_count=samples=0
    for flow in spec['flows']:
        folder=root/'pilot/physical'/flow['name']/'added_train'
        geometry=np.load(folder/'geometry.npy',mmap_mode='r');m=data.read_metadata(folder)
        for first in range(0,len(geometry),32):
            sl=slice(first,first+32)
            g=torch.as_tensor(np.array(geometry[sl]),device='cuda')
            counts=torch.as_tensor(m['counts'][sl],device='cuda')
            for resolution in spec['encoding']['voxel_resolutions']:
                byte_count+=sum(a.nbytes for a in engine.sparse_voxels(g,counts,resolution))
            samples+=len(g)
    estimated=byte_count/samples*spec['expected_counts']['total']
    required=estimated*2+spec['expected_counts']['total']*(27*32*3*4*3+27*142*4+12000)+10*2**30
    available=shutil.disk_usage(root).free
    result=dict(complete=available>=required,identity=identity(config),pilot_samples=samples,
        estimated_sparse_bytes=estimated,required_bytes=required,available_bytes=available)
    write(root/'capacity.json',result);print(json.dumps(result),flush=True)
    assert result['complete'],'Insufficient storage; no existing data removed'


def audit(spec,config):
    root=Path(spec['output']);totals={s:0 for s in SPLITS};reports={}
    for flow in spec['flows']:
        name=flow['name'];folder=root/'physical'/name;source=Path(spec['source_output'])/'physical'/name
        report=json.loads((folder/'preparation.json').read_text())
        assert report['identity']['config_sha256']==sha(config) and not report['pilot']
        assert report['source_commit']==spec['source_scientific_commit']
        for split in SPLITS:
            entry=report['splits'][split]
            for filename,digest in entry['files'].items():assert sha(folder/split/filename)==digest
            before=data.read_metadata(source/split);after=data.read_metadata(folder/split);n=len(before['labels'])
            totals[split]+=len(after['labels'])
            for key in before:assert np.array_equal(before[key],after[key][:n],equal_nan=True),key
            for filename in ('geometry.npy','seeds.npy'):
                old=np.load(source/split/filename,mmap_mode='r');new=np.load(folder/split/filename,mmap_mode='r')
                for first in range(0,n,512):assert np.array_equal(old[first:first+512],new[first:min(first+512,n)])
            if split!='train':
                for filename,digest in report['source_hashes'][split].items():assert sha(folder/split/filename)==digest
        old=data.read_metadata(source/'train');training=data.read_metadata(folder/'train');n=len(old['labels'])
        extra={k:v[n:] for k,v in training.items()};assert len(extra['labels'])==n*4
        assert sha(folder/'augmentation_index.npz')==report['augmentation_index_sha256']
        with np.load(folder/'augmentation_index.npz') as z:source_row=z['source_row'];replica=z['replica'];lower=z['lower_pool']
        assert np.array_equal(np.bincount(source_row,minlength=n),np.full(n,4))
        assert np.array_equal(np.unique(source_row*4+replica),np.arange(n*4))
        assert np.array_equal(lower,data.lower_pool_flags(old,spec)[source_row])
        for key in ('labels','head_component','instance','scale_id','ds','maxiteration','requested_half_length'):
            assert np.array_equal(extra[key],old[key][source_row]),key
        assert np.array_equal(np.bincount(training['labels'],minlength=2),5*np.bincount(old['labels'],minlength=2))
        assert len(np.unique(extra['center'],axis=0))==len(extra['center'])
        assert cKDTree(old['center']).query(extra['center'])[0].min()>1e-12
        with np.load(source/'cell_split.npz') as z:assert np.isin(extra['source_cell'],z['train']).all()
        catalog={r['head_component']:r for r in report['generation']['catalog']['labels']}
        for h,label,instance in zip(extra['head_component'],extra['labels'],extra['instance']):
            assert catalog[int(h)]['label']==label and catalog[int(h)]['instance']==instance
        plan=original.scale_plan(spec,name)
        assert np.array_equal(extra['ds'],[plan[s]['ds'] for s in extra['scale_id']])
        assert np.array_equal(extra['maxiteration'],[plan[s]['maxiteration'] for s in extra['scale_id']])
        counts=extra['counts'];valid=np.arange(27)[None]<counts[:,None]
        assert np.all((counts>=10)&(counts<=27))
        ratio=(extra['half_arc_lengths']/extra['requested_half_length'][:,None,None])[valid]
        assert np.isfinite(ratio).all()
        assert ratio.min()>=spec['integration']['minimum_actual_half_arc_fraction']
        assert ratio.max()<=spec['integration']['maximum_actual_half_arc_fraction']
        assert not np.any((extra['half_step_counts']>extra['maxiteration'][:,None,None])[valid])
        geometry=np.load(folder/'train/geometry.npy',mmap_mode='r')[n:]
        for first in range(0,len(geometry),512):
            sl=slice(first,first+512);g=np.asarray(geometry[sl],np.float64)
            assert np.isfinite(g).all() and np.all(g[~valid[sl]]==0)
            assert np.allclose(g.sum((1,2))/(counts[sl,None]*32),0,atol=2e-6)
            assert np.allclose(np.linalg.norm(g,axis=-1).max((1,2)),1,atol=2e-6)
        minimum={};tree=cKDTree(training['center'])
        for split in ('validation','test'):
            m=data.read_metadata(folder/split);distance=tree.query(m['center'])[0]/m['local_grid_scale']
            assert distance.min()>=1-1e-12;minimum[split]=float(distance.min())
        reports[name]=dict(original_train=n,new_train=len(extra['labels']),total_train=len(training['labels']),
            class_counts=np.bincount(training['labels'],minlength=2).tolist(),sampling_strata_exactly_fivefold=True,
            all_added_centers_unique_and_new=True,source_prefix_unchanged=True,validation_test_byte_identical=True,
            minimum_evaluation_distance_over_h=minimum,source_strata=report['generation']['source_strata'])
    assert totals=={s:spec['expected_counts'][s] for s in SPLITS}
    write(root/'data_audit.json',dict(complete=True,identity=identity(config),counts=totals,flows=reports))
    print(json.dumps(reports),flush=True)


def merge(spec,config):
    fn=previous.merge
    FunctionType(fn.__code__,dict(previous.__dict__,identity=identity,run_engine=run_engine),argdefs=fn.__defaults__)(spec,config)
    file=Path(spec['output'])/'viewer_package/manifest.json';manifest=json.loads(file.read_text())
    manifest['evidence_note']='Fivefold 1.1 training with fresh seeds and unchanged sampling strata; original evaluation files; seed 96611.'
    write(file,manifest)


def submit(spec,config):
    root=Path(spec['output']).resolve();assert shutil.disk_usage(Path.cwd()).free>30*2**30
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();write(root/'config.frozen.json',spec);dependency=None
    phases=[('preflight',None,True,'00:20:00'),('pilot','0-1%2',False,'02:00:00'),
        ('capacity',None,True,'00:30:00'),('prepare','0-1%2',False,'12:00:00'),('audit',None,False,'01:00:00'),
        ('encode','0-1%2',True,'03:00:00'),('train','0-8%6',True,'1-00:00:00'),('merge',None,False,'01:00:00')]
    for phase,array,gpu,limit in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time='+limit,
            '--job-name=t4c-fivefold-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_bottom_density_1p2.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),command=command,
            dependency=dependency,expected_device='V100' if gpu else 'CPU',identity=identity(config))
        engine.append_locked(root/'submissions.jsonl',json.dumps(row)+'\n');print(json.dumps(row),flush=True);dependency=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','pilot','capacity','prepare','audit','encode','train','merge','submit','runtime'))
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
