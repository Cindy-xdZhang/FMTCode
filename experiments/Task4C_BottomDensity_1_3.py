"""8/12-cubed Conv3D on the frozen fivefold dataset, with unchanged training."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
import numpy as np
import torch
from experiments import Task4C_BottomDensity_1_2 as source
from experiments import Task4C_BottomDensity_1_1 as merge_source
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine

CONFIG='config/Ablation_Task4C_BottomDensity_1.3.json'
SPLITS=engine.SPLITS
write=engine.write
sha=engine.sha


def load_spec(config):
    definition=json.loads(Path(config).read_text())
    assert sha(definition['base_config'])==definition['base_config_sha256']
    spec=source.load_spec(definition['base_config'])
    allowed={'version','user_version','execution_revision','output','base_config','base_config_sha256',
        'source_output','source_scientific_commit','dataset_policy','methods','voxel_resolutions'}
    assert set(definition)==allowed
    spec.update(definition);spec['encoding']['voxel_resolutions']=definition['voxel_resolutions']
    assert spec['methods']==['conv8','conv12'] and spec['encoding']['voxel_resolutions']==[8,12]
    assert spec['dataset_policy']=='reuse_all_source_geometry_seeds_metadata_without_sampling'
    assert spec['model_parameters']['conv3d']==72192
    return spec


def identity(config):
    result=source.identity(config)
    for name in ('experiments/Task4C_BottomDensity_1_3.py','ibex_bash/task4c_bottom_density_1p3.sh',
                 'config/Ablation_Task4C_BottomDensity_1.2.json'):
        result['sources'][name]=sha(name)
    return result


def run_engine(name,*args):
    fn=getattr(engine,name)
    return FunctionType(fn.__code__,dict(engine.__dict__,identity=identity),argdefs=fn.__defaults__)(*args)


def reuse(spec,config):
    root=Path(spec['output']);src=Path(spec['source_output'])
    audit=json.loads((src/'data_audit.json').read_text())
    assert audit['complete'] and audit['identity']['git_commit']==spec['source_scientific_commit']
    assert audit['identity']['config_sha256']==spec['base_config_sha256']
    counts={s:0 for s in SPLITS};hashes={}
    for flow in spec['flows']:
        name=flow['name'];folder=src/'physical'/name
        report=json.loads((folder/'preparation.json').read_text())
        assert report['identity']['git_commit']==spec['source_scientific_commit']
        assert report['identity']['config_sha256']==spec['base_config_sha256']
        for split in SPLITS:
            entry=report['splits'][split]
            for filename,digest in entry['files'].items():
                f=folder/split/filename;assert sha(f)==digest
                hashes[str(f.relative_to(src))]=digest
            with np.load(folder/split/'metadata.npz') as z:assert len(z['labels'])==entry['samples']
            counts[split]+=entry['samples']
    assert counts=={s:spec['expected_counts'][s] for s in SPLITS}
    (root/'physical').symlink_to((src/'physical').resolve(),target_is_directory=True)
    write(root/'data_audit.json',dict(complete=True,identity=identity(config),counts=counts,
        source_output=str(src),source_commit=spec['source_scientific_commit'],all_source_files_hash_equal=True,
        source_data_audit_sha256=sha(src/'data_audit.json'),physical_files=hashes))
    print(json.dumps(dict(counts=counts,verified_files=len(hashes),physical_sampling=False)),flush=True)


def encode(spec,config,index):
    """Use the exact frozen voxelizer and cache layout, without unused FMT tokens."""
    engine.deterministic('cuda');root=Path(spec['output']);name=spec['flows'][index]['name']
    report=engine.physical_reports(spec)[name]
    audit=json.loads((root/'data_audit.json').read_text());assert audit['complete']
    records={}
    for split in SPLITS:
        assert shutil.disk_usage(root).free>10*2**30
        folder=root/'physical'/name/split
        for filename,digest in report['splits'][split]['files'].items():assert sha(folder/filename)==digest
        geometry=np.load(folder/'geometry.npy',mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:counts=z['counts'].astype(np.int64)
        dest=root/'encoded'/name/split;dest.mkdir(parents=True,exist_ok=False)
        sparse={r:dict(offsets=[np.array([0],np.int64)],indices=[],values=[]) for r in spec['encoding']['voxel_resolutions']}
        for first in range(0,len(geometry),spec['encoding']['batch_size']):
            sl=slice(first,first+spec['encoding']['batch_size'])
            g=torch.as_tensor(np.array(geometry[sl]),device='cuda');c=torch.as_tensor(counts[sl],device='cuda')
            for resolution,p in sparse.items():
                offsets,indices,values=engine.sparse_voxels(g,c,resolution)
                p['offsets'].append(offsets[1:]+p['offsets'][-1][-1]);p['indices'].append(indices);p['values'].append(values)
        for resolution,p in sparse.items():
            target=dest/f'r{resolution}';target.mkdir()
            for key,arrays in p.items():np.save(target/f'{key}.npy',np.concatenate(arrays))
        records[split]=dict(samples=len(geometry),files={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*.npy')})
        write(dest/'manifest.json',records[split]);print('encoded',name,split,len(geometry),flush=True)
    write(root/'encoding'/f'{index}.json',dict(complete=True,identity=identity(config),splits=records))


def merge(spec,config):
    fn=merge_source.merge
    FunctionType(fn.__code__,dict(merge_source.__dict__,identity=identity,run_engine=run_engine),argdefs=fn.__defaults__)(spec,config)
    file=Path(spec['output'])/'viewer_package/manifest.json';manifest=json.loads(file.read_text())
    manifest['evidence_note']='Conv8/Conv12 on exactly the frozen fivefold 1.2 dataset; no new sampling; seed 96611.'
    write(file,manifest)


def submit(spec,config):
    root=Path(spec['output']).resolve();assert shutil.disk_usage(Path.cwd()).free>15*2**30
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();write(root/'config.frozen.json',spec);dependency=None
    phases=[('preflight',None,True,'00:20:00'),('reuse',None,False,'00:30:00'),
        ('encode','0-1%2',True,'01:00:00'),('train','0-5%6',True,'08:00:00'),('merge',None,False,'01:00:00')]
    for phase,array,gpu,limit in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time='+limit,
            '--job-name=t4c-conv812-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_bottom_density_1p3.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),command=command,
            dependency=dependency,expected_device='V100' if gpu else 'CPU',identity=identity(config))
        engine.append_locked(root/'submissions.jsonl',json.dumps(row)+'\n');print(json.dumps(row),flush=True);dependency=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','reuse','encode','train','merge','submit','runtime'))
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',default='cuda');parser.add_argument('--runtime-phase')
    parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=load_spec(args.config)
    if args.phase=='preflight':run_engine('preflight',spec,args.config,args.device)
    elif args.phase=='train':run_engine('train',spec,args.config,args.index)
    elif args.phase=='encode':encode(spec,args.config,args.index)
    elif args.phase=='runtime':run_engine('runtime',spec,args.config,args.runtime_phase,args.state,args.exit_code)
    else:globals()[args.phase](spec,args.config)


if __name__=='__main__':main()
