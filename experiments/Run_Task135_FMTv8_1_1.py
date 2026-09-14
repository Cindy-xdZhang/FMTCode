"""Five geometry encoders with frozen Task1/3/5 training and evaluation rules."""
from __future__ import annotations
import argparse
import collections
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch

from FMT_Utils.FMT_V8_Comparison_3D import WIDTHS,shared_features,select_features
from FMT_Utils.fmt_v8 import pool_neighbor_features
from experiments import Run_Task1235_AngleFeatures_1_1 as frozen
from experiments.Record_Task1235_AngleFeatures_1_1 import append_record

DEFAULT_CONFIG='config/mainExp_Task135_FMTv8_1.1.json'
TIMINGS=[]
ORIGINAL_RAW=frozen.train_raw
ORIGINAL_RESIDUAL=frozen.train_residual
ORIGINAL_KMEANS=frozen.fit_kmeans_transform


def features(rows,name,task,device):
    started=time.perf_counter();parts=[];computed=0
    for r in rows:
        if name=='raw':value=r['raw'].reshape(len(r['raw']),-1)
        else:
            if '_v8_shared' not in r:
                r['_v8_shared']=shared_features(r['raw'],r['cached_fmt']);computed+=len(r['raw'])
            value=select_features(r['_v8_shared'],name)
            assert value.shape==(len(r['raw']),WIDTHS[name])
        parts.append(value)
    result=np.concatenate(parts).astype(np.float32)
    TIMINGS.append(dict(kind='feature_preparation',feature=name,task=task,rows=len(result),
        shared_rows_computed=computed,seconds=time.perf_counter()-started,
        scope='cached_original161; direction_and_angle_shared_once; includes_array_assembly'))
    return result


def timed_raw(*args,**kwargs):
    started=time.perf_counter();result=ORIGINAL_RAW(*args,**kwargs)
    TIMINGS.append(dict(kind='raw_training',arm=args[2],seconds=time.perf_counter()-started))
    return result


def timed_residual(*args,**kwargs):
    started=time.perf_counter();result=ORIGINAL_RESIDUAL(*args,**kwargs)
    directory=args[-1] if len(args)>=7 else kwargs.get('output_dir','unknown')
    TIMINGS.append(dict(kind='residual_training',arm=Path(directory).name,seconds=time.perf_counter()-started))
    return result


def timed_kmeans(*args,**kwargs):
    started=time.perf_counter();result=ORIGINAL_KMEANS(*args,**kwargs)
    TIMINGS.append(dict(kind='clustering_fit',dimensions=args[0].shape[-1],seconds=time.perf_counter()-started))
    return result


def source_identity(config):
    manifest=json.loads(Path('SOURCE_MANIFEST.json').read_text())
    for name,expected in manifest.items():
        assert frozen.sha(name)==expected,name
    return dict(git_commit=Path('SOURCE_COMMIT.txt').read_text().strip(),
        config_sha256=frozen.sha(config),source_manifest_sha256=frozen.sha('SOURCE_MANIFEST.sha256'))


def provenance(spec,config,task,dataset,seed,device):
    return dict(experiment=spec['experiment'],task=task,dataset=dataset,seed=seed,
        **source_identity(config),runner_sha256=frozen.sha(__file__),
        encoder_hashes={name:frozen.sha('FMT_Utils/'+name) for name in ('fmt_v8.py','FMT_V8_Comparison_3D.py','DFT_FMT_3D.py','FMT_Angles_3D.py')},
        job=os.environ.get('SLURM_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),node=socket.gethostname(),
        start_time=time.time(),device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        torch=torch.__version__,numpy=np.__version__)


def configure():
    frozen.features=features;frozen.provenance=provenance
    frozen.train_raw=timed_raw;frozen.train_residual=timed_residual;frozen.fit_kmeans_transform=timed_kmeans


def preflight(spec,config):
    configure();source_identity(config)
    # The old preflight checks source hashes, labels, late cylinder times and all
    # requested representations without opening evaluation feature arrays.
    frozen.preflight(spec,config)
    checks=[]
    for dataset in spec['datasets']:
        rows=frozen.records(spec,'Task3',dataset,'train')
        r=rows[0];n=min(len(r['raw']),48)
        shared=shared_features(r['raw'][:n],r['cached_fmt'][:n]);nb=shared[:,23:161]
        expected=np.take_along_axis(nb,np.argsort(-np.abs(nb),axis=-1,kind='stable')[:,:4],1)
        value=select_features(shared,'fmt_v8_100')
        assert np.array_equal(value[:,:23],shared[:,:23]) and np.array_equal(value[:,23:95],shared[:,161:233])
        assert np.array_equal(value[:,95:99],expected)
        np.testing.assert_allclose(value[:,99],nb.mean(1),atol=1e-6,rtol=1e-6)
        assert np.array_equal(shared[:,:161],r['cached_fmt'][:n])
        angle=select_features(shared,'fmt_v5_326')
        assert np.array_equal(angle[:,:161],shared[:,:161]) and np.array_equal(angle[:,161:],shared[:,233:])
        checks.append(dict(dataset=dataset,rows=n,common_blocks_exact=True,pooling_numpy_audit=True))
    frozen.write_json(Path(spec['output_root'])/'geometry_audit.json',dict(status='PASS',checks=checks,
        widths=WIDTHS,old_kinematic_blocks_absent=True,identity=source_identity(config)))


def smoke(spec,config):
    """Exercise the actual frozen training chain on small real training fixtures."""
    from unittest.mock import patch
    configure();s=copy.deepcopy(spec);s['output_root']=str(Path(spec['output_root'])/'engineering_smoke')
    s['datasets']=[spec['datasets'][0]];s['training']['max_epochs']=1;s['training']['patience']=1
    for task in s['tasks']:s[task.lower()]['seeds']=[0]
    rows=frozen.records(spec,'Task3',s['datasets'][0],'train')
    raw=frozen.raw_array(rows);labels=frozen.labels(rows);cached=np.concatenate([r['cached_fmt'] for r in rows])
    pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0)
    assert len(pos)>=150 and len(neg)>=700
    data={}
    for role,pi,ni in [('train',slice(0,100),slice(0,500)),('validation',slice(100,125),slice(500,600)),('confirmation',slice(125,150),slice(600,700))]:
        ids=np.r_[pos[pi],neg[ni]]
        data[role]=[dict(raw=raw[ids],cached_fmt=cached[ids],labels=labels[ids],scale_id=np.arange(len(ids))%9)]
    root=Path(s['output_root']);root.mkdir(parents=True,exist_ok=False)
    smoke_config=root/'config.json';frozen.write_json(smoke_config,s)
    frozen.write_json(root/'preflight.json',dict(status='PASS',config_sha256=frozen.sha(smoke_config),smoke_only=True))
    def load(s,task,dataset,role):
        if role=='confirmation':assert (root/'shards'/task/dataset/'seed0/frozen_models.json').exists()
        return copy.deepcopy(data[role])
    with patch.object(frozen,'records',side_effect=load),patch.object(frozen,'evidence',return_value=[]):
        for task in s['tasks']:
            TIMINGS.clear();frozen.run(s,str(smoke_config),task,s['datasets'][0],0)
            frozen.audit_shard(root/'shards'/task/s['datasets'][0]/'seed0',s[task.lower()]['arms'])
    deleted=frozen.cleanup_checkpoints(root/'shards')
    frozen.write_json(Path(spec['output_root'])/'smoke.json',dict(status='PASS',training_only_fixtures=True,
        no_scientific_metrics=True,deleted_checkpoints=deleted,source_identity=source_identity(config)))


def run_chain(spec,config,phase,index):
    configure();tasks=['Task3','Task5'] if phase=='Task35' else ['Task1']
    seeds=spec[tasks[0].lower()]['seeds'];di,si=divmod(index,len(seeds));dataset=spec['datasets'][di];seed=seeds[si]
    if phase=='Task35':assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    for task in tasks:
        TIMINGS.clear();frozen.run(spec,config,task,dataset,seed)
        target=Path(spec['output_root'])/'shards'/task/dataset/f'seed{seed}'
        frozen.write_json(target/'timing.json',dict(events=TIMINGS,units='seconds',
            residual_total_cost='shared_raw_training_plus_own_residual_training; shared_raw_count_once_per_method',
            input_dimensions=WIDTHS))
        frozen.audit_shard(target,spec[task.lower()]['arms'])
    deleted=[]
    for task in tasks:
        deleted.extend(frozen.cleanup_checkpoints(Path(spec['output_root'])/'shards'/task/dataset/f'seed{seed}'))
    frozen.write_json(Path(spec['output_root'])/'chain_audits'/phase/dataset/f'seed{seed}.json',
        dict(status='PASS',deleted_checkpoints=deleted,end_time=time.time()))


def audit(spec,config):
    rows=[];root=Path(spec['output_root'])
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            for seed in spec[task.lower()]['seeds']:
                rows.extend(frozen.audit_shard(root/'shards'/task/dataset/f'seed{seed}',spec[task.lower()]['arms']))
    summary=[];paired=[]
    for task in spec['tasks']:
        for arm in spec[task.lower()]['arms']:
            part=[r for r in rows if r['task']==task and r['arm']==arm]
            item=dict(task=task,arm=arm,runs=len(part))
            for metric in ('f1','average_precision','iou','ari','nmi'):
                if all(r.get(metric) not in (None,'') for r in part):
                    values=[float(r[metric]) for r in part]
                    seedmeans=[np.mean([float(r[metric]) for r in part if int(r['seed'])==s]) for s in spec[task.lower()]['seeds']]
                    item[metric+'_mean']=float(np.mean(values));item[metric+'_seed_macro_std']=float(np.std(seedmeans,ddof=1))
            summary.append(item)
        for dataset in spec['datasets']:
            part=[r for r in rows if r['task']==task and r['dataset']==dataset]
            for baseline in [a for a in WIDTHS if a!='fmt_v8_100']:
                differences=[float(next(r['f1'] for r in part if r['arm']=='fmt_v8_100' and int(r['seed'])==s))-
                    float(next(r['f1'] for r in part if r['arm']==baseline and int(r['seed'])==s)) for s in spec[task.lower()]['seeds']]
                paired.append(dict(task=task,dataset=dataset,method='fmt_v8_100',baseline=baseline,
                    f1_gain_mean=float(np.mean(differences)),f1_gain_seed_std=float(np.std(differences,ddof=1))))
    frozen.write_csv(root/'per_run_metrics.csv',rows);frozen.write_csv(root/'summary.csv',summary);frozen.write_csv(root/'paired_f1.csv',paired)
    remaining=[str(p) for suffix in ('*.pt','*.pth','*.ckpt') for p in root.rglob(suffix)]
    assert not remaining,remaining
    frozen.write_json(root/'final_audit.json',dict(status='PASS',rows=len(rows),shards=90,
        summary_sha256=frozen.sha(root/'summary.csv'),no_weight_files=True,source_identity=source_identity(config)))


def runtime(spec,config,phase,state,code=None):
    root=Path(spec['output_root']);events=root/'lifecycle_events';events.mkdir(parents=True,exist_ok=True)
    row=dict(experiment=spec['experiment'],phase=phase,state=state,exit_code=code,
        time=datetime.now(timezone.utc).isoformat(),job=os.environ.get('SLURM_JOB_ID'),array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        host=socket.gethostname(),device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',**source_identity(config))
    frozen.write_json(events/f"{row['job']}_{state}_{time.time_ns()}.json",row)
    append_record(root/'runtime_events.jsonl',row,'FMTv8 1.1 runtime');append_record('docs/ibex_run_registry.md',row,'FMTv8 1.1 runtime')


def submit(spec,config,phase,dependency=None):
    root=Path(spec['output_root']);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True);gpu=phase in ('smoke','Task35')
    cmd=['sbatch','--parsable','--cpus-per-task=4','--mem=64G','--time='+('08:00:00' if phase=='Task35' else '02:00:00'),
        '--job-name=fmtv8-'+phase,'--output='+str(logs/(phase+'.%A_%a.out')),'--error='+str(logs/(phase+'.%A_%a.err'))]
    if gpu:cmd+=['--gres=gpu:1','--constraint=v100']
    if phase in ('Task1','Task35'):cmd+=['--array=0-29%'+('12' if phase=='Task1' else '6')]
    if dependency:cmd+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    cmd+=['ibex_bash/mainexp_task135_fmtv8_1p1.sh',phase,config]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    row=dict(experiment=spec['experiment'],phase=phase,job_id=job,command=cmd,submitted_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config,expected_device='V100' if gpu else 'CPU',**source_identity(config))
    append_record(root/'submissions.jsonl',row,'FMTv8 1.1 submitted');append_record('docs/ibex_run_registry.md',row,'FMTv8 1.1 submitted')
    print(json.dumps(row),flush=True);return job


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','smoke','Task1','Task35','audit','runtime','submit'])
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int);p.add_argument('--submit-phase');p.add_argument('--dependency')
    a=p.parse_args();s=json.loads(Path(a.config).read_text());configure()
    if a.phase in ('Task1','Task35'):run_chain(s,a.config,a.phase,a.index)
    elif a.phase=='runtime':runtime(s,a.config,a.runtime_phase,a.state,a.exit_code)
    elif a.phase=='submit':submit(s,a.config,a.submit_phase,a.dependency)
    else:{'preflight':preflight,'smoke':smoke,'audit':audit}[a.phase](s,a.config)


if __name__=='__main__':main()
