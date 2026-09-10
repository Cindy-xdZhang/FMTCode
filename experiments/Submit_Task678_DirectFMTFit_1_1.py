"""Submit fit diagnostics early while enlarged training data are integrated."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
from FMT_Utils.FlowMapData_3D import sha256
from experiments.Build_Task678_DirectFMTFit_1_1 import DEFAULT_CONFIG
from experiments.Submit_Task678_FlowMap_1_1 import append_registry


def event(spec,config,phase,state,code):
    device='CPU'
    if phase.startswith('train'):
        import torch
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable'
    item=dict(experiment=spec['experiment'],phase=phase,state=state,time=datetime.datetime.now().astimezone().isoformat(),
        job_id=os.getenv('SLURM_ARRAY_JOB_ID',os.getenv('SLURM_JOB_ID','local')),array_index=os.getenv('SLURM_ARRAY_TASK_ID',''),
        node=socket.gethostname(),device=device,exit_code=code,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=config,config_sha256=sha256(config))
    root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'job_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(item)+'\n')
    append_registry(f"\n- {spec['experiment']} `{item['job_id']}_{item['array_index']}` | {phase} | {state} | {item['time']} | {item['node']} | {device} | exit={code} | commit `{item['git_commit']}` | config SHA256 `{item['config_sha256']}`\n")
    print(json.dumps(item),flush=True)


def submit(spec,config,dry_run=False):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=Path(spec['output_root']).resolve();registry=root/'submissions.jsonl'
    if registry.exists() and not dry_run:
        raise FileExistsError('Submissions already exist; do not duplicate jobs')
    if not dry_run:
        (root/'logs').mkdir(parents=True,exist_ok=True)
    n=len(spec['datasets'])*len(spec['seeds']);jobs={}
    assert list(spec['conditions'])==['memorize16','small_base','large_base','large_expanded']
    phases=[('smoke',[],['--time=00:20:00','--cpus-per-task=4','--mem=16G']),
        ('prepare',['smoke'],[f'--array=0-{len(spec["datasets"])-1}%4','--time=00:45:00','--cpus-per-task=4','--mem=24G']),
        ('build',['prepare'],[f'--array=0-{len(spec["datasets"])-1}%3','--time=04:00:00','--cpus-per-task=8','--mem=48G']),
        ('train_base',['prepare'],[f'--array=0-{3*n-1}%6','--time=12:00:00','--cpus-per-task=4','--mem=32G','--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']),
        ('train_expanded',['build'],[f'--array={3*n}-{4*n-1}%3','--time=12:00:00','--cpus-per-task=4','--mem=48G','--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']),
        ('audit',['train_base','train_expanded'],['--time=02:00:00','--cpus-per-task=4','--mem=48G'])]
    for phase,deps,extra in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1',f'--job-name=FMTfit_{phase}',
            f'--output={root}/logs/{phase}.%A_%a.out',f'--error={root}/logs/{phase}.%A_%a.err']+extra
        if deps:
            command+=['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        command+=['ibex_bash/verify_task678_directfmtfit_1p1.sh',phase,config,commit]
        if dry_run:
            jobs[phase]='DRY_'+phase;print(json.dumps(command));continue
        job=subprocess.check_output(command,text=True).strip().split(';')[0];assert job.isdigit();jobs[phase]=job
        item=dict(experiment=spec['experiment'],phase=phase,job_id=job,submitted=datetime.datetime.now().astimezone().isoformat(),
            config=config,config_sha256=sha256(config),git_commit=commit,command=command,
            expected_device='one A100 or V100 per array element' if phase.startswith('train') else 'CPU')
        with registry.open('a',encoding='utf-8') as f:
            f.write(json.dumps(item)+'\n');f.flush();os.fsync(f.fileno())
        append_registry(f"\n- **SUBMITTED {spec['experiment']}** `{job}` | {phase} | {item['submitted']} | config `{config}` SHA256 `{item['config_sha256']}` | commit `{commit}` | {item['expected_device']} | dependencies `{','.join(jobs[d] for d in deps) or 'none'}`\n")
        print(json.dumps(item),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--event',choices=['RUNNING','COMPLETED','FAILED']);p.add_argument('--phase');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text())
    event(spec,a.config,a.phase,a.event,a.exit_code) if a.event else submit(spec,a.config,a.dry_run)

if __name__=='__main__':
    main()
