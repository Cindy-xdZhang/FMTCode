"""Directional Fourier extension on an already audited shared dataset."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
from FMT_Utils.FlowMapData_3D import sha256
from experiments.Run_Task678_VectorFMT_1_1 import DEFAULT_CONFIG
from experiments.Run_Task678_VectorFMT_1_1 import matrix
from experiments.Submit_Task678_FlowMap_1_1 import append_registry
from experiments.Submit_Task678_DirectFMTFit_1_1 import event


def submit(spec, config, dry_run=False):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=Path(spec['output_root']).resolve(); registry=root/'submissions.jsonl'
    if registry.exists() and not dry_run:
        raise FileExistsError('Existing submissions must not be duplicated')
    if not dry_run:
        (root/'logs').mkdir(parents=True,exist_ok=True)
    d=len(spec['datasets']); n=len(matrix(spec)); jobs={}
    phases=[('smoke',[],['--time=00:20:00','--cpus-per-task=4','--mem=16G']),
        ('train',['smoke'],[f'--array=0-{n-1}%6','--time=02:00:00','--cpus-per-task=4','--mem=48G',
             '--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']),
        ('affine',['smoke'],[f'--array=0-{d-1}%3','--time=01:00:00','--cpus-per-task=4','--mem=48G']),
        ('audit',['train','affine'],['--time=03:00:00','--cpus-per-task=4','--mem=48G'])]
    for phase,deps,extra in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1',f'--job-name=FMTvec_{phase}',
            f'--output={root}/logs/{phase}.%A_%a.out',f'--error={root}/logs/{phase}.%A_%a.err']+extra
        if deps:
            command+=['--dependency=afterok:'+':'.join(jobs[k] for k in deps)]
        command+=['ibex_bash/verify_task678_vectorfmt_1p1.sh',phase,config,commit]
        if dry_run:
            jobs[phase]='DRY_'+phase; print(json.dumps(command)); continue
        job=subprocess.check_output(command,text=True).strip().split(';')[0]; assert job.isdigit(); jobs[phase]=job
        item=dict(experiment=spec['experiment'],phase=phase,job_id=job,submitted=datetime.datetime.now().astimezone().isoformat(),
            config=config,config_sha256=sha256(config),git_commit=commit,command=command,
            expected_device='one A100 or V100 per array element' if phase=='train' else 'CPU')
        with registry.open('a',encoding='utf-8') as f:
            f.write(json.dumps(item)+'\n'); f.flush(); os.fsync(f.fileno())
        append_registry(f"\n- **SUBMITTED {spec['experiment']}** `{job}` | {phase} | {item['submitted']} | config `{config}` SHA256 `{item['config_sha256']}` | commit `{commit}` | {item['expected_device']} | dependencies `{','.join(jobs[d] for d in deps) or 'none'}`\n")
        print(json.dumps(item),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default=DEFAULT_CONFIG); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--event',choices=['RUNNING','COMPLETED','FAILED']); p.add_argument('--phase'); p.add_argument('--exit-code',type=int)
    a=p.parse_args(); spec=json.loads(Path(a.config).read_text())
    event(spec,a.config,a.phase,a.event,a.exit_code) if a.event else submit(spec,a.config,a.dry_run)


if __name__ == '__main__':
    main()
