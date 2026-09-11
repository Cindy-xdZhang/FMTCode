"""Submit versioned tuning phases and log all processes immediately."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from FMT_Utils.FlowMapData_3D import sha256
from Submit_Task6_PNNTrans_1_1 import append_locked


def runtime(config,phase,state,exit_code=None):
    spec=json.loads(Path(config).read_text())
    now=datetime.datetime.now().astimezone().isoformat()
    array=os.getenv('SLURM_ARRAY_TASK_ID')
    job=os.getenv('SLURM_ARRAY_JOB_ID',os.getenv('SLURM_JOB_ID','local'))
    name=job+(f'_{array}' if array is not None else '')
    device='CPU'
    if phase in ('capacity_check','screen','refine','final'):
        import torch
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable'
    append_locked('docs/ibex_run_registry.md',f'\n- {now} | {name} | {spec["experiment"]} Task6 {phase} | '
                  f'{state} node={socket.gethostname()} device={device} exit={exit_code}\n')


def submit(mode,config):
    spec=json.loads(Path(config).read_text())
    root=Path(spec['output_root']).resolve()
    nd=len(spec['datasets'])
    if mode=='search':
        root.mkdir(parents=True,exist_ok=False)
        (root/'logs').mkdir()
        shutil.copyfile(config,root/'config.frozen.json')
        stages=[('preflight',4,16,'00:20:00',None,False),
            ('prepare',4,16,'00:20:00',None,False),
            ('capacity_check',4,32,'00:30:00','0-1%2',True),
            ('screen',4,32,'01:30:00',f'0-{len(spec["screen_datasets"])*len(spec["candidates"])-1}%18',True),
            ('promote',2,4,'00:10:00',None,False),
            ('refine',4,32,'03:00:00',f'0-{nd*spec["promote_count"]-1}%18',True),
            ('select',2,4,'00:10:00',None,False),
            ('advance',2,4,'00:10:00',None,False)]
    else:
        from experiments.Task6_PNNTrans_1_1 import frozen_selection
        selection=frozen_selection(spec,config)
        assert selection['beats_both_baselines'] and selection['numeric_gate_passed']
        assert sha256(config)==sha256(root/'config.frozen.json')
        with (root/'final_submission.lock').open('x') as f:
            f.write(datetime.datetime.now().astimezone().isoformat())
        stages=[('final',4,32,'04:00:00',f'0-{nd*len(spec["train_sizes"])*len(spec["seeds"])-1}%18',True),
            ('audit',4,24,'01:00:00',f'0-{nd-1}%9',False),
            ('merge',2,4,'00:10:00',None,False)]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    previous=None
    for phase,cpu,mem,limit,array,gpu in stages:
        command=['sbatch','--parsable',f'--job-name=t6-pnn12-{phase}','--nodes=1','--ntasks=1',
            f'--cpus-per-task={cpu}',f'--mem={mem}G',f'--time={limit}',
            f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array:
            command += [f'--array={array}']
        if previous:
            command += [f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes']
        if gpu:
            command += ['--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']
        command += ['ibex_bash/task6_pnn_tuning_1p2.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,experiment=spec['experiment'],task=6,phase=phase,
            submitted=datetime.datetime.now().astimezone().isoformat(),config=config,
            config_sha256=sha256(config),git_commit=commit,array=array,command=command,
            device='GPU A100/V100' if gpu else 'CPU',dependency=previous)
        append_locked(root/'submissions.jsonl',json.dumps(row)+'\n')
        append_locked('docs/ibex_run_registry.md',f'\n- {row["submitted"]} | {job} | {spec["experiment"]} Task6 {phase} | '
            f'{config} SHA256={row["config_sha256"]} | {commit} | {row["device"]} array={array} | SUBMITTED\n')
        print(json.dumps(row),flush=True)
        previous=job


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['search','final','runtime'])
    parser.add_argument('--config',default='config/Verify_Task6_PNNTrans_1.2.json')
    parser.add_argument('--phase')
    parser.add_argument('--state')
    parser.add_argument('--exit-code',type=int)
    args=parser.parse_args()
    if args.mode=='runtime':
        runtime(args.config,args.phase,args.state,args.exit_code)
    else:
        submit(args.mode,args.config)
