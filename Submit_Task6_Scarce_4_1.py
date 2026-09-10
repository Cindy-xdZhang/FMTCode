"""Register separate validation search and frozen final evaluation on Ibex."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

from FMT_Utils.FlowMapData_3D import sha256


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['search','final'])
    args=p.parse_args()
    config='config/Verify_Task6_ScarceGeneralization_4.1.json'
    spec=json.loads(Path(config).read_text())
    root=Path(spec['output_root']).resolve();root.mkdir(parents=True,exist_ok=True)
    ledger=root/f'submissions_{args.stage}.jsonl'
    if ledger.exists(): raise FileExistsError(ledger)
    (root/'logs').mkdir(exist_ok=True)
    if args.stage=='search':
        shutil.copyfile(config,root/'config.frozen.json')
    else:
        assert sha256(config)==sha256(root/'config.frozen.json')
        assert json.loads((root/'selection.json').read_text())['test_read'] is False
        shutil.copyfile(root/'selection.json',root/'selection.before_test.json')
    count=len(spec['datasets'])*len(spec['train_sizes'])
    phases=[('preflight',4,12,'00:20:00',None,False),('prepare',4,16,'01:00:00','0-8%3',False),
            ('search',4,16,'04:00:00',f'0-{count-1}%18',True),('select',2,4,'00:15:00',None,False)] if args.stage=='search' else [
            ('final',4,16,'04:00:00',f'0-{count-1}%18',True),('audit',8,24,'02:00:00',None,False)]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    previous=None
    for phase,cpu,mem,limit,array,gpu in phases:
        command=['sbatch','--parsable',f'--job-name=t6-scarce-{phase}','--nodes=1','--ntasks=1',
            f'--cpus-per-task={cpu}',f'--mem={mem}G',f'--time={limit}',
            f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array: command += [f'--array={array}']
        if previous: command += [f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes']
        if gpu: command += ['--gres=gpu:1','--constraint=a100|v100','--exclude=gpu203-02-r']
        command += ['ibex_bash/task6_scarce_4p1.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,experiment=spec['experiment'],task=6,phase=phase,
            submitted=datetime.datetime.now().astimezone().isoformat(),config=config,config_sha256=sha256(config),
            git_commit=commit,device='GPU A100/V100' if gpu else 'CPU',array=array,command=command)
        for target,text in [(ledger,json.dumps(row)+'\n'),(Path('docs/ibex_run_registry.md'),
            f"\n- {row['submitted']} | {job} | {spec['experiment']} Task6 {phase} | {config} | {commit} | {row['device']} array={array} | SUBMITTED\n")]:
            with target.open('a',encoding='utf-8') as f:
                f.write(text);f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True);previous=job
