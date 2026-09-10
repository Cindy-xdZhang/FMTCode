"""Append start/end evidence for every Slurm process, including failures."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess

from experiments.Run_Task135_GeometricControls import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    parser.add_argument('--event', choices=['start','end'], required=True)
    parser.add_argument('--exit-code', type=int)
    args = parser.parse_args()
    config = Path('config/Verify_Task1235_ObjectiveFMTnTDO_2.1.json')
    spec = json.loads(config.read_text())
    gpu = 'CPU'
    if args.phase in ('Task2','Task35'):
        probe = subprocess.run(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],capture_output=True,text=True)
        gpu = probe.stdout.strip() or 'GPU query unavailable'
    item = {'experiment':spec['experiment'], 'event':args.event,'phase':args.phase,
            'time':datetime.datetime.now().astimezone().isoformat(), 'job':os.environ.get('SLURM_JOB_ID'),
            'array_job':os.environ.get('SLURM_ARRAY_JOB_ID'), 'array_task':os.environ.get('SLURM_ARRAY_TASK_ID'),
            'node':socket.gethostname(),'device':gpu,'config':str(config),'config_sha256':sha(config),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'exit_code':args.exit_code}
    root = Path(spec['output_root'])
    for path in [root/'lifecycle.jsonl',Path('docs/ibex_run_registry.md')]:
        with path.open('a',encoding='utf-8') as f:
            record = json.dumps(item)
            f.write(record+'\n' if path.suffix=='.jsonl' else '\n- **nTDO v2 '+args.event.upper()+'** '+record+'\n')
            f.flush();os.fsync(f.fileno())
    print(json.dumps(item),flush=True)


if __name__ == '__main__': main()
