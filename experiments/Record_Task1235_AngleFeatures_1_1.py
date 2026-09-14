"""Durable start/end records for each Slurm process, including failed runs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    parser.add_argument('--event', choices=['start', 'end'], required=True)
    parser.add_argument('--exit-code', type=int)
    args = parser.parse_args()
    config = Path('config/Verify_Task1235_AngleFeatures_1.1.json')
    spec = json.loads(config.read_text())
    device = 'CPU'
    if args.phase in ['Task2', 'Task35']:
        probe = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'],
                               capture_output=True, text=True)
        device = probe.stdout.strip() or 'GPU query unavailable'
    item = {'experiment': spec['experiment'], 'phase': args.phase, 'event': args.event,
            'time': datetime.datetime.now().astimezone().isoformat(),
            'job': os.environ.get('SLURM_JOB_ID'), 'array_job': os.environ.get('SLURM_ARRAY_JOB_ID'),
            'array_task': os.environ.get('SLURM_ARRAY_TASK_ID'), 'node': socket.gethostname(),
            'device': device, 'config': str(config),
            'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
            'git_commit': Path('SOURCE_COMMIT.txt').read_text().strip(), 'exit_code': args.exit_code}
    for path in [Path(spec['output_root']) / 'lifecycle.jsonl', Path('docs/ibex_run_registry.md')]:
        with path.open('a', encoding='utf-8') as handle:
            record = json.dumps(item)
            handle.write(record + '\n' if path.suffix == '.jsonl' else '\n- **Angles 1.1 ' + args.event.upper() + '** ' + record + '\n')
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
