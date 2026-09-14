"""Durable start/end records for each Slurm process, including failed runs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def append_record(path, item, label):
    """Serialize cooperating writers on the shared Ibex filesystem."""
    path = Path(path)
    with path.open('a', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            record = json.dumps(item)
            handle.write(record + '\n' if path.suffix == '.jsonl' else '\n- **' + label + '** ' + record + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    root = Path(spec['output_root'])
    events = root / 'lifecycle_events'
    events.mkdir(parents=True, exist_ok=True)
    # Each process also has an independent durable record, so aggregate logs
    # are never the sole evidence for an event.
    event_file = events / f"{item['job']}_{args.event}_{time.time_ns()}.json"
    with event_file.open('x', encoding='utf-8') as handle:
        json.dump(item, handle)
        handle.flush()
        os.fsync(handle.fileno())
    for path in [root / 'lifecycle.jsonl', Path('docs/ibex_run_registry.md')]:
        append_record(path, item, 'Angles 1.1 ' + args.event.upper())
    print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
