"""Submit checks or the full paired experiment, registering each job at once."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

from experiments.Run_Task135_GeometricControls import sha, write_json

CONFIG = 'config/Verify_Task1235_AngleFeatures_1.1.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['checks', 'performance'], required=True)
    args = parser.parse_args()
    spec = json.loads(Path(CONFIG).read_text())
    root = Path(spec['output_root'])
    marker = root / ('submitted_' + args.stage + '.json')
    if marker.exists():
        raise FileExistsError('Stage already submitted; inspect recorded jobs before retrying.')
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    if args.stage == 'checks':
        shutil.copyfile(CONFIG, root / 'run_config.json')
        shutil.copyfile('SOURCE_MANIFEST.sha256', root / 'SOURCE_MANIFEST.sha256')
        phases = [('smoke', [], ['--time=00:20:00']),
                  ('preflight', [], ['--time=01:00:00'])]
    else:
        for name in ['smoke', 'preflight']:
            result = json.loads((root / (name + '.json')).read_text())
            assert result['status'] == 'PASS'
        assert json.loads((root / 'preflight.json').read_text())['config_sha256'] == sha(CONFIG)
        assert sha(root / 'SOURCE_MANIFEST.sha256') == sha('SOURCE_MANIFEST.sha256')
        gpu = ['--gres=gpu:1', '--constraint=a100|v100', '--exclude=gpu203-02-r']
        phases = [('Task1', [], ['--array=0-29%6', '--time=01:00:00']),
                  ('Task2', [], ['--array=0-29%6', '--time=04:00:00'] + gpu),
                  ('Task35', [], ['--array=0-29%6', '--time=06:00:00'] + gpu),
                  ('audit', ['Task1', 'Task2', 'Task35'], ['--time=00:30:00'])]
    jobs = {}
    for phase, dependencies, extra in phases:
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=32G',
                   '--job-name=FMTAngles_' + phase,
                   f'--output={root}/logs/{phase}.%A_%a.out',
                   f'--error={root}/logs/{phase}.%A_%a.err'] + extra
        if dependencies:
            command += ['--dependency=afterok:' + ':'.join(jobs[d] for d in dependencies)]
        command += ['ibex_bash/verify_task1235_angles_1p1.sh', phase]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        assert job.isdigit()
        jobs[phase] = job
        item = {'experiment': spec['experiment'], 'phase': phase, 'job': job,
                'submitted': datetime.datetime.now().astimezone().isoformat(),
                'config': CONFIG, 'config_sha256': sha(CONFIG),
                'git_commit': Path('SOURCE_COMMIT.txt').read_text().strip(),
                'source_manifest_sha256': sha('SOURCE_MANIFEST.sha256'),
                'expected_device': 'A100 or V100' if phase in ('Task2','Task35') else 'CPU',
                'command': command}
        for path in [root / 'submissions.jsonl', Path('docs/ibex_run_registry.md')]:
            with path.open('a', encoding='utf-8') as handle:
                payload = json.dumps(item)
                handle.write(payload + '\n' if path.suffix == '.jsonl' else '\n- **SUBMITTED angles 1.1** ' + payload + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        write_json(marker, jobs)
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
