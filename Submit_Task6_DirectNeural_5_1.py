"""Submit and immediately register the preregistered Task6 5.1 dependency chain."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

from FMT_Utils.FlowMapData_3D import sha256


def main():
    config = 'config/Verify_Task6_DirectNeural_5.1.json'
    spec = json.loads(Path(config).read_text())
    root = Path(spec['output_root']).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    shutil.copyfile(config, root/'config.frozen.json')
    ledger = root/'submissions.jsonl'
    nd = len(spec['datasets'])
    stages = [('preflight', 4, 12, '00:15:00', None, False),
        ('prepare', 4, 12, '00:30:00', f'0-{nd-1}%3', False),
        ('fit_check', 4, 16, '00:30:00', f'0-{nd-1}%9', True),
        ('search', 4, 20, '01:00:00', f'0-{nd*len(spec["candidates"])-1}%18', True),
        ('select', 2, 4, '00:15:00', None, False),
        ('final', 4, 24, '01:00:00', f'0-{nd*len(spec["train_sizes"])*len(spec["seeds"])-1}%18', True),
        ('audit', 4, 24, '01:00:00', f'0-{nd-1}%9', False),
        ('merge', 2, 4, '00:15:00', None, False)]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    previous = None
    for phase, cpu, mem, limit, array, gpu in stages:
        command = ['sbatch', '--parsable', f'--job-name=t6-direct-{phase}', '--nodes=1', '--ntasks=1',
            f'--cpus-per-task={cpu}', f'--mem={mem}G', f'--time={limit}',
            f'--output={root}/logs/{phase}_%A_%a.out', f'--error={root}/logs/{phase}_%A_%a.err']
        if array:
            command += [f'--array={array}']
        if previous:
            command += [f'--dependency=afterok:{previous}', '--kill-on-invalid-dep=yes']
        if gpu:
            command += ['--gres=gpu:1', '--constraint=a100|v100', '--exclude=gpu203-02-r']
        command += ['ibex_bash/task6_direct_neural_5p1.sh', phase, config]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        row = dict(job_id=job, experiment=spec['experiment'], task=6, phase=phase,
            submitted=datetime.datetime.now().astimezone().isoformat(), config=config,
            config_sha256=sha256(config), git_commit=commit, array=array, command=command,
            device='GPU A100/V100' if gpu else 'CPU', dependency=previous)
        note = f"\n- {row['submitted']} | {job} | {spec['experiment']} Task6 {phase} | {config} SHA256={row['config_sha256']} | {commit} | {row['device']} array={array} | SUBMITTED\n"
        for target, content in [(ledger, json.dumps(row)+'\n'), (Path('docs/ibex_run_registry.md'), note)]:
            with target.open('a', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
        print(json.dumps(row), flush=True)
        previous = job


if __name__ == '__main__':
    main()
