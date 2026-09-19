"""Submit the authorized six fits exactly once, with independent preflight dependencies."""
from pathlib import Path
import json
import shutil
import subprocess
from datetime import datetime, timezone
from experiments import Task4C_V2NewLabelTraining_1_1 as run


def submit():
    s = run.spec(); root = Path(s['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(), 'Already submitted; do not duplicate jobs'
    assert shutil.disk_usage(root).free > 100*2**30
    jobs = {}
    def add(name, phase, deps=(), array=None, gpu=False, wall='02:00:00', memory='48G'):
        cmd = ['sbatch', '--parsable', '--propagate=NONE', '--kill-on-invalid-dep=yes', '--cpus-per-task=4',
            '--mem='+memory, '--time='+wall, '--job-name=NL11_'+name,
            '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: cmd += ['--dependency=afterok:'+':'.join(deps)]
        if array: cmd += ['--array='+array]
        if gpu: cmd += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
        cmd += ['ibex_bash/task4c_v2_newlabel_training_1p1.sh', phase]
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]; assert job.isdigit()
        jobs[name] = job
        run.old.fmt_old.baseline.append_line(root/'submissions.jsonl', json.dumps(dict(name=name, job=job,
            command=cmd, at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        run.write(root/'submission.json', dict(identity=run.identity(), jobs=jobs, seed=96611, formal_fits=6,
            index_map={str(i):c['id'] for i,c in enumerate(s['candidates'])}, selection=s['selection']))
        print(name, job, flush=True)
        return job
    neighbor = add('neighbors', 'neighbors', array='0-1%2')
    source = add('source', 'verify-source', [neighbor])
    checks = add('preflight', 'preflight', [source], array='0-5%6', gpu=True, wall='01:00:00')
    for i in range(6):
        add('train'+str(i), 'train', [checks+'_'+str(i)], array=str(i), gpu=True, wall='7-00:00:00', memory='96G' if i>=4 else '32G')
    add('audit', 'audit-results', [jobs['train'+str(i)] for i in range(6)])


if __name__ == '__main__': submit()
