"""Four replay checks, 24 bounded screens, then at most four selected refinements."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
from experiments import Task4C_FMTOptimization_1_1 as run


def submit():
    s = run.spec(); root = Path(s['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(), 'Already submitted'
    jobs = {}
    def add(name, phase, deps=(), array=None, gpu=False, wall='01:00:00'):
        cmd = ['sbatch', '--parsable', '--propagate=NONE', '--kill-on-invalid-dep=yes', '--cpus-per-task=4',
            '--mem=32G', '--time='+wall, '--job-name=FMTO11_'+name,
            '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: cmd += ['--dependency=afterok:'+':'.join(deps)]
        if array: cmd += ['--array='+array]
        if gpu: cmd += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
        cmd += ['ibex_bash/task4c_fmt_optimization_1p1.sh', phase]
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]; assert job.isdigit(); jobs[name] = job
        run.base.old.fmt_old.baseline.append_line(root/'submissions.jsonl', json.dumps(dict(name=name, job=job,
            command=cmd, at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        run.write(root/'submission.json', dict(identity=run.identity(), jobs=jobs, seed=96611, screen_fits=24,
            maximum_final_fits=4, max_concurrent_gpu=6, screen_index_map={str(i):dict(model=s['candidates'][i//6]['id'],
                recipe=s['recipes'][i%6]['id']) for i in range(24)}))
        print(name, job, flush=True)
        return job
    source = add('source', 'verify-source')
    checks = add('replay', 'preflight', [source], array='0-3%4', gpu=True)
    screen = add('screen', 'screen', [checks], array='0-23%6', gpu=True, wall='02:00:00')
    select = add('select', 'select', [screen])
    final = add('final', 'final', [select], array='0-3%4', gpu=True, wall='08:00:00')
    add('summary', 'summarize', [final])


if __name__ == '__main__': submit()
