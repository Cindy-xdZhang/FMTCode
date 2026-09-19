"""Submit the 10 single-seed 3h comparisons with explicit success dependencies."""
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
from experiments import Task4C_BallQuery_2_5 as fmt
from experiments import Task4C_BallQueryBaselines_2_5 as baselines


def submit():
    spec = json.loads(Path(fmt.CONFIG).read_text()); root = Path(spec['output'])
    assert spec['final']['seeds'] == [96611] and len(spec['candidates']) == 4
    for config in baselines.CONFIGS.values():
        for family in baselines.FAMILIES:
            b = baselines.load_spec(config, family)
            assert b['training']['seeds'] == [96611] and len(b['methods']) == 1
    assert shutil.disk_usage('.').free > 80*2**30
    (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(), 'Already submitted'
    jobs = {}
    def add(key, driver, phase, deps=(), array=None, gpu=False, wall='01:00:00', memory='64G'):
        cmd = ['sbatch', '--parsable', '--propagate=NONE', '--kill-on-invalid-dep=yes', '--cpus-per-task=4',
               '--mem='+memory, '--time='+wall, '--job-name=BQ25_'+key,
               '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: cmd += ['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        if array: cmd += ['--array='+array]
        if gpu: cmd += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
        cmd += ['ibex_bash/task4c_ball_query_2p5.sh', driver, phase]
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]; assert job.isdigit()
        jobs[key] = job
        fmt.baseline.append_line(root/'submissions.jsonl', json.dumps(dict(key=key, job=job, command=cmd, at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        fmt.baseline.write(root/'submission.json', dict(identity=fmt.identity(fmt.CONFIG), jobs=jobs, formal_fits=10, seed=96611,
            baseline_array={str(i): dict(neighbors=(6,16)[i//3], family=baselines.FAMILIES[i%3]) for i in range(6)}))
        print(key, job, flush=True)
    add('source', 'fmt', 'verify-source', wall='01:00:00')
    add('fmt_check', 'fmt', 'gpu-check', ['source'], gpu=True, wall='00:20:00')
    add('fmt_train', 'fmt', 'train', ['fmt_check'], '0-3%4', True, '24:00:00')
    add('fmt_audit', 'fmt', 'audit-results', ['fmt_train'])
    add('export', 'baselines', 'export', ['source'], '0-3%4', wall='02:00:00', memory='96G')
    add('prepare', 'baselines', 'prepare', ['export'], '0-1%2')
    add('baseline_check', 'baselines', 'gpu-check', ['prepare'], '0-5%6', True, '00:30:00')
    add('conv_encode', 'baselines', 'encode', ['baseline_check'], '0-3%4', True, '04:00:00')
    add('baseline_train', 'baselines', 'train', ['conv_encode'], '0-5%6', True, '3-00:00:00')
    add('baseline_audit', 'baselines', 'summarize', ['baseline_train'], '0-5%6')
    print(json.dumps(jobs), flush=True)


if __name__ == '__main__': submit()
