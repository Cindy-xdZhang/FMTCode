"""Single-seed ball-query ablation on frozen Task4-c v2, reusing the frozen 2.1 trainer."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone
from types import FunctionType, SimpleNamespace
from experiments import Task4C_FixedDatasetFMT_2_1 as baseline
from FMT_Utils import Task4C_BallQuery_2_2 as ball

CONFIG = 'config/Ablation_Task4C_BallQuery_2.2.json'
FILES = tuple(dict.fromkeys(baseline.FILES+('experiments/Task4C_BallQuery_2_2.py', 'FMT_Utils/Task4C_BallQuery_2_2.py',
                  'experiments/Analyze_Task4C_BallQuery_2_2.py', 'ibex_bash/task4c_ball_query_2p2.sh', 'docs/Task4C_ball_query_protocol_2.2.md')))


def identity(config):
    # Host belongs to runtime provenance, not scientific identity shared across Slurm nodes.
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=baseline.sha(config), sources={p: baseline.sha(p) for p in FILES})


def environment(candidate):
    return dict(baseline.environment(candidate), Dataset=ball.Dataset, identity=identity)


def invoke(name, *args):
    adapter = SimpleNamespace(**dict(baseline.v2.__dict__, Dataset=ball.Dataset, build_neighbor_tables=ball.build_neighbor_tables))
    globals_ = dict(baseline.__dict__, v2=adapter, environment=environment, identity=identity)
    f = getattr(baseline, name)
    return FunctionType(f.__code__, globals_, argdefs=f.__defaults__)(*args)


def gpu_check(spec, config):
    invoke('gpu_check', spec, config)
    path = Path(spec['output'])/'gpu_check.json'; report = json.loads(path.read_text())
    import torch
    from FMT_Utils.FMTNoConvolution_1_1 import assert_no_fmt_convolution
    for c in spec['candidates']:
        model = baseline.method.make_model(c['base_method'])
        assert_no_fmt_convolution(model)
        report['checks'][c['id']]['network'] = str(model)
        report['checks'][c['id']]['no_convolution'] = True
    report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
    baseline.write(path, report)


def submit(spec, config):
    assert spec['final']['seeds'] == [96611], 'One declared seed only'
    root = Path(spec['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(), 'Already submitted'
    jobs = {}
    phases = [('verify-source', None, False, '00:30:00'), ('gpu-check', None, True, '00:20:00'),
              ('train', f'0-{len(spec["candidates"])-1}%{spec["concurrency"]}', True, '24:00:00'),
              ('audit-results', None, False, '00:30:00')]
    for phase, array, gpu, wall in phases:
        command = ['sbatch', '--parsable', '--propagate=NONE', '--kill-on-invalid-dep=yes', '--job-name=Ball22_'+phase,
                   '--cpus-per-task=4', '--mem=64G', '--time='+wall, '--output='+str(root/'logs/%x_%A_%a.out'),
                   '--error='+str(root/'logs/%x_%A_%a.err')]
        if jobs: command += ['--dependency=afterok:'+list(jobs.values())[-1]]
        if array: command += ['--array='+array]
        if gpu: command += ['--gres=gpu:1', '--constraint=v100', '--exclude=gpu213-18']
        command += ['ibex_bash/task4c_ball_query_2p2.sh', phase, config]
        jobs[phase] = subprocess.check_output(command, text=True).strip().split(';')[0]
        baseline.append_line(root/'submissions.jsonl', json.dumps(dict(phase=phase, job=jobs[phase], command=command,
                             submitted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        baseline.write(root/'submission.json', dict(identity=identity(config), jobs=jobs))
    print(json.dumps(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['verify-source', 'gpu-check', 'train', 'audit-results', 'submit', 'runtime'])
    p.add_argument('--config', default=CONFIG); p.add_argument('--index', type=int, default=0)
    p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args(); spec = json.loads(Path(a.config).read_text())
    if a.phase == 'submit': submit(spec, a.config)
    elif a.phase == 'gpu-check': gpu_check(spec, a.config)
    elif a.phase == 'runtime':
        baseline.append_line(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(a.config), phase=a.runtime_phase,
            state=a.state, exit_code=a.exit_code, job_id=os.environ.get('SLURM_JOB_ID'), array_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
            hostname=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    elif a.phase == 'train': invoke('train', spec, a.config, a.index)
    else: invoke(a.phase.replace('-', '_'), spec, a.config)


if __name__ == '__main__': main()
