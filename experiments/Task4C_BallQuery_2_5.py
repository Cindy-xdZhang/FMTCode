"""FMT entry point for the single-seed 3h experiment, with frozen training code."""
import argparse
import json
import os
from pathlib import Path
import socket
from datetime import datetime, timezone
from types import FunctionType, SimpleNamespace
from experiments import Task4C_BallQuery_2_2 as previous
from FMT_Utils import Task4C_BallQuery_2_5 as ball

baseline = previous.baseline
CONFIG = 'config/Ablation_Task4C_BallQuery_2.5.json'
FILES = tuple(dict.fromkeys(previous.FILES+('FMT_Utils/Task4C_BallQuery_2_5.py', 'experiments/Task4C_BallQuery_2_5.py',
    'experiments/Task4C_BallQueryBaselines_2_5.py', 'experiments/Submit_Task4C_BallQuery_2_5.py',
    'ibex_bash/task4c_ball_query_2p5.sh', 'docs/Task4C_ball_query_protocol_2.5.md', CONFIG)))


def identity(config):
    return FunctionType(previous.identity.__code__, dict(previous.__dict__, FILES=FILES))(config)


def environment(candidate):
    return dict(baseline.environment(candidate), Dataset=ball.Dataset, identity=identity)


def invoke(name, *args):
    adapter = SimpleNamespace(**dict(baseline.v2.__dict__, Dataset=ball.Dataset, build_neighbor_tables=ball.build_neighbor_tables))
    fn = getattr(baseline, name)
    return FunctionType(fn.__code__, dict(baseline.__dict__, v2=adapter, identity=identity, environment=environment), argdefs=fn.__defaults__)(*args)


def gpu_check(spec, config):
    return FunctionType(previous.gpu_check.__code__, dict(previous.__dict__, invoke=invoke))(spec, config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['verify-source', 'gpu-check', 'train', 'audit-results', 'runtime'])
    p.add_argument('--config', default=CONFIG); p.add_argument('--index', type=int, default=0)
    p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args(); spec = json.loads(Path(a.config).read_text())
    assert spec['final']['seeds'] == [96611]
    if a.phase == 'gpu-check': gpu_check(spec, a.config)
    elif a.phase == 'runtime':
        Path(spec['output']).mkdir(parents=True, exist_ok=True)
        baseline.append_line(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(a.config), phase=a.runtime_phase,
            state=a.state, exit_code=a.exit_code, job_id=os.environ.get('SLURM_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
            host=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    elif a.phase == 'train': invoke('train', spec, a.config, a.index)
    else: invoke(a.phase.replace('-', '_'), spec, a.config)


if __name__ == '__main__': main()
