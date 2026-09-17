"""Extend the frozen FPS search shortlist to 20, reusing the original 36 fits."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType

from experiments import Task4C_FPSAugmentSearch_1_1 as parent

CONFIG = 'config/Ablation_Task4C_FPSAugmentSearch_1.2.json'
write, sha, append_locked = parent.write, parent.sha, parent.append_locked


def load_spec(config):
    extension = json.loads(Path(config).read_text())
    assert sha(extension['parent_config']) == extension['parent_config_sha256']
    spec = parent.load_spec(extension['parent_config'])
    spec.update(extension)
    spec['refine'] = dict(spec['refine'], candidates=20)
    assert spec['reuse_candidates'] == 12 and len(spec['expected_top20']) == 20
    assert spec['refine']['seeds'] == [96612, 96613, 96711]
    assert spec['final_scope'] == 'selected_and_reference'
    return spec


def identity(config):
    result = parent.identity(config)
    for path in ('experiments/Task4C_FPSAugmentSearch_1_2.py',
                 'ibex_bash/task4c_fps_augment_search_1p2.sh'):
        result['sources'][path] = sha(path)
    return result


def call(name, *args, expected_identity=None, **kwargs):
    """Use frozen function bytecode; replace only bookkeeping/validation hooks."""
    fn = getattr(parent, name)
    env = dict(parent.__dict__, identity=identity, check_result=check_result)
    if expected_identity is not None:
        env['identity'] = lambda config: expected_identity
    return FunctionType(fn.__code__, env, argdefs=fn.__defaults__)(*args, **kwargs)


def source_identity(spec):
    return dict(git_commit=spec['parent_scientific_commit'],
                config_sha256=spec['parent_config_sha256'])


def check_result(spec, config, phase, candidate, seed, role='validation'):
    if phase == 'refine' and candidate['id'] in spec['expected_top20'][:12]:
        return call('check_result', spec, spec['parent_config'], phase, candidate,
                    seed, role, expected_identity=source_identity(spec))
    return call('check_result', spec, config, phase, candidate, seed, role)


def verify_source(spec):
    root = Path(spec['parent_output'])
    assert not (root/'selection.lock.json').exists(), 'Old final selection must remain suspended'
    audit = json.loads((root/'preflight_cuda.json').read_text())
    assert audit['complete'] and audit['identity']['git_commit'] == spec['parent_scientific_commit']
    # The old absolute source path points into its separate, frozen checkout.
    for name, digest in audit['identity']['sources'].items():
        path = Path(name)
        if path.is_absolute():
            path = path.relative_to(root.parents[1])
        assert sha(path) == digest, str(path)
    for name in ('data_audit.json', 'pilot.json'):
        report = json.loads((root/name).read_text())
        assert report['complete'] and report['identity']['git_commit'] == spec['parent_scientific_commit']
    frozen = parent.load_spec(spec['parent_config'])
    frozen['output'] = str(root)
    results = [call('check_result', frozen, spec['parent_config'], 'screen', c,
                    frozen['screen']['seed'], expected_identity=source_identity(spec))
               for c in parent.candidates(frozen)]
    results.sort(key=lambda r: (-r['validation']['f1'], -r['validation']['average_precision'], r['candidate']['id']))
    rows = [dict(candidate=r['candidate'], f1=r['validation']['f1'],
                 average_precision=r['validation']['average_precision']) for r in results]
    old = json.loads((root/'shortlist.json').read_text())
    assert rows == old['rankings'] and old['candidates'] == [r['candidate'] for r in rows[:12]]
    assert [r['candidate']['id'] for r in rows[:20]] == spec['expected_top20']
    return rows


def prepare(spec, config):
    root = Path(spec['output'])
    rows = verify_source(spec)
    call('reuse', spec, config)
    chosen = [r['candidate'] for r in rows[:20]]
    write(root/'shortlist.json', dict(complete=True, identity=identity(config),
          rankings=rows, candidates=chosen, test_read=False,
          parent_shortlist_sha256=sha(Path(spec['parent_output'])/'shortlist.json')))
    (root/'refine').mkdir()
    for candidate in chosen[:12]:
        (root/'refine'/candidate['id']).symlink_to(
            Path(spec['parent_output'])/'refine'/candidate['id'], target_is_directory=True)
    write(root/'extension_audit.json', dict(complete=True, identity=identity(config),
          source_screen_fits=264, reused_refinement_fits=36, added_refinement_fits=24,
          reused_candidates=chosen[:12], added_candidates=chosen[12:],
          training_function_unchanged=True, no_test_read=True))


def train(spec, config, index, phase):
    if phase == 'refine':
        assert 0 <= index < 24
        index += 36  # Full shortlist index: only ranks 13 through 20 are new.
    else:
        assert phase == 'final' and 0 <= index < 6
    call('train', spec, config, index, phase)


def select(spec, config):
    call('select', spec, config)


def merge(spec, config):
    call('merge', spec, config)
    path = Path(spec['output'])/'summary.json'
    result = json.loads(path.read_text())
    result['parent_output'] = spec['parent_output']
    result['parent_scientific_commit'] = spec['parent_scientific_commit']
    result['refinement_candidates'] = 20
    result['reused_refinement_fits'] = 36
    write(path, result)


def runtime(spec, config, phase, state, code):
    fn = parent.engine.runtime
    FunctionType(fn.__code__, dict(parent.engine.__dict__, identity=identity),
                 argdefs=fn.__defaults__)(spec, config, phase, state, code)


def submit(spec, config):
    root = Path(spec['output']).resolve()
    assert not root.exists(), 'Do not resubmit an existing experiment directory'
    # Reduce the original array's future concurrency without interrupting jobs.
    old = spec['parent_refine_job']
    subprocess.run(['scontrol', 'update', f'JobId={old}', 'ArrayTaskThrottle=12'], check=True)
    active = subprocess.check_output(['squeue', '-h', '-r', '-j', old, '-t', 'RUNNING,COMPLETING,CONFIGURING', '-o', '%i'], text=True)
    running = len(active.splitlines())
    extra_limit = min(12, 24-max(12, running))
    assert extra_limit >= 1, 'All 24 GPUs still active: retry submission after one finishes'
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    write(root/'config.frozen.json', spec)
    phases = [
        ('prepare', None, False, '01:00:00'),
        ('refine', f'0-23%{extra_limit}', True, '1-12:00:00'),
        ('select', None, False, '00:30:00'),
        ('final', '0-5%6', True, '1-12:00:00'),
        ('merge', None, False, '00:30:00'),
    ]
    dependency = None
    for phase, array, gpu, limit in phases:
        command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
                   '--mem=32G', '--time='+limit, '--job-name=t4c-fpsaug20-'+phase,
                   f'--output={root}/logs/{phase}_%A_%a.out', f'--error={root}/logs/{phase}_%A_%a.err']
        if array:
            command += ['--array='+array]
        if gpu:
            command += ['--gres=gpu:1', '--constraint=v100']
        if dependency:
            dependencies = dependency + (':'+old if phase == 'select' else '')
            command += ['--dependency=afterok:'+dependencies, '--kill-on-invalid-dep=yes']
        command += ['ibex_bash/task4c_fps_augment_search_1p2.sh', phase, config]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        row = dict(job_id=job, version=spec['version'], phase=phase,
                   submitted_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                   identity=identity(config), expected_device='V100' if gpu else 'CPU',
                   parent_refine_throttle=12, extra_refine_throttle=extra_limit)
        append_locked(root/'submissions.jsonl', json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        dependency = job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'refine', 'select', 'final', 'merge', 'submit', 'runtime'))
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--runtime-phase')
    parser.add_argument('--state')
    parser.add_argument('--exit-code', type=int)
    args = parser.parse_args()
    spec = load_spec(args.config)
    if args.phase in ('refine', 'final'):
        train(spec, args.config, args.index, args.phase)
    elif args.phase == 'runtime':
        runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else:
        globals()[args.phase](spec, args.config)


if __name__ == '__main__':
    main()
