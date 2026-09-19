"""Build Task4-c v3, 40M initial points -> 8M seeds; no training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from types import FunctionType
import numpy as np
from experiments import Task4C_FixedDataset_2_1 as previous
from FMT_Utils import Task4C_FixedDataset_3_1 as dense

CONFIG = 'config/mainExp_Task4C_FixedDataset_3.1.json'
FILES = tuple(dict.fromkeys(previous.FILES + (
    'FMT_Utils/Task4C_FixedDataset_3_1.py', 'experiments/Task4C_FixedDataset_3_1.py',
    'tests/test_task4c_fixed_dataset_3_1.py', 'ibex_bash/task4c_fixed_dataset_3p1.sh',
    'docs/Task4C_fixed_dataset_protocol_3.1.md', 'config/mainExp_Task4C_FixedDataset_2.1.json', CONFIG)))
sha, write = previous.sha, previous.write


def identity(config):
    # Host belongs to execution records, not scientific identity shared between Slurm nodes.
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=sha(config), sources={p: sha(p) for p in FILES})


def validate_spec(spec):
    old = json.loads(Path('config/mainExp_Task4C_FixedDataset_2.1.json').read_text())
    for key in ('flows', 'integration', 'folds', 'physical_config', 'audit', 'input_roots'):
        assert spec[key] == old[key], key
    for key in old['sampling']:
        if key not in ('pool_per_flow', 'samples_per_flow'):
            assert spec['sampling'][key] == old['sampling'][key], key
    assert spec['sampling']['pool_per_flow'] == 20_000_000
    assert spec['sampling']['samples_per_flow'] == 4_000_000
    assert spec['pilot'] == dict(pool_per_flow=200_000, samples_per_flow=40_000)
    assert spec['output'] == 'outputs/mainExp_Task4C_FixedDataset_3.1'
    assert spec['reference_audit_sha256'] == 'e5aaa4fb8a8e01882c461da5916b3bb96883e2787b231b018b512d7e30d3a2a8'


def invoke(name, spec, config):
    fn = getattr(previous, name)
    return FunctionType(fn.__code__, dict(previous.__dict__, data=dense.adapter(), identity=identity),
                        argdefs=fn.__defaults__)(spec, config)


def verify_inputs(spec, config):
    validate_spec(spec)
    reference = Path(spec['reference_dataset'])
    assert sha(reference/'data_audit.json') == spec['reference_audit_sha256']
    for flow in spec['flows']:
        name = flow['name']
        prep = json.loads((reference/'physical'/name/'preparation.json').read_text())
        for key in ('fold_of_instance', 'groups'):
            assert spec['reference_folds'][name][key] == prep['folds'][key]
    root = Path(spec['output']); root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    assert disk.free >= 40*1024**3, 'Need at least 40 GiB free for dataset and working space'
    invoke('verify_inputs', spec, config)
    write(root/'resource_check.json', dict(complete=True, free_bytes=disk.free,
        maximum_curve_bytes=8_000_000*3*32*3*4, formal_fits=0, gpu_jobs=0,
        target_per_flow=4_000_000, initial_pool_per_flow=20_000_000))


def audit_data(spec, config):
    root = Path(spec['output'])
    for flow in spec['flows']:
        name = flow['name']
        prep = json.loads((root/'physical'/name/'preparation.json').read_text())
        assert prep['folds']['fold_of_instance'] == spec['reference_folds'][name]['fold_of_instance']
        assert prep['pool']['requested'] == 20_000_000 and prep['poisson']['target'] == 4_000_000
    invoke('audit_data', spec, config)
    audit = json.loads((root/'data_audit.json').read_text())
    audit.update(rules=spec['rules'], reference_audit_sha256=spec['reference_audit_sha256'],
                 exact_v2_instance_fold_map=True)
    write(root/'data_audit.json', audit)


def pilot_check(spec, config, index):
    """Gate full builds on actual-field dense/sparse equality and reintegration."""
    from scipy.spatial import cKDTree
    data = dense.v2
    name = spec['flows'][index]['name']
    folder = Path(spec['output'])/'pilot'/name
    prep = json.loads((folder/'preparation.json').read_text())
    seeds, curves, meta = data.load(folder)
    assert prep['complete'] and prep['pilot'] and len(seeds) > 0
    assert prep['identity'] == identity(config)
    assert prep['folds']['fold_of_instance'] == spec['reference_folds'][name]['fold_of_instance']
    for filename, digest in prep['files'].items():
        assert sha(folder/filename) == digest
    scene = data.load_scene(spec, index)
    sampling = dict(spec['sampling'], **spec['pilot'])
    cells = data.candidate_cells(data.candidate_mask(scene['lambda2'], scene['oyf'], scene['threshold']))
    pool, _ = data.initial_pool(scene['axes'], scene['lambda2'], scene['oyf'], scene['threshold'], cells,
        sampling['pool_per_flow'], sampling['pool_seed']+index, sampling['pool_chunk'])
    order = np.random.default_rng([sampling['poisson_seed'], index]).permutation(len(pool))
    selected, radius, log = data.poisson_disk(pool[order], sampling['samples_per_flow'],
        prep['poisson']['radius_guess'], sampling['radius_relative_tolerance'], sampling['grid_capacity_per_cell'])
    assert radius == prep['poisson']['radius'] and log == prep['poisson']['bisection']
    assert np.array_equal(pool[meta['pool_index']], seeds)
    assert np.isin(meta['pool_index'], order[selected]).all()
    assert (cKDTree(seeds).query(seeds, k=2)[0][:, 1] >= radius*(1-1e-12)).all()
    owner, _ = data.sample_gt(scene['gt'], seeds, scene['locator'])
    assert np.array_equal(owner, meta['gt_owner']) and np.array_equal(owner >= 0, meta['label'])
    sub = np.sort(np.random.default_rng([spec['audit']['seed'], index]).choice(len(seeds), min(500, len(seeds)), replace=False))
    re = data.trace_curves(scene['grid'], seeds[sub], scene['flow_spec']['ds'], scene['steps'], spec['integration'])
    assert re['valid'].all() and np.allclose(re['curves'], curves[sub], atol=1e-6, rtol=1e-6)
    assert np.allclose(re['arcs'], meta['half_arc_lengths'][sub])
    assert np.array_equal(re['termination'], meta['half_termination'][sub])
    write(folder/'pilot_check.json', dict(complete=True, identity=identity(config), samples=len(seeds),
        target=sampling['samples_per_flow'], dense_sparse_radius_and_selection_pass=True,
        exact_v2_instance_fold_map=True, reintegrated=len(sub)))
    print(json.dumps(dict(flow=name, pilot_check='PASS', samples=len(seeds))), flush=True)


def runtime(spec, config, phase, state, code):
    root = Path(spec['output']); root.mkdir(parents=True, exist_ok=True)
    previous.frozen.append_locked(root/'runtime.jsonl', json.dumps(dict(identity=identity(config),
        phase=phase, state=state, exit_code=code, job_id=os.environ.get('SLURM_JOB_ID'),
        array_index=os.environ.get('SLURM_ARRAY_TASK_ID'), host=socket.gethostname(),
        at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit(spec, config):
    validate_spec(spec)
    root = Path(spec['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(), 'Already submitted; do not duplicate'
    jobs = {}
    phases = [('verify-inputs', [], None, '00:30:00', '16G'),
              ('pilot', ['verify-inputs'], '0-1%2', '02:00:00', '32G'),
              ('build', ['pilot'], '0-1%2', '24:00:00', '96G'),
              ('audit-data', ['build'], None, '12:00:00', '96G')]
    for phase, deps, array, wall, memory in phases:
        command = ['sbatch', '--parsable', '--propagate=NONE', '--partition=batch',
            '--job-name=FixedDS31_'+phase, '--cpus-per-task=4', '--mem='+memory, '--time='+wall,
            '--kill-on-invalid-dep=yes', '--output='+str(root/'logs/%x_%A_%a.out'),
            '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps:
            command += ['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        if array:
            command += ['--array='+array]
        command += ['ibex_bash/task4c_fixed_dataset_3p1.sh', phase, config]
        jobs[phase] = subprocess.check_output(command, text=True).strip().split(';')[0]
        previous.frozen.append_locked(root/'submissions.jsonl', json.dumps(dict(phase=phase,
            job=jobs[phase], command=command, at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        write(root/'submission.json', dict(identity=identity(config), jobs=jobs))
    print(json.dumps(jobs), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['verify-inputs', 'pilot', 'build', 'audit-data', 'submit', 'runtime'])
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--runtime-phase'); parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = json.loads(Path(args.config).read_text())
    validate_spec(spec)
    if args.phase in ('pilot', 'build'):
        if args.phase == 'build':
            for flow in spec['flows']:
                check = json.loads((Path(spec['output'])/'pilot'/flow['name']/'pilot_check.json').read_text())
                assert check['complete'] and check['identity'] == identity(args.config)
        dense.prepare(spec, args.index, write, sha, lambda: identity(args.config), pilot=args.phase == 'pilot')
        if args.phase == 'pilot':
            pilot_check(spec, args.config, args.index)
    elif args.phase == 'runtime':
        runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else:
        globals()[args.phase.replace('-', '_')](spec, args.config)


if __name__ == '__main__':
    main()
