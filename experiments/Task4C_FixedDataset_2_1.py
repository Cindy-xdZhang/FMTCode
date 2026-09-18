"""Build and audit the Task4-c fixed dataset v2 (mainExp_Task4C_FixedDataset_2.1).

Phases: verify-inputs -> pilot (both flows, small pool) -> build (both flows) -> audit-data.
The rules are docs/Task4C_fixed_dataset_rules_v2.md.  Every phase runs on the local host or
on Ibex (``submit`` chains the Slurm jobs); ``identity`` records the commit and source hashes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_FixedDataset_2_1 as data
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
from experiments import Task4C_FPSAugmentSearch_1_1 as frozen

sha, write = frozen.sha, frozen.write
CONFIG = 'config/mainExp_Task4C_FixedDataset_2.1.json'
FILES = ('FMT_Utils/Task4C_FixedDataset_2_1.py', 'experiments/Task4C_FixedDataset_2_1.py', 'FMT_Utils/Task4C_GTHeadCoverage_1_1.py',
         'FMT_Utils/Task4C_Multiscale_4_1.py', 'FMT_Utils/Task4C_HairpinBinary_2_1.py', 'FMT_Utils/Task4C_PaperBundles_3_1.py',
         'FMT_Utils/Task4C_Bundles_1_1.py', 'config/mainExp_Task4C_GTHeadCoverage_1.1.json', 'docs/Task4C_fixed_dataset_rules_v2.md',
         'ibex_bash/task4c_fixed_dataset_2p1.sh')


def identity(config):
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), config_sha256=sha(config),
                sources={p: sha(p) for p in FILES}, host=socket.gethostname())


def verify_inputs(spec, config):
    physical = json.loads(Path(spec['physical_config']).read_text()); root = Path(data.resolve_input_root(spec)); checked = {}
    for fi, flow in enumerate(physical['flows']):
        assert flow['name'] == spec['flows'][fi]['name']
        for key in ('flow', 'gt'):
            digest = sha(root/flow[key]); assert digest == flow[key+'_sha256'], (flow['name'], key); checked[str(root/flow[key])] = digest
        data.half_steps(spec['flows'][fi]['half_lengths'], spec['flows'][fi]['ds'])
    write(Path(spec['output'])/'input_verification.json', dict(complete=True, identity=identity(config), input_root=str(root), files=checked))


def audit_data(spec, config):
    """Recompute every rule of docs/Task4C_fixed_dataset_rules_v2.md from the written files and the frozen fields."""
    root = Path(spec['output']); sampling = spec['sampling']; integration = spec['integration']; report = {}; frozen_files = {}; total = 0
    for fi, flow in enumerate(spec['flows']):
        name = flow['name']; folder = root/'physical'/name; prep = json.loads((folder/'preparation.json').read_text())
        assert prep['complete'] and not prep['pilot'] and prep['identity'] == identity(config), 'preparation identity differs from the audited code'
        for filename, digest in prep['files'].items(): assert sha(folder/filename) == digest, (name, filename)
        frozen_files[name] = prep['files']
        seeds, curves, meta = data.load(folder); n = len(seeds); total += n; target = sampling['samples_per_flow']
        assert set(meta) == set(data.METADATA_KEYS) and prep['samples'] == n <= target
        scene = data.load_scene(spec, fi); axes = scene['axes']; thr = scene['threshold']; steps = scene['steps']; ds = flow['ds']
        assert list(meta['half_steps']) == steps and float(meta['ds']) == ds and float(meta['lambda2_threshold']) == thr
        assert np.allclose(meta['requested_half_length'], flow['half_lengths'])
        K = len(steps); P = integration['points_per_curve']; assert curves.shape == (n, K, P, 3)
        # step1 at every seed.
        l, inside, _ = interpolate_scalar(seeds, axes, scene['lambda2']); o, _, _ = interpolate_scalar(seeds, axes, scene['oyf'])
        assert inside.all() and np.all((l < thr) & (o > 0)) and np.allclose(l, meta['lambda2']) and np.allclose(o, meta['oyf'])
        # Poisson-disk: pairwise distance >= radius, and the selection reproduces from the recorded seeds.
        radius = float(meta['poisson_radius']); d, _ = cKDTree(seeds).query(seeds, k=2)
        assert np.all(d[:, 1] >= radius*(1-1e-12)), 'two samples closer than the Poisson radius'
        mask = data.candidate_mask(scene['lambda2'], scene['oyf'], thr); cells = data.candidate_cells(mask)
        pool, _ = data.initial_pool(axes, scene['lambda2'], scene['oyf'], thr, cells, sampling['pool_per_flow'], sampling['pool_seed']+fi, sampling['pool_chunk'])
        assert np.array_equal(pool[meta['pool_index']], seeds), 'seeds are not the recorded pool points'
        order = np.random.default_rng([sampling['poisson_seed'], fi]).permutation(len(pool))
        accepted = data.dart_throw(pool[order], radius, target, sampling['grid_capacity_per_cell'])
        assert len(accepted) >= target and set(meta['pool_index'].tolist()) <= set(order[accepted[:target]].tolist()), 'Poisson selection does not reproduce'
        assert len(data.dart_throw(pool[order], radius*(1+10*sampling['radius_relative_tolerance']), target, sampling['grid_capacity_per_cell'])) < target, 'radius is not maximal'
        # Curves: lengths, steps, geometry, and a re-integrated subset.
        L = np.asarray(flow['half_lengths']); ratio = meta['half_arc_lengths']/L[None, :, None]
        out = (meta['half_termination'][:, None, :] == data.OUT_OF_DOMAIN) & (meta['half_step_counts'] < np.asarray(steps)[None, :, None])
        lo = np.where(out, integration['out_of_domain_min_half_arc_fraction'], integration['minimum_actual_half_arc_fraction'])
        hi = np.where(out, integration['out_of_domain_max_half_arc_fraction'], integration['maximum_actual_half_arc_fraction'])
        assert np.all(ratio >= lo) and np.all(ratio <= hi), 'a half line violates its arc-length band'
        assert np.array_equal(meta['boundary_relaxed'], out.any(2)), 'relaxed flags differ from the recorded terminations'
        assert np.all(meta['half_step_counts'] <= np.asarray(steps)[None, :, None]) and np.isfinite(curves).all()
        for first in range(0, n, 4096):
            block = np.asarray(curves[first:first+4096], np.float64); assert np.all(np.ptp(block, axis=2) > 0)
            assert np.all(block.min(2) >= meta['curve_bounds'][first:first+4096, :, 0]-1e-5) and np.all(block.max(2) <= meta['curve_bounds'][first:first+4096, :, 1]+1e-5)
        rng = np.random.default_rng([spec['audit']['seed'], fi]); sub = np.sort(rng.choice(n, min(spec['audit']['reintegrate_samples'], n), replace=False))
        re = data.trace_curves(scene['grid'], seeds[sub], ds, steps, integration)
        assert re['valid'].all() and np.allclose(re['curves'], curves[sub], atol=1e-6, rtol=1e-6) and np.allclose(re['arcs'], meta['half_arc_lengths'][sub])
        assert np.array_equal(re['counts'], meta['half_step_counts'][sub]) and np.array_equal(re['termination'], meta['half_termination'][sub]) and np.array_equal(re['relaxed'], meta['boundary_relaxed'][sub])
        # Labels and assignment.
        owner, _ = data.sample_gt(scene['gt'], seeds, scene['locator'])
        assert np.array_equal(owner, meta['gt_owner']) and np.array_equal(meta['label'], (owner >= 0).astype(meta['label'].dtype))
        labels, sizes, _ = data.candidate_components(mask)
        comp_instance, _, candidate_points = data.component_instances(mask, labels, axes, scene['gt'], scene['locator'])
        k, j, i = np.nonzero(mask); seed_component = labels[k, j, i][cKDTree(candidate_points).query(seeds)[1]]
        assert np.array_equal(seed_component, meta['component']) and np.array_equal(sizes[seed_component], meta['component_size'])
        instance, kind = data.assign_instances(seeds, owner, scene['instances'], scene['box_low'], scene['box_high'], scene['gt_points'], scene['gt_point_instance'], seed_component, comp_instance)
        assert np.array_equal(instance, meta['instance']) and np.array_equal(kind, meta['assignment_kind'])
        assert np.all(meta['instance'][meta['label'] == 1] == owner[meta['label'] == 1])
        # Folds: one fold per instance, recorded assignment, split from the test fold.
        fold_of = {int(a): b for a, b in prep['folds']['fold_of_instance'].items()}
        assert set(fold_of) == set(scene['instances']) and all(0 <= f < spec['folds']['count'] for f in fold_of.values())
        assert np.array_equal(meta['fold'], np.array([fold_of[int(v)] for v in meta['instance']]))
        assert np.array_equal(meta['split'], np.where(meta['fold'] == spec['folds']['test_fold'], data.SPLIT_TEST, data.SPLIT_TRAIN))
        assert [int(np.sum(meta['fold'] == f)) for f in range(spec['folds']['count'])] == prep['folds']['fold_samples']
        assert np.allclose(meta['local_grid_scale'], data.local_grid_scale(seeds, axes))
        report[name] = dict(samples=n, target=target, radius=radius, classes=prep['classes'], splits=prep['splits'], fold_samples=prep['folds']['fold_samples'],
                            assignment_kinds=prep['assignment']['per_kind'], cleaning=prep['cleaning'], reintegrated=int(len(sub)),
                            relaxed_curves_per_length=meta['boundary_relaxed'].sum(0).tolist(), samples_with_out_of_domain_half=int(np.any(meta['half_termination'] == data.OUT_OF_DOMAIN, axis=1).sum()))
        del scene
    write(root/'data_audit.json', dict(complete=True, identity=identity(config), samples=total, flows=report, frozen_files=frozen_files,
                                       rules='docs/Task4C_fixed_dataset_rules_v2.md recomputed', test_fold=spec['folds']['test_fold']))
    print(json.dumps(dict(status='PASS', samples=total, flows={k: v['samples'] for k, v in report.items()})), flush=True)


def runtime(spec, config, phase, state, code):
    frozen.append_locked(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(config), phase=phase, state=state, exit_code=code,
        job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        hostname=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit(spec, config):
    root = Path(spec['output']); (root/'logs').mkdir(parents=True, exist_ok=True); assert not (root/'submission.json').exists(); jobs = {}
    phases = [('verify-inputs', [], None, '00:20:00', '16G'), ('pilot', ['verify-inputs'], '0-1', '02:00:00', '64G'),
              ('build', ['pilot'], '0-1', '12:00:00', '96G'), ('audit-data', ['build'], None, '06:00:00', '96G')]
    for phase, deps, array, wall, memory in phases:
        command = ['sbatch', '--parsable', '--propagate=NONE', '--job-name=FixedDS21_'+phase, '--cpus-per-task=4', '--mem='+memory, '--time='+wall,
                   '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: command += ['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        if array: command += ['--array='+array]
        command += ['ibex_bash/task4c_fixed_dataset_2p1.sh', phase, config]
        jobs[phase] = subprocess.check_output(command, text=True).strip().split(';')[0]
        frozen.append_locked(root/'submissions.jsonl', json.dumps(dict(phase=phase, job=jobs[phase], command=command, submitted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        write(root/'submission.json', dict(identity=identity(config), jobs=jobs))
    print(json.dumps(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['verify-inputs', 'pilot', 'build', 'audit-data', 'submit', 'runtime'])
    p.add_argument('--config', default=CONFIG); p.add_argument('--index', type=int, default=0)
    p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args(); spec = json.loads(Path(a.config).read_text())
    if a.phase == 'verify-inputs': verify_inputs(spec, a.config)
    elif a.phase in ('pilot', 'build'): data.prepare(spec, a.index, write, sha, lambda: identity(a.config), pilot=a.phase == 'pilot')
    elif a.phase == 'audit-data': audit_data(spec, a.config)
    elif a.phase == 'submit': submit(spec, a.config)
    else: runtime(spec, a.config, a.runtime_phase, a.state, a.exit_code)


if __name__ == '__main__': main()
