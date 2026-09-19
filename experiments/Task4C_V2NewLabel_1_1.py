"""Build and audit v2_newlabel_rule, with shortest-curve majority labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import FunctionType
import numpy as np
from scipy.spatial import cKDTree

from FMT_Utils import Task4C_V2NewLabel_1_1 as data
from experiments import Task4C_FixedDataset_2_1 as original

CONFIG = 'config/mainExp_Task4C_V2NewLabel_1.1.json'
FILES = tuple(dict.fromkeys(original.FILES + (
    original.CONFIG,
    'FMT_Utils/Task4C_FixedDataset_3_1.py', 'FMT_Utils/Task4C_FixedDataset_3_2.py',
    'FMT_Utils/Task4C_V2NewLabel_1_1.py', 'experiments/Task4C_V2NewLabel_1_1.py',
    'tests/test_task4c_v2_newlabel_1_1.py', 'ibex_bash/task4c_v2_newlabel_1p1.sh',
    'docs/Task4C_v2_newlabel_protocol_1.1.md', CONFIG)))


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    temporary.replace(path)


def identity(config):
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=sha(config), sources={name: sha(name) for name in FILES})


def validate_spec(spec):
    old = json.loads(Path(original.CONFIG).read_text())
    for key in ('physical_config', 'flows', 'integration', 'sampling', 'pilot', 'folds', 'audit', 'input_roots'):
        assert spec[key] == old[key], key
    assert spec['dataset_name'] == 'v2_newlabel_rule' and spec['output'] == 'outputs/v2_newlabel_rule'
    assert spec['labeling'] == dict(curve_index=0, points=32, positive_minimum_hits=17,
        ties='negative', membership='union_of_exact_GT_cells_including_instance_zero',
        shared_across_three_lengths=True, seed_label_used_for_training=False)
    assert spec['balance']['positive_to_negative_ratio'] == .5


def check(path, config):
    result = json.loads(Path(path).read_text())
    assert result['complete'] and result['identity'] == identity(config)
    return result


def verify_inputs(spec, config):
    validate_spec(spec)
    reference = Path(spec['reference_dataset'])
    assert sha(reference/'data_audit.json') == spec['reference_audit_sha256']
    for flow in spec['flows']:
        prep = json.loads((reference/'physical'/flow['name']/'preparation.json').read_text())
        for key in ('fold_of_instance', 'groups'):
            assert prep['folds'][key] == spec['reference_folds'][flow['name']][key]
    FunctionType(original.verify_inputs.__code__, dict(original.__dict__, identity=identity, write=write))(spec, config)


def audit_flow(spec, config, index, pilot=False):
    api = data.v2; flow = spec['flows'][index]; name = flow['name']
    folder = Path(spec['output'])/('pilot' if pilot else 'physical')/name
    prep = check(folder/'preparation.json', config)
    assert prep['pilot'] == pilot
    for file, digest in prep['files'].items(): assert sha(folder/file) == digest, file
    seeds, curves, meta = api.load(folder); n = len(seeds)
    sampling = dict(spec['sampling'])
    if pilot: sampling.update(spec['pilot'])
    assert set(meta) == set(data.METADATA_KEYS) and n == prep['samples'] <= sampling['samples_per_flow']
    scene = api.load_scene(spec, index); steps = scene['steps']; integration = spec['integration']
    assert curves.dtype == np.float32 and curves.shape == (n, 3, 32, 3) and np.isfinite(curves).all()
    assert list(meta['half_steps']) == steps and float(meta['ds']) == flow['ds']
    assert float(meta['lambda2_threshold']) == scene['threshold']
    np.testing.assert_array_equal(meta['requested_half_length'], flow['half_lengths'])
    lam, inside, _ = api.interpolate_scalar(seeds, scene['axes'], scene['lambda2'])
    oyf, _, _ = api.interpolate_scalar(seeds, scene['axes'], scene['oyf'])
    assert inside.all() and np.all((lam < scene['threshold']) & (oyf > 0))
    np.testing.assert_allclose(lam, meta['lambda2']); np.testing.assert_allclose(oyf, meta['oyf'])
    mask = api.candidate_mask(scene['lambda2'], scene['oyf'], scene['threshold'])
    components, sizes, _ = api.candidate_components(mask)
    pool, _ = api.initial_pool(scene['axes'], scene['lambda2'], scene['oyf'], scene['threshold'],
        api.candidate_cells(mask), sampling['pool_per_flow'], sampling['pool_seed']+index, sampling['pool_chunk'])
    proposal = np.load(folder/'proposal_labels.npy'); proposal_valid = np.load(folder/'proposal_short_valid.npy')
    proposal_hits = np.load(folder/'proposal_gt_count.npy'); selected_ids = np.load(folder/'selected_pool_indices.npy')
    assert len(proposal) == len(pool) and np.all(np.isin(proposal, [0, 1]))
    np.testing.assert_array_equal(proposal[proposal_valid], proposal_hits[proposal_valid] > 16)
    assert np.all(proposal_hits[~proposal_valid] == -1) and np.all(proposal[~proposal_valid] == 0)
    radius = float(meta['poisson_radius']); order = np.random.default_rng([sampling['poisson_seed'], index]).permutation(len(pool))
    limits = np.array(prep['balance']['selected_by_class'])
    reproduced = data.quota.dart_throw(pool[order], proposal[order], radius, limits)
    np.testing.assert_array_equal(order[reproduced], selected_ids)
    assert len(reproduced) == sampling['samples_per_flow']
    assert len(data.quota.dart_throw(pool[order], proposal[order], radius*(1+10*sampling['radius_relative_tolerance']), limits)) < len(selected_ids)
    np.testing.assert_array_equal(pool[meta['pool_index']], seeds)
    assert np.isin(meta['pool_index'], selected_ids).all()
    np.testing.assert_array_equal(np.bincount(proposal[selected_ids], minlength=2), limits)
    distance = cKDTree(pool[selected_ids]).query(pool[selected_ids], k=2)[0][:, 1]
    assert np.all(distance >= radius*(1-1e-12)), 'Poisson distance includes opposite-label pairs'
    arc_ratio = meta['half_arc_lengths']/np.asarray(flow['half_lengths'])[None, :, None]
    out = (meta['half_termination'][:, None, :] == api.OUT_OF_DOMAIN) & (meta['half_step_counts'] < np.asarray(steps)[None, :, None])
    low = np.where(out, integration['out_of_domain_min_half_arc_fraction'], integration['minimum_actual_half_arc_fraction'])
    high = np.where(out, integration['out_of_domain_max_half_arc_fraction'], integration['maximum_actual_half_arc_fraction'])
    assert np.all(arc_ratio >= low) and np.all(arc_ratio <= high)
    np.testing.assert_array_equal(meta['boundary_relaxed'], out.any(2))
    assert np.all(meta['half_step_counts'] <= np.asarray(steps)[None, :, None])
    assert np.all(np.ptp(curves, axis=2) > 0)
    assert np.all(curves.min(2) >= meta['curve_bounds'][:, :, 0]-1e-5)
    assert np.all(curves.max(2) <= meta['curve_bounds'][:, :, 1]+1e-5)
    # Every published label is recomputed from all 32 saved shortest-curve points.
    labels = data.shortest_curve_labels(scene['gt'], curves, scene['locator'])
    for key, value in labels.items(): np.testing.assert_array_equal(value, meta[key])
    owner, _ = api.sample_gt(scene['gt'], seeds, scene['locator'])
    np.testing.assert_array_equal(owner, meta['gt_owner'])
    np.testing.assert_array_equal(owner >= 0, meta['seed_label'])
    comp_owner, _, candidate_points = api.component_instances(mask, components, scene['axes'], scene['gt'], scene['locator'])
    k, j, i = np.nonzero(mask); component = components[k, j, i][cKDTree(candidate_points).query(seeds)[1]]
    np.testing.assert_array_equal(component, meta['component'])
    np.testing.assert_array_equal(sizes[component], meta['component_size'])
    instance, kind = api.assign_instances(seeds, owner, scene['instances'], scene['box_low'], scene['box_high'],
        scene['gt_points'], scene['gt_point_instance'], component, comp_owner)
    np.testing.assert_array_equal(instance, meta['instance']); np.testing.assert_array_equal(kind, meta['assignment_kind'])
    mapping = spec['reference_folds'][name]['fold_of_instance']
    assert prep['folds']['fold_of_instance'] == mapping
    np.testing.assert_array_equal(meta['fold'], [mapping[str(v)] for v in instance])
    np.testing.assert_array_equal(meta['split'], np.where(meta['fold'] == 0, api.SPLIT_TEST, api.SPLIT_TRAIN))
    np.testing.assert_allclose(meta['local_grid_scale'], api.local_grid_scale(seeds, scene['axes']))
    rng = np.random.default_rng([spec['audit']['seed'], index])
    subset = np.sort(rng.choice(n, min(spec['audit']['reintegrate_samples'], n), replace=False))
    retraced = api.trace_curves(scene['grid'], seeds[subset], flow['ds'], steps, integration)
    assert retraced['valid'].all()
    np.testing.assert_allclose(retraced['curves'], curves[subset], atol=1e-6, rtol=1e-6)
    for key, stored in [('arcs', 'half_arc_lengths'), ('counts', 'half_step_counts'), ('termination', 'half_termination'), ('relaxed', 'boundary_relaxed')]:
        np.testing.assert_allclose(retraced[key], meta[stored][subset])
    proposal_subset = np.sort(rng.choice(len(pool), min(500, len(pool)), replace=False))
    probe = api.trace_curves(scene['grid'], pool[proposal_subset], flow['ds'], [steps[0]], integration)
    valid = probe['valid'][:, 0]
    np.testing.assert_array_equal(valid, proposal_valid[proposal_subset])
    probe_labels = data.shortest_curve_labels(scene['gt'], np.repeat(probe['curves'][valid], 3, axis=1), scene['locator'])
    np.testing.assert_array_equal(probe_labels['short_curve_gt_count'], proposal_hits[proposal_subset[valid]])
    kept = np.bincount(meta['label'], minlength=2)
    np.testing.assert_array_equal(kept, prep['balance']['kept_by_class'])
    return dict(complete=True, identity=identity(config), samples=n, files=prep['files'],
        selected_by_class=limits.tolist(), kept_by_class=kept.tolist(), balance=prep['balance'],
        per_fold={str(f): np.bincount(meta['label'][meta['fold'] == f], minlength=2).tolist() for f in range(5)},
        shortest_curve_point_labels_recomputed=n*32, step1_all_pass=True, global_poisson_all_pairs=True,
        original_instance_assignment=True, exact_v2_fold_map=True, validation_samples=0,
        reintegrated=len(subset), proposal_reintegrated=len(proposal_subset), label_changes=prep['label_changes'])


def calibrate(spec, config):
    flows = {}
    for flow in spec['flows']:
        name = flow['name']; path = Path(spec['output'])/'pilot'/name/'pilot_check.json'
        report = check(path, config)
        fraction, survival = data.balanced_fraction(report['selected_by_class'], report['kept_by_class'])
        flows[name] = dict(positive_fraction=fraction, survival_by_class=survival, pilot_sha256=sha(path))
    output = Path(spec['output'])/'calibration.json'; assert not output.exists()
    write(output, dict(complete=True, identity=identity(config), target_ratio=.5, flows=flows))


def audit_data(spec, config):
    reports = {flow['name']: audit_flow(spec, config, i) for i, flow in enumerate(spec['flows'])}
    write(Path(spec['output'])/'data_audit.json', dict(complete=True, identity=identity(config),
        dataset_name=spec['dataset_name'], version=spec['version'], samples=sum(r['samples'] for r in reports.values()),
        flows=reports, frozen_files={name: r['files'] for name, r in reports.items()},
        labeling=spec['labeling'], target_positive_to_negative_ratio=.5,
        balance_target_met=all(r['balance']['within_relative_tolerance'] for r in reports.values()),
        test_fold=0, validation_samples=0, rules=spec['rules']))


def runtime(spec, config, phase, state, code):
    path = Path(spec['output'])/'runtime.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(dict(identity=identity(config), phase=phase, state=state, exit_code=code,
            job=os.environ.get('SLURM_JOB_ID'), index=os.environ.get('SLURM_ARRAY_TASK_ID'),
            at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit(spec, config):
    root = Path(spec['output']); (root/'logs').mkdir(parents=True, exist_ok=True)
    assert not (root/'submission.json').exists(), 'Already submitted'
    jobs = {}
    phases = [('verify-inputs', [], None, '00:30:00', '16G'), ('pilot', ['verify-inputs'], '0-1%2', '04:00:00', '48G'),
              ('calibrate', ['pilot'], None, '00:10:00', '4G'), ('build', ['calibrate'], '0-1%2', '12:00:00', '64G'),
              ('audit-data', ['build'], None, '04:00:00', '64G')]
    for phase, deps, array, wall, memory in phases:
        cmd = ['sbatch', '--parsable', '--propagate=NONE', '--partition=batch', '--job-name=V2NewLabel_'+phase,
               '--cpus-per-task=4', '--mem='+memory, '--time='+wall, '--kill-on-invalid-dep=yes',
               '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: cmd += ['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        if array: cmd += ['--array='+array]
        cmd += ['ibex_bash/task4c_v2_newlabel_1p1.sh', phase, config]
        jobs[phase] = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        write(root/'submission.json', dict(identity=identity(config), jobs=jobs, gpu_jobs=0, formal_fits=0))
        with (root/'submissions.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(phase=phase, job=jobs[phase], command=cmd, at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        print(phase, jobs[phase], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['verify-inputs', 'pilot', 'calibrate', 'build', 'audit-data', 'submit', 'runtime'])
    parser.add_argument('--config', default=CONFIG); parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--runtime-phase'); parser.add_argument('--state'); parser.add_argument('--exit-code', type=int)
    args = parser.parse_args(); spec = json.loads(Path(args.config).read_text()); validate_spec(spec)
    if args.phase in ('pilot', 'build'):
        if args.phase == 'build': check(Path(spec['output'])/'calibration.json', args.config)
        data.prepare(spec, args.index, write, sha, lambda: identity(args.config), args.phase == 'pilot')
        if args.phase == 'pilot': write(Path(spec['output'])/'pilot'/spec['flows'][args.index]['name']/'pilot_check.json', audit_flow(spec, args.config, args.index, True))
    elif args.phase == 'runtime': runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else: globals()[args.phase.replace('-', '_')](spec, args.config)


if __name__ == '__main__': main()
