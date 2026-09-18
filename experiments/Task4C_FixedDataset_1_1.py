"""Build and audit the Task4-c fixed bundle dataset v1 (mainExp_Task4C_FixedDataset_1.1).

Phases: verify-inputs -> pilot (both flows) -> build (both flows) -> audit-data.  Training
experiments consume the audited files read-only through their own versions.
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
from FMT_Utils import Task4C_FixedDataset_1_1 as data
from FMT_Utils import Task4C_FPS16_Data_1_2 as rules
from experiments import Task4C_FPSAugmentSearch_1_1 as frozen

sha, write = frozen.sha, frozen.write
CONFIG = 'config/mainExp_Task4C_FixedDataset_1.1.json'
FILES = ('FMT_Utils/Task4C_FixedDataset_1_1.py', 'experiments/Task4C_FixedDataset_1_1.py', 'FMT_Utils/Task4C_FPS16_Data_1_2.py',
         'FMT_Utils/Task4C_GTHeadCoverage_1_1.py', 'FMT_Utils/Task4C_InstanceCoverage_9_15.py', 'FMT_Utils/Task4C_Multiscale_4_1.py',
         'FMT_Utils/Task4C_HairpinBinary_2_1.py', 'FMT_Utils/Task4C_PaperBundles_3_1.py', 'FMT_Utils/Task4C_Bundles_1_1.py',
         'FMT_Utils/Task4C_OriginalCenter_1_1.py', 'experiments/Task4C_PhysicalLength_4_14.py', 'config/mainExp_Task4C_GTHeadCoverage_1.1.json',
         'ibex_bash/task4c_fixed_dataset_1p1.sh')


def identity(config):
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                config_sha256=sha(config), sources={p: sha(p) for p in FILES})


def verify_inputs(spec, config):
    physical = json.loads(Path(spec['physical_config']).read_text()); root = Path(physical['input_root']); checked = {}
    for flow in physical['flows']:
        for key in ('flow', 'gt'):
            digest = sha(root/flow[key]); assert digest == flow[key+'_sha256'], (flow['name'], key); checked[str(root/flow[key])] = digest
    counts = spec['dataset']['per_flow_counts']; n = len(spec['flows'])
    assert {k: v*n for k, v in counts.items()} == spec['dataset']['expected_counts']
    write(Path(spec['output'])/'input_verification.json', dict(complete=True, identity=identity(config), files=checked))


def audit_data(spec, config):
    """Recompute every rule of protocol section 2b from the written files and the frozen fields."""
    root = Path(spec['output']); physical = json.loads(Path(spec['physical_config']).read_text()); dist = spec['distances']
    minimum = spec['proposals']['minimum_valid_lines']; cover = spec['coverage']['min_head_centers_per_instance_per_split']
    totals = {r: 0 for r in data.ROLES}; report = {}; frozen_files = {}
    for fi, flow in enumerate(spec['flows']):
        name = flow['name']; folder = root/'physical'/name; prep = json.loads((folder/'preparation.json').read_text())
        assert prep['complete'] and not prep['pilot'] and prep['identity'] == identity(config)
        scene = data.coverage.load_scene(physical, fi); threshold = float(scene['flow']['lambda2_threshold'])
        heldout = set(prep['instance_split']['heldout']); covered = prep['instance_split']['covered']
        exclusion = [(np.array(lo), np.array(hi)) for lo, hi in prep['heldout_exclusion_boxes']]
        metas = {}; index = {}; attrs = {}
        for role in data.ROLES:
            src = folder/role
            for filename, digest in prep['splits'][role]['files'].items(): assert sha(src/filename) == digest, (name, role, filename)
            frozen_files[f'{name}/{role}'] = prep['splits'][role]['files']
            with np.load(src/'metadata.npz') as z: m = {k: z[k] for k in z.files}
            with np.load(src/data.INDEX_FILE) as z: idx = {k: z[k] for k in z.files}
            with np.load(src/data.ATTRIBUTE_FILE) as z: at = {k: z[k] for k in z.files}
            metas[role], index[role], attrs[role] = m, idx, at; n = len(m['labels']); totals[role] += n
            removed = prep['unpaired_test_rows_removed'] if role == 'test' else 0
            assert n == spec['dataset']['per_flow_counts'][role]-removed and set(m) == set(data.METADATA_KEYS)
            assert np.all(m['counts'] >= minimum) and np.all(idx['original_center_id'] == 0)
            g = np.load(src/'geometry.npy', mmap_mode='r'); s = np.load(src/'seeds.npy', mmap_mode='r')
            center, _ = rules.original_center_indices(s, m); assert np.all(center == 0)
            valid = np.arange(27)[None] < m['counts'][:, None]
            ratio = m['half_arc_lengths']/m['requested_half_length'][:, None, None]
            assert np.all(ratio[valid] >= .95) and np.all(ratio[valid] <= 1.002) and not np.any((m['half_step_counts'] > m['maxiteration'][:, None, None])[valid])
            for first in range(0, n, 512):
                block = np.asarray(g[first:first+512], np.float64); v = valid[first:first+512]
                assert np.isfinite(block).all() and np.all(block[~v] == 0)
                assert np.allclose(block.sum((1, 2))/(m['counts'][first:first+512, None]*32), 0, atol=2e-6)
                assert np.allclose(np.linalg.norm(block, axis=-1).max((1, 2)), 1, atol=2e-6)
            signature = np.column_stack([m['center'], m['scale_id']]); assert len(np.unique(signature, axis=0)) == n
            # Seeds, per-line attributes and flags recomputed from the frozen fields.
            points = at['seed_points']; assert np.array_equal(np.isfinite(points).all(-1), valid) and at['exact_stencil_coordinates'].all()
            recon = rules.reconstructed_seeds(s, m); err = np.linalg.norm(np.nan_to_num(points-recon), axis=-1)/m['neighbor_distance'][:, None]
            assert err[valid].max() < 1e-3
            for i in range(0, n, max(1, n//2000)):
                grid = rules.stencil(m['center'][i], m['neighbor_distance'][i]); k = int(m['counts'][i])
                assert np.array_equal(points[i, :k], grid[at['stencil_slot'][i, :k]]) and at['stencil_slot'][i, 0] == 0
            fresh = rules.line_attributes(points, scene['axes'], scene['lambda2'], scene['oyf'], threshold)
            for key in ('lambda2', 'oyf'): assert np.allclose(np.nan_to_num(fresh[key]), np.nan_to_num(at[key]), atol=1e-6, rtol=1e-6)
            for key in ('inside', 'head_candidate', 'oyf_positive'): assert np.array_equal(fresh[key], at[key])
            box_rows = idx['sample_kind'] == data.KIND_GT_BOX
            assert np.all(at['head_candidate'][~idx['relaxed'] & ~box_rows, 0]) and float(at['lambda2_threshold']) == threshold
            # Labels, instances, groups and exclusion zones.
            owner, _ = data.sample_gt(scene['gt'], m['center'], scene['locator']); assert np.array_equal(owner, idx['center_gt_owner'])
            angle, _ = data.head_mask(data.vector_at(m['center'], scene['axes'], scene['velocity']), data.vector_at(m['center'], scene['axes'], scene['omega']))
            assert np.array_equal((owner >= 0) & angle, idx['center_in_gt_head'])
            gt_rows = idx['sample_kind'] == data.KIND_GT_HEAD
            assert np.all(m['labels'][gt_rows] == 1) and np.all(owner[gt_rows] == m['instance'][gt_rows]) and np.all(m['head_component'][gt_rows] == -m['instance'][gt_rows]-1)
            relaxed = idx['relaxed']; assert np.array_equal(idx['neighbor_filter'] == data.FILTER_RELAXED, relaxed)
            assert np.all(m['labels'][relaxed] == (owner[relaxed] >= 0)) and np.all(np.where(owner[relaxed] >= 0, owner[relaxed], -1) == m['instance'][relaxed])
            # Held-out box rows: test only, center inside that instance's GT box, label = GT membership of the center, no filter asserted.
            bounds = {int(i): (np.array(lo), np.array(hi)) for i, (lo, hi) in prep['gt_boxes'].items()}
            assert not np.any(box_rows) or role == 'test'
            for i in np.flatnonzero(box_rows):
                lo, hi = bounds[int(idx['nearest_instance'][i])]; assert np.all(m['center'][i] >= lo) and np.all(m['center'][i] <= hi) and int(idx['nearest_instance'][i]) in heldout
            assert np.all(m['labels'][box_rows] == (owner[box_rows] >= 0)) and np.all(np.where(owner[box_rows] >= 0, owner[box_rows], -1) == m['instance'][box_rows])
            assert np.array_equal(idx['heldout_instance'], np.isin(idx['nearest_instance'], list(heldout)))
            if role != 'test':
                assert not np.any(np.isin(m['instance'][m['labels'] == 1], list(heldout))) and not idx['heldout_instance'].any()
                for lo, hi in exclusion: assert not np.any(np.all((m['center'] >= lo) & (m['center'] <= hi), axis=1))
            cov = {str(i): int(np.sum(np.all((m['center'] >= bounds[i][0]) & (m['center'] <= bounds[i][1]), axis=1)) if i in heldout else np.sum(idx['center_in_gt_head'] & (owner == i)))
                   for i in prep['instance_split']['covered']+prep['instance_split']['heldout']}
            assert cov == prep['splits'][role]['coverage_head_centers_per_instance']
            if role == 'train': assert all(cov[str(i)] >= cover for i in covered)
            if role == 'test': assert all(cov[str(i)] >= cover for i in covered+list(heldout))
        # Pairing: every covered-instance test positive has a paired same-instance training center 3h..6h away.
        pair = index['train']['paired_test_center']; test_m = metas['test']; test_idx = index['test']
        assert np.all(pair[index['train']['sample_kind'] == data.KIND_HEAD_REGION] == -1)
        paired_targets = pair[pair >= 0]; offset_rows = np.flatnonzero(test_idx['offset_test'])
        assert set(paired_targets.tolist()) == set(offset_rows.tolist())
        for i in np.flatnonzero(pair >= 0):
            j = pair[i]; d = np.linalg.norm(metas['train']['center'][i]-test_m['center'][j]); h = test_m['local_grid_scale'][j]
            assert metas['train']['instance'][i] == test_m['instance'][j] and metas['train']['labels'][i] == 1 and test_m['labels'][j] == 1
            assert dist['eval_min_to_any_train_over_h']*h-1e-9 <= d <= dist['test_max_to_same_instance_train_over_h']*h+1e-9
        train = metas['train']; tree = cKDTree(train['center']); v_tree = cKDTree(metas['validation']['center'])
        inst = {i: cKDTree(train['center'][(train['labels'] == 1) & (train['instance'] == i)]) for i in covered if np.any((train['labels'] == 1) & (train['instance'] == i))}
        for role in ('validation', 'test'):
            m = metas[role]; d = tree.query(m['center'])[0]
            assert np.all(d >= dist['eval_min_to_any_train_over_h']*m['local_grid_scale']-1e-12) and np.allclose(d, m['nearest_train_center_distance'], rtol=1e-12, atol=1e-12)
            if role == 'test':
                offset = (m['labels'] == 1) & ~np.isin(m['instance'], list(heldout)); assert np.array_equal(offset, index[role]['offset_test'])
                for i in np.flatnonzero(offset):
                    same = inst[int(m['instance'][i])].query(m['center'][i])[0]
                    assert same <= dist['test_max_to_same_instance_train_over_h']*m['local_grid_scale'][i]+1e-12 and abs(same-m['nearest_same_head_train_center_distance'][i]) < 1e-9
                assert np.all(v_tree.query(m['center'])[0] >= dist['test_min_to_validation_over_h']*m['local_grid_scale']-1e-12)
        report[name] = dict(instance_split=prep['instance_split'], relaxed=prep['relaxed'], rejections=prep['rejections'],
                            splits={r: {k: prep['splits'][r][k] for k in ('samples', 'classes', 'gt_head_rows', 'relaxed', 'heldout_instance_rows', 'line_count_histogram', 'lines')} for r in data.ROLES},
                            test_composition=dict(offset_positive=int(index['test']['offset_test'].sum()), heldout_rows=int(index['test']['heldout_instance'].sum()),
                                                  heldout_positive=int(np.sum(index['test']['heldout_instance'] & (metas['test']['labels'] == 1)))))
        del scene
    removed_total = sum(json.loads((root/'physical'/f['name']/'preparation.json').read_text())['unpaired_test_rows_removed'] for f in spec['flows'])
    assert totals == dict(spec['dataset']['expected_counts'], test=spec['dataset']['expected_counts']['test']-removed_total)
    write(root/'data_audit.json', dict(complete=True, identity=identity(config), counts=totals, flows=report, frozen_files=frozen_files,
                                       rules='protocol_2b_recomputed', distances=dist, minimum_valid_lines=minimum, coverage_minimum=cover))


def runtime(spec, config, phase, state, code):
    frozen.append_locked(Path(spec['output'])/'runtime.jsonl', json.dumps(dict(identity=identity(config), phase=phase, state=state, exit_code=code,
        job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        hostname=socket.gethostname(), at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit(spec, config):
    root = Path(spec['output']); (root/'logs').mkdir(parents=True, exist_ok=True); assert not (root/'submission.json').exists(); jobs = {}
    phases = [('verify-inputs', [], None, '00:20:00', '16G'), ('pilot', ['verify-inputs'], '0-1', '04:00:00', '64G'),
              ('build', ['pilot'], '0-1', '12:00:00', '64G'), ('audit-data', ['build'], None, '02:00:00', '64G')]
    for phase, deps, array, wall, memory in phases:
        command = ['sbatch', '--parsable', '--propagate=NONE', '--job-name=FixedDS11_'+phase, '--cpus-per-task=4', '--mem='+memory, '--time='+wall,
                   '--output='+str(root/'logs/%x_%A_%a.out'), '--error='+str(root/'logs/%x_%A_%a.err')]
        if deps: command += ['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        if array: command += ['--array='+array]
        command += ['ibex_bash/task4c_fixed_dataset_1p1.sh', phase, config]
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
