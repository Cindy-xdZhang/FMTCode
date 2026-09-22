"""16h post-filter of frozen 2.2 cores; retain every seed, curve and neighbor."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from FMT_Utils.Task6CoreLength_3D import filter_core_lengths
from FMT_Utils.Task6CorelineDataset_1_1 import coreline_labels
from FMT_Utils.Task6Sampling_2_1 import SegmentQuery
from experiments.Filter_Task6_CoreLength_1_4 import read_cores, sha256
from experiments.Extract_Task6_VortexCore_1_2 import write_lines

SOURCE = ROOT / 'outputs/mainExp_Task6_CorelineDataset_2.2'
TARGET = ROOT / 'outputs/mainExp_Task6_CorelineDataset_2.3'
PARENT_SHA = 'ab5694e5132fbfff3fae716bc8404fc228800f9d54881865486e386f9b7384f2'
VERSION = 'mainExp_Task6_CorelineDataset_2.3'
CHANGED = {'corelines.vtp', 'labels.npy', 'positive_distance.npy', 'positive_core_id.npy',
           'sample_seeds.npz', 'length_filter.json', 'sampling.json', 'candidate_core_support.json',
           'audit.json', 'completed.json'}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(p)


def reuse(source, target):
    """Shared immutable files; all changed artifacts are separate new files."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha256(source) == sha256(target)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def remap_ids(values, mapping):
    values = np.asarray(values)
    result = np.full(values.shape, -1, np.int32)
    valid = values >= 0
    assert not valid.any() or values[valid].max() < len(mapping)
    result[valid] = mapping[values[valid]]
    return result


def run_frame(position):
    started = time.perf_counter()
    parent = read(SOURCE / 'dataset_frozen.json')['frames'][position]
    flow, index, h = parent['flow'], parent['index'], parent['h']
    src = SOURCE / flow / f'frame_{index:03d}'
    dst = TARGET / flow / f'frame_{index:03d}'
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / 'frame_result.json').exists():
        return read(dst / 'frame_result.json')
    completed = read(src / 'completed.json')
    for name, digest in completed['files'].items():
        assert sha256(src / name) == digest, (flow, index, name)
        if name not in CHANGED:
            reuse(src / name, dst / name)
    old_cores = read_cores(src / 'corelines.vtp')
    cores, lengths, keep = filter_core_lengths(old_cores, h, minimum_h=16.)
    kept = np.flatnonzero(keep)
    mapping = np.full(len(old_cores), -1, np.int32)
    mapping[kept] = np.arange(len(kept), dtype=np.int32)
    write_lines(dst / 'corelines.vtp', cores)
    saved = read_cores(dst / 'corelines.vtp')
    for a, b in zip(cores, saved):
        np.testing.assert_array_equal(a, b)
    old_lengths = read(src / 'length_filter.json')
    merged_lengths = np.array(old_lengths['lengths'])
    merged_keep = merged_lengths >= 16 * h
    retained_merged_ids = np.asarray(old_lengths['retained_merged_ids'])[kept].tolist()
    assert retained_merged_ids == np.flatnonzero(merged_keep).tolist()
    length_report = {**old_lengths, 'version': VERSION, 'stage': 'post-filter existing merged geometry; no re-extraction',
                     'minimum_length_h': 16., 'minimum_length': 16*h, 'retained_count': len(cores),
                     'retained_merged_ids': retained_merged_ids,
                     'removed_merged_ids': np.flatnonzero(~merged_keep).tolist(),
                     'total_retained_length': float(lengths[keep].sum()),
                     'parent_core_count': len(old_cores), 'retained_parent_core_ids': kept.tolist(),
                     'removed_parent_core_ids': np.flatnonzero(~keep).tolist(),
                     'parent_to_current_core_ids': mapping.tolist(),
                     'corelines_sha256': sha256(dst / 'corelines.vtp')}
    write(dst / 'length_filter.json', length_report)
    seeds = np.load(src / 'seeds.npy', mmap_mode='r')
    old_labels = np.load(src / 'labels.npy')
    old_owner = np.load(src / 'positive_core_id.npy')
    # Removal can never turn an old negative into a positive. Re-query every old
    # positive against all retained cores, including overlapping neighborhoods.
    rows = np.flatnonzero(old_labels)
    relabeled = coreline_labels(seeds[rows], cores, h)
    labels = np.zeros(len(seeds), np.uint8)
    distances = np.full(len(seeds), np.inf, np.float64)
    owner = np.full(len(seeds), -1, np.int32)
    labels[rows] = relabeled['label']
    distances[rows] = relabeled['positive_distance']
    owner[rows] = relabeled['positive_core_id']
    # Independent, reverse search: segment-midpoint tree queried by all400k seeds.
    independently = SegmentQuery(cores).distance(seeds, h, workers=2)
    np.testing.assert_array_equal(labels, independently < h)
    np.testing.assert_allclose(distances[labels == 1], independently[labels == 1], rtol=1e-10, atol=1e-12)
    assert np.all(labels <= old_labels)
    np.save(dst / 'labels.npy', labels)
    np.save(dst / 'positive_distance.npy', distances)
    np.save(dst / 'positive_core_id.npy', owner)
    with np.load(src / 'sample_seeds.npz') as z:
        attributes = {name: z[name] for name in z.files}
    for name in ('sampling_target_core_id', 'near_core_id'):
        attributes['parent_' + name] = attributes[name].copy()
        attributes[name] = remap_ids(attributes[name], mapping)
    np.savez_compressed(dst / 'sample_seeds.npz', **attributes)
    with np.load(src / 'sample_seeds.npz') as before, np.load(dst / 'sample_seeds.npz') as after:
        for name in before.files:
            if name not in ('sampling_target_core_id', 'near_core_id'):
                np.testing.assert_array_equal(before[name], after[name])
    old_support = read(src / 'candidate_core_support.json')
    support_rows = []
    for old_id in kept:
        r = copy.deepcopy(old_support['cores'][int(old_id)])
        r.update(core=int(mapping[old_id]), parent_core=int(old_id))
        support_rows.append(r)
    support = dict(cores=support_rows, eligible=[r['core'] for r in support_rows if r['candidate_support']],
                   absent=[r['core'] for r in support_rows if not r['candidate_support']],
                   geometry_kept=True, source_dataset=SOURCE.name)
    write(dst / 'candidate_core_support.json', support)
    sampling = read(src / 'sampling.json')
    sampling.update(version=VERSION, sampling_reused=True, source_dataset=SOURCE.name,
                    original_sampling=sampling.copy(), eligible_cores=support['eligible'],
                    near_core_quota=int((attributes['sampling_target_core_id'] >= 0).sum()),
                    formerly_targeted_removed_cores=int(((attributes['parent_sampling_target_core_id'] >= 0) &
                                                        (attributes['sampling_target_core_id'] < 0)).sum()),
                    quota_reallocated=False, note='All original samples retained; target IDs remapped; no new quotas imposed.')
    write(dst / 'sampling.json', sampling)
    old_audit = read(src / 'audit.json')
    coverage = []
    for old_id in kept:
        c = copy.deepcopy(old_audit['coverage'][int(old_id)])
        c.update(core=int(mapping[old_id]), parent_core=int(old_id))
        coverage.append(c)
    audit = {**old_audit, 'version': VERSION, 'cores': len(cores), 'positive': int(labels.sum()),
             'negative': int(len(labels)-labels.sum()), 'coverage': coverage,
             'source_audit_sha256': sha256(src / 'audit.json'),
             'integration_reused_without_recomputation': True, 'seeds_and_curves_unchanged': True,
             'minimum_merged_arc_length_h': 16., 'all_labels_independently_recomputed': True,
             'labels_independently_checked_rows': len(seeds),
             'positive_to_negative': int((old_labels-labels).sum()), 'negative_to_positive': 0}
    write(dst / 'audit.json', audit)
    neighbor_src = SOURCE / 'neighbors' / flow / f'frame_{index:03d}'
    neighbor_dst = TARGET / 'neighbors' / flow / f'frame_{index:03d}'
    nm = read(neighbor_src / 'manifest.json')
    for name, digest in nm['files'].items():
        assert sha256(neighbor_src / name) == digest
        reuse(neighbor_src / name, neighbor_dst / name)
    reuse(neighbor_src / 'manifest.json', neighbor_dst / 'manifest.json')
    source_owners_retained = (old_owner >= 0) & (remap_ids(old_owner, mapping) >= 0)
    reassigned = (old_labels == 1) & ~source_owners_retained & (labels == 1)
    result = {**parent, 'source_dataset': SOURCE.name, 'parent_cores': len(old_cores), 'cores': len(cores),
              'removed_cores': int((~keep).sum()), 'parent_positive': int(old_labels.sum()),
              'positive': int(labels.sum()), 'negative': int(len(labels)-labels.sum()),
              'positive_to_negative': int((old_labels-labels).sum()),
              'still_positive_after_removed_nearest_core': int(reassigned.sum()),
              'minimum_length': 16*h, 'minimum_length_h': 16.,
              'retained_arc_length': float(lengths[keep].sum()),
              'candidate_unsupported_cores': support['absent'],
              'positive_per_core': np.bincount(owner[owner >= 0], minlength=len(cores)).tolist(),
              'retained_parent_core_ids': kept.tolist(),
              'source_manifest': 'deployment_source_sha256.json',
              'candidate_sampler_revision': 'reuse_2.2_without_resampling',
              'parent_source_manifest': parent['source_manifest'],
              'seconds': time.perf_counter()-started}
    for name, field in [('corelines.vtp','core_sha256'), ('labels.npy','labels_sha256'),
                        ('seeds.npy','seeds_sha256'), ('sample_streamlines.npy','curves_sha256'),
                        ('audit.json','audit_sha256')]:
        result[field] = sha256(dst / name)
    assert result['seeds_sha256'] == parent['seeds_sha256'] and result['curves_sha256'] == parent['curves_sha256']
    write(dst / 'completed.json', dict(passed=True, frame=completed['frame'], neighbors_complete=True,
          source_manifest_sha256=sha256(TARGET / 'deployment_source_sha256.json'),
          source_dataset=SOURCE.name, source_completed_sha256=sha256(src / 'completed.json'),
          files={p.name:sha256(p) for p in dst.iterdir() if p.is_file() and p.name not in ('completed.json','frame_result.json')}))
    write(dst / 'frame_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    assert sha256(SOURCE / 'dataset_frozen.json') == PARENT_SHA
    TARGET.mkdir(parents=True, exist_ok=True)
    source_files = ['experiments/Filter_Task6_Dataset_2_3.py', 'FMT_Utils/Task6CoreLength_3D.py',
                    'FMT_Utils/Task6CorelineDataset_1_1.py', 'FMT_Utils/Task6Sampling_2_1.py',
                    'experiments/Filter_Task6_CoreLength_1_4.py', 'experiments/Extract_Task6_VortexCore_1_2.py',
                    'FMT_Utils/Task6VortexCore_3D.py']
    hashes = {name:sha256(ROOT / name) for name in source_files}
    if (TARGET / 'deployment_source_sha256.json').exists():
        assert read(TARGET / 'deployment_source_sha256.json') == hashes
    else:
        write(TARGET / 'deployment_source_sha256.json', hashes)
    inventory = read(SOURCE / 'frame_inventory.json')
    inventory.update(version=VERSION, parent_frozen_sha256=PARENT_SHA, cleanup_policy='Reuse2.2 samples; retain merged physical arc length >=16h')
    inventory['extraction']['minimum_merged_arc_length_h'] = 16.
    write(TARGET / 'frame_inventory.json', inventory)
    indices = [1] if args.pilot else list(range(61))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_frame, i):i for i in indices}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            report = dict(completed=len(results), total=len(indices), flow=result['flow'], index=result['index'],
                          before=result['parent_cores'], after=result['cores'], positive=result['positive'],
                          positive_to_negative=result['positive_to_negative'])
            print(json.dumps(report), flush=True)
            write(TARGET / 'run_status.json', dict(state='FILTERING', **report))
    if args.pilot:
        return
    results.sort(key=lambda f:next(i for i,r in enumerate(inventory['frames']) if (r['flow'],r['index']) == (f['flow'],f['index'])))
    parent = read(SOURCE / 'dataset_frozen.json')
    frozen = dict(complete=True, version=VERSION, parent_version=parent['version'], parent_frozen_sha256=PARENT_SHA,
                  counts=parent['counts'], train_frames=40, test_frames=21, frames=results,
                  minimum_merged_arc_length_h=16., all_samples_and_neighbors_reused=True,
                  source_manifests={sha256(TARGET/'deployment_source_sha256.json'):'deployment_source_sha256.json'},
                  inventory_sha256=sha256(TARGET/'frame_inventory.json'),
                  total_corelines=sum(r['cores'] for r in results), total_positive=sum(r['positive'] for r in results),
                  total_samples=24400000, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(TARGET / 'dataset_frozen.json', frozen)
    summary = dict(complete=True, frozen_sha256=sha256(TARGET/'dataset_frozen.json'), frames=61, samples=24400000,
                   cores_before=sum(r['parent_cores'] for r in results), cores_after=frozen['total_corelines'],
                   positive_before=sum(r['parent_positive'] for r in results), positive_after=frozen['total_positive'],
                   positive_to_negative=sum(r['positive_to_negative'] for r in results),
                   overlap_reassignments=sum(r['still_positive_after_removed_nearest_core'] for r in results),
                   per_flow={flow:dict(cores_before=sum(r['parent_cores'] for r in results if r['flow']==flow),
                                      cores_after=sum(r['cores'] for r in results if r['flow']==flow),
                                      positive_before=sum(r['parent_positive'] for r in results if r['flow']==flow),
                                      positive_after=sum(r['positive'] for r in results if r['flow']==flow))
                             for flow in dict.fromkeys(r['flow'] for r in results)})
    write(TARGET / 'completion_summary.json', summary)
    write(TARGET / 'run_status.json', dict(state='COMPLETED_FROZEN', **summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
