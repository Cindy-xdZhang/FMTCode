"""Disjoint patch manifests for the version 5.1 pooled segmentation task.

All integer bounds use half-open z,y,x voxel indices. Splits use only instance
geometry, never model predictions or four-class validation/test performance.
"""
from __future__ import annotations
import numpy as np


def overlaps(low, high, other_low, other_high, gap=0):
    return np.all((np.asarray(low) < np.asarray(other_high) + gap) &
                  (np.asarray(high) > np.asarray(other_low) - gap), axis=-1)


def instance_boxes(ids, padding):
    shape = np.array(ids.shape)
    boxes = []
    for instance in np.unique(ids[ids > 0]):
        points = np.argwhere(ids == instance)
        boxes.append(dict(instance_id=int(instance),
            low=np.maximum(points.min(0)-padding, 0).tolist(),
            high=np.minimum(points.max(0)+1+padding, shape).tolist()))
    return boxes


def components(boxes, gap):
    lows = np.array([p['low'] for p in boxes])
    highs = np.array([p['high'] for p in boxes])
    edges = overlaps(lows[:, None], highs[:, None], lows[None], highs[None], gap)
    seen, result = set(), []
    for i in range(len(boxes)):
        if i in seen:
            continue
        stack, group = [i], []
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j); group.append(j)
            stack.extend(np.flatnonzero(edges[j]).tolist())
        result.append(sorted(group))
    return result


def select_test_components(groups, target, rng):
    # Exact subset sum; component ordering is randomized once, not searched
    # against labels or model results. Fail rather than split an instance group.
    solutions = {0: []}
    for g in rng.permutation(len(groups)):
        for size, picked in list(solutions.items()):
            new = size + len(groups[g])
            if new <= target and new not in solutions:
                solutions[new] = picked + [int(g)]
    if target not in solutions:
        raise ValueError(f'Cannot obtain {target} test instances with disjoint boxes; component sizes={[len(g) for g in groups]}')
    return {i for g in solutions[target] for i in groups[g]}


def make_manifest(ids, *, seed, training_count, test_count, test_fraction,
                  padding, gap, max_attempts):
    rng = np.random.default_rng(seed)
    boxes = instance_boxes(ids, padding)
    groups = components(boxes, gap)
    target = max(1, int(np.floor(len(boxes)*test_fraction+.5)))
    test_members = select_test_components(groups, target, rng)
    patches = []
    for i, box in enumerate(boxes):
        patches.append(dict(box, kind='hairpin_bbox', split='test' if i in test_members else 'train'))
    templates = [np.array(p['high'])-p['low'] for p in patches if p['split']=='train']
    shape = np.array(ids.shape)
    attempts = {}
    for split, count in [('test',test_count),('train',training_count)]:
        opposite = [p for p in patches if p['split'] != split]
        lo = np.array([p['low'] for p in opposite]); hi = np.array([p['high'] for p in opposite])
        known = {(tuple(p['low']), tuple(p['high'])) for p in patches}
        current = sum(p['split']==split for p in patches)
        tried = 0
        while current < count:
            tried += 1
            if tried > max_attempts:
                raise ValueError(f'Patch rejection sampling exhausted for {split}: {current}/{count}')
            size = templates[int(rng.integers(len(templates)))]
            low = rng.integers(0,shape-size+1); high = low+size
            key = tuple(low),tuple(high)
            if key in known or overlaps(low,high,lo,hi,gap).any():
                continue
            patches.append(dict(instance_id=0,kind='random_window',split=split,
                                low=low.tolist(),high=high.tolist()))
            known.add(key);current += 1
        attempts[split] = tried
    for i,p in enumerate(patches):
        p['patch_id']=i
    report = audit_manifest(ids,patches,gap)
    report.update(instance_component_sizes=[len(g) for g in groups],
                  rejection_sampling_attempts=attempts,seed=seed)
    return patches, report


def audit_manifest(ids, patches, gap):
    sets = {}
    for split in ('train','test'):
        sets[split] = {p['instance_id'] for p in patches if p['split']==split and p['kind']=='hairpin_bbox'}
    assert sets['train'].isdisjoint(sets['test'])
    assert sets['train'] | sets['test'] == set(np.unique(ids[ids>0]))
    train = [p for p in patches if p['split']=='train']
    test = [p for p in patches if p['split']=='test']
    lo = np.array([p['low'] for p in test]); hi = np.array([p['high'] for p in test])
    for p in train:
        assert not overlaps(p['low'],p['high'],lo,hi,gap).any(), 'Cross-split patch overlap'
    for p in patches:
        slices = tuple(slice(a,b) for a,b in zip(p['low'],p['high']))
        present = set(np.unique(ids[slices])) - {0}
        assert present <= sets[p['split']], 'Patch contains opposite-split hairpin instance'
    return dict(certified=True,minimum_gap_voxels=gap,cross_split_patch_overlap_count=0,
        opposite_split_instance_occurrences=0,
        patch_counts={s:sum(p['split']==s for p in patches) for s in sets},
        instance_ids={s:sorted(map(int,values)) for s,values in sets.items()},
        instance_counts={s:len(values) for s,values in sets.items()})


def local_step(seeds, inner_low, inner_high, base_step, steps, offset_ratio):
    # A unit-speed RK4 step moves at most h (intermediate queries also <=h).
    # This radius therefore bounds every query of all seven trajectories.
    margin = np.minimum(seeds-inner_low,inner_high-seeds).min(axis=-1)
    h = np.minimum(base_step, np.maximum(margin,0)*.95/(steps+offset_ratio+1))
    return h
