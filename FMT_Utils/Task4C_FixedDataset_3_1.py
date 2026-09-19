"""Density-only Task4-c v3: sparse Poisson indexing, frozen v2 scientific rules."""
from types import FunctionType, SimpleNamespace
import numpy as np
from numba import njit
from FMT_Utils import Task4C_FixedDataset_2_1 as v2


@njit(cache=True)
def _slot(keys, key):
    slot = np.int64((np.uint64(key) * np.uint64(11400714819323198485)) & np.uint64(len(keys)-1))
    while keys[slot] != -1 and keys[slot] != key:
        slot = (slot+1) & (len(keys)-1)
    return slot


@njit(cache=True)
def _sparse_dart(points, radius, low, dims, target, table_size):
    keys = np.full(table_size, -1, np.int64)
    heads = np.full(table_size, -1, np.int32)
    links = np.full(target, -1, np.int32)
    accepted = np.empty(target, np.int64)
    found = 0
    r2 = radius*radius
    for idx in range(len(points)):
        x, y, z = points[idx, 0], points[idx, 1], points[idx, 2]
        cx = int((x-low[0])/radius)
        cy = int((y-low[1])/radius)
        cz = int((z-low[2])/radius)
        ok = True
        for a in range(max(cx-1, 0), min(cx+2, dims[0])):
            for b in range(max(cy-1, 0), min(cy+2, dims[1])):
                for c in range(max(cz-1, 0), min(cz+2, dims[2])):
                    key = (a*dims[1]+b)*dims[2]+c
                    slot = _slot(keys, key)
                    entry = heads[slot]
                    while entry != -1:
                        j = accepted[entry]
                        dx, dy, dz = points[j, 0]-x, points[j, 1]-y, points[j, 2]-z
                        if dx*dx+dy*dy+dz*dz < r2:
                            ok = False
                            break
                        entry = links[entry]
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            key = (cx*dims[1]+cy)*dims[2]+cz
            slot = _slot(keys, key)
            keys[slot] = key
            links[found] = heads[slot]
            heads[slot] = found
            accepted[found] = idx
            found += 1
            if found == target:
                break
    return accepted[:found]


def dart_throw(points, radius, target, capacity=8, max_cells=None):
    """Same ordered greedy distance test as v2, memory proportional to accepted points."""
    points = np.ascontiguousarray(points, np.float64)
    assert len(points) >= target > 0 and target < 2**31 and radius > 0
    assert np.isfinite(points).all()
    low = points.min(0)-1e-12
    dims = np.floor((points.max(0)-low)/radius).astype(np.int64)+1
    assert all(int(v) > 0 for v in dims)
    assert int(dims[0])*int(dims[1])*int(dims[2]) < 2**63
    table_size = 1 << (2*target-1).bit_length()
    return _sparse_dart(points, float(radius), low, dims, int(target), table_size)


poisson_disk = FunctionType(v2.poisson_disk.__code__, dict(v2.__dict__, dart_throw=dart_throw),
                            argdefs=v2.poisson_disk.__defaults__)


def fixed_instance_folds(instances, low, high, counts, folds, seed, reference):
    """Keep the exact v2 instance-to-fold map; report new sample counts in those folds."""
    fold_of = {int(k): int(v) for k, v in reference['fold_of_instance'].items()}
    assert set(fold_of) == set(int(i) for i in instances)
    assert all(0 <= f < folds for f in fold_of.values())
    for i, identifier in enumerate(instances):
        for j in range(i):
            if np.all(low[i] <= high[j]) and np.all(low[j] <= high[i]):
                assert fold_of[int(identifier)] == fold_of[int(instances[j])]
    totals = [sum(int(counts.get(i, 0)) for i in fold_of if fold_of[i] == f) for f in range(folds)]
    groups = [dict(group=g['group'], fold=g['fold'], samples=sum(int(counts.get(i, 0)) for i in g['group']))
              for g in reference['groups']]
    return fold_of, dict(groups=groups, fold_samples=totals, source='frozen_v2_r3_instance_map')


def prepare(spec, index, write, sha, identity, pilot=False):
    reference = spec['reference_folds'][spec['flows'][index]['name']]
    def folds(instances, low, high, counts, count, seed):
        return fixed_instance_folds(instances, low, high, counts, count, seed, reference)
    build_rows = FunctionType(v2.build_rows.__code__, dict(v2.__dict__, instance_folds=folds))
    fn = FunctionType(v2.prepare.__code__, dict(v2.__dict__, poisson_disk=poisson_disk, build_rows=build_rows),
                      argdefs=v2.prepare.__defaults__)
    return fn(spec, index, write, sha, identity, pilot)


def adapter():
    return SimpleNamespace(**dict(v2.__dict__, dart_throw=dart_throw, poisson_disk=poisson_disk, prepare=prepare))
