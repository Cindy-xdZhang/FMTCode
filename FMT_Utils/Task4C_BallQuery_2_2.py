"""Radius-limited, adaptively expanded neighborhoods on the frozen Task4-c v2 seeds."""
from pathlib import Path
from types import FunctionType
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_FixedDatasetFMT_2_1 as baseline


def minimum_grid_spacing(axes):
    """One physical h per flow: minimum positive adjacent-grid spacing over x, y, z."""
    spacings = [np.diff(np.asarray(a, np.float64)) for a in axes]
    if any(not np.all(d > 0) for d in spacings):
        raise ValueError('Grid axes must be strictly increasing')
    return min(float(d.min()) for d in spacings)


def select_fps(points, center, ids, k):
    """Center-started FPS, with ascending sample id breaking exact distance ties."""
    ids = np.sort(np.asarray(ids, np.int64))
    minimum = np.sum((points[ids] - center)**2, axis=1)
    result = np.empty(k, np.int64)
    for j in range(k):
        pick = int(np.argmax(minimum)); result[j] = ids[pick]
        distance = np.sum((points[ids] - points[ids[pick]])**2, axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[np.isin(ids, result[:j+1])] = -np.inf
    return result


def ball_tables(seeds, radius, ks=(6, 16), workers=4):
    """Exclude self; expand to the kth distance only when needed; FPS only if count > k.

    Radius expansion is computed independently for each k. No labels, folds or GT are used.
    A one-ULP outward rounding includes points exactly on the computed boundary.
    """
    seeds = np.asarray(seeds, np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError('Expected finite [n, 3] seed points')
    radius = np.broadcast_to(np.asarray(radius, np.float64), (len(seeds),))
    if not np.isfinite(radius).all() or np.any(radius <= 0) or len(seeds) <= max(ks):
        raise ValueError('Positive radius and more than max(k) samples are required')
    tree = cKDTree(seeds)
    distances, nearest = tree.query(seeds, k=max(ks)+1, workers=workers)
    if np.any(distances[:, 1] == 0):
        raise ValueError('Duplicate seeds are not allowed')
    assert np.array_equal(nearest[:, 0], np.arange(len(seeds)))
    count = tree.query_ball_point(seeds, np.nextafter(radius, np.inf), return_length=True, workers=workers)-1
    output = dict(initial_count=count.astype(np.int64))
    for k in ks:
        effective = np.maximum(radius, distances[:, k])
        expanded = count < k
        candidates = tree.query_ball_point(seeds, np.nextafter(effective, np.inf), workers=workers, return_sorted=True)
        order = np.empty((len(seeds), k), np.int64)
        for i, pool in enumerate(candidates):
            ids = np.asarray([p for p in pool if p != i], np.int64)
            # Different floating-point distance implementations can disagree by a few ULPs.
            ids = np.unique(np.r_[ids, nearest[i, 1:k+1]]) if expanded[i] else ids
            dist = np.linalg.norm(seeds[ids]-seeds[i], axis=1)
            tolerance = 8*np.finfo(np.float64).eps*max(1., effective[i])
            ids = ids[dist <= effective[i]+tolerance]
            assert len(ids) >= k, (i, k, len(ids))
            order[i] = select_fps(seeds, seeds[i], ids, k) if len(ids) > k else ids
        selected_distance = np.linalg.norm(seeds[order]-seeds[:, None], axis=2)
        assert np.all(selected_distance <= effective[:, None]+8*np.finfo(float).eps*np.maximum(1., effective[:, None]))
        assert np.all(order != np.arange(len(seeds))[:, None])
        output.update({f'order{k}': order, f'effective_radius{k}': effective,
                       f'expanded{k}': expanded, f'selected_max_distance{k}': selected_distance.max(1)})
    return output


def describe(values):
    values = np.asarray(values)
    if not len(values):
        return dict(n=0)
    return dict(n=int(len(values)), minimum=float(values.min()), maximum=float(values.max()), mean=float(values.mean()),
                quantiles=dict(zip(('p05', 'p25', 'p50', 'p75', 'p95', 'p99'), np.quantile(values, [.05, .25, .5, .75, .95, .99]).tolist())))


def count_summary(count):
    count = np.asarray(count, np.int64)
    result = describe(count)
    bins = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, np.inf]
    hist, _ = np.histogram(count, bins)
    result.update(zero=int(np.sum(count == 0)), fewer6=int(np.sum(count < 6)), fewer16=int(np.sum(count < 16)),
                  histogram=hist.tolist(), bin_labels=['0', '1', '2', '3', '4', '5', '6–7', '8–11', '12–15', '16–23',
                                                      '24–31', '32–63', '64–127', '128–255', '256–511', '512–1023', '≥1024'])
    return result


def build_neighbor_tables(spec, sha):
    root = Path(spec['output'])/'neighbors'; root.mkdir(parents=True, exist_ok=True)
    records = {}
    for flow in spec['flows']:
        folder = Path(spec['source_output'])/'physical'/flow['name']
        path = root/f"neighbors_{flow['name']}.npz"
        source_hash = sha(folder/'seeds.npy')
        h_path = Path(spec['grid_scale_output'])/f"h_{flow['name']}.npy"
        assert sha(h_path) == flow['h_sha256']
        h = np.load(h_path)
        contract = dict(seeds_sha256=source_hash, h_sha256=flow['h_sha256'], h_mode=spec['neighbors']['h_mode'], radius_h=spec['neighbors']['radius_h'],
                        ks=[6, 16], rule=spec['neighbors']['rule'])
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['contract'])) == contract
        else:
            seeds = np.load(folder/'seeds.npy')
            assert h.shape == (len(seeds),)
            arrays = ball_tables(seeds, h*spec['neighbors']['radius_h'])
            np.savez_compressed(path, contract=json.dumps(contract, sort_keys=True), **arrays)
        with np.load(path) as z:
            stats = {str(k): dict(expanded=int(z[f'expanded{k}'].sum()), effective_radius_over_h=describe(z[f'effective_radius{k}']/h),
                                 effective_radius_physical=describe(z[f'effective_radius{k}'])) for k in (6, 16)}
            stats['initial_count'] = count_summary(z['initial_count'])
        records[flow['name']] = dict(file=path.name, sha256=sha(path), contract=contract, statistics=stats)
    return records


def load_neighbors(spec, flow):
    root = Path(spec.get('_root_output', spec['output']))/'neighbors'
    with np.load(root/f'neighbors_{flow}.npz') as z:
        return z[f"order{spec['candidate']['neighbors']}"].astype(np.int64)


class Dataset(baseline.Dataset):
    """Frozen v2 cluster normalization and FMT encoding; only the neighbor table changes."""
    __init__ = FunctionType(baseline.Dataset.__init__.__code__, dict(baseline.__dict__, load_neighbors=load_neighbors),
                            argdefs=baseline.Dataset.__init__.__defaults__)

    def encode(self, candidate, batch=128):
        super().encode(candidate, batch)
        if self.evidence:
            path = self.evidence/f'{self.role}_encoding.json'
            record = json.loads(path.read_text())
            record['cluster'] = 'same-flow ball query; expand to kth neighbor if insufficient; FPS inside the ball only when more than k; same length'
            record['center_policy'] = 'one original seed, no rotation'
            path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
