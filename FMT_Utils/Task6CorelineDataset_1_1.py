"""Task6 seed labels, whole-frame sampling and Task4-c-compatible neighborhoods."""
import time
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils.Task4C_BallQuery_3_1 import fps


def select_frames(eligible, train_count=10, test_count=45, policy=None):
    """Select times without consulting labels; roles require an explicit policy."""
    eligible = np.asarray(eligible, np.int64)
    total = train_count + test_count
    if len(eligible) < total or np.any(np.diff(eligible) <= 0):
        raise ValueError('Need enough strictly increasing eligible frame indices')
    selected = eligible[np.rint(np.linspace(0, len(eligible)-1, total)).astype(int)]
    if policy is None:
        return {'selected': selected, 'train': None, 'test': None}
    if policy == 'interleaved':
        positions = np.rint(np.linspace(0, total-1, train_count)).astype(int)
    elif policy == 'chronological':
        positions = np.arange(train_count)
    else:
        raise ValueError(policy)
    training = np.zeros(total, bool)
    training[positions] = True
    return {'selected': selected, 'train': selected[training], 'test': selected[~training]}


def coreline_labels(seeds, cores, h):
    """Exact distance to polyline segments, strict d<h; inf means no segment within h.

    The tree only eliminates seeds whose distance to a segment cannot be <h.
    No approximation or vertex-only distance determines a label.
    """
    seeds = np.asarray(seeds, np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError('Expected finite N by 3 physical seed coordinates')
    if not np.isfinite(h) or h <= 0:
        raise ValueError('h must be positive')
    tree = cKDTree(seeds)
    distance = np.full(len(seeds), np.inf, np.float64)
    owner = np.full(len(seeds), -1, np.int32)
    for cid, core in enumerate(cores):
        core = np.asarray(core, np.float64)
        if core.ndim != 2 or core.shape[1] != 3 or not np.isfinite(core).all():
            raise ValueError('Invalid core coordinates')
        if len(core) < 2:
            raise ValueError('A polyline needs at least two vertices')
        for a, b in zip(core[:-1], core[1:]):
            delta = b-a
            squared = float(delta@delta)
            radius = h + .5*np.sqrt(squared)
            tolerance = 16*np.finfo(float).eps*max(1., radius, np.abs(a).max(), np.abs(b).max())
            ids = np.asarray(tree.query_ball_point((a+b)*.5, radius+tolerance), np.int64)
            if not len(ids):
                continue
            alpha = np.clip((seeds[ids]-a)@delta/squared, 0., 1.) if squared else np.zeros(len(ids))
            d = np.linalg.norm(seeds[ids]-a-alpha[:, None]*delta, axis=1)
            update = (d < h) & (d < distance[ids])
            distance[ids[update]] = d[update]
            owner[ids[update]] = cid
    return {'label': (distance < h).astype(np.uint8), 'positive_distance': distance,
            'positive_core_id': owner}


def ball_tables(seeds, h, ks=(6, 16), rows=None, workers=4, chunk=2048,
                candidate_budget=250000, progress=None):
    """Same 3h/expansion/full-ball FPS as Task4-c, bounded candidate-list memory.

    Only centers specified by rows are computed; all seeds remain eligible neighbors.
    The candidate budget splits work, never truncates the ball or changes FPS.
    Repeated coordinates are allowed, but every selected sample ID is distinct.
    """
    seeds = np.asarray(seeds, np.float64)
    n = len(seeds)
    rows = np.arange(n) if rows is None else np.asarray(rows, np.int64)
    ks = tuple(sorted(set(ks), reverse=True))
    if seeds.shape != (n, 3) or not np.isfinite(seeds).all() or not ks or n <= max(ks):
        raise ValueError('Invalid seeds or neighbor count')
    if h <= 0 or not np.isfinite(h) or candidate_budget < 1 or chunk < 1:
        raise ValueError('Invalid spacing or memory budget')
    if rows.ndim != 1 or ((rows < 0) | (rows >= n)).any():
        raise ValueError('Invalid center IDs')
    radius = 3*float(h)
    tree = cKDTree(seeds)
    output = {'sample_id': rows.copy(), 'initial_count': np.empty(len(rows), np.int64)}
    for k in ks:
        output.update({f'order{k}': np.empty((len(rows), k), np.int64),
                       f'effective_radius{k}': np.empty(len(rows), np.float64),
                       f'expanded{k}': np.empty(len(rows), bool),
                       f'selected_max_distance{k}': np.empty(len(rows), np.float64)})
    started = time.perf_counter()
    for first in range(0, len(rows), chunk):
        last = min(first+chunk, len(rows))
        ids = rows[first:last]
        centers = seeds[ids]
        distances, nearest = tree.query(centers, k=max(ks)+1, workers=workers)
        count = tree.query_ball_point(centers, np.nextafter(radius, np.inf),
                                     return_length=True, workers=workers)-1
        output['initial_count'][first:last] = count
        for k in ks:
            effective = np.maximum(radius, distances[:, k])
            expanded = count < k
            sizes = tree.query_ball_point(centers, np.nextafter(effective, np.inf),
                                         return_length=True, workers=workers)
            lo = 0
            while lo < len(ids):
                hi = lo+1
                entries = int(sizes[lo])
                while hi < len(ids) and entries+sizes[hi] <= candidate_budget:
                    entries += int(sizes[hi]); hi += 1
                pools = tree.query_ball_point(centers[lo:hi], np.nextafter(effective[lo:hi], np.inf),
                                             workers=workers, return_sorted=True)
                for j, pool in enumerate(pools, lo):
                    center_id = ids[j]
                    candidates = np.asarray(pool, np.int64)
                    candidates = candidates[candidates != center_id]
                    if expanded[j]:
                        extra = nearest[j][nearest[j] != center_id][:k]
                        candidates = np.unique(np.r_[candidates, extra])
                    d = np.linalg.norm(seeds[candidates]-centers[j], axis=1)
                    tolerance = 8*np.finfo(float).eps*max(1., effective[j])
                    candidates = candidates[d <= effective[j]+tolerance]
                    if len(candidates) < k:
                        raise RuntimeError('Expanded ball has insufficient distinct sample IDs')
                    selected = fps(seeds, centers[j], candidates, k) if len(candidates) > k else candidates
                    output[f'order{k}'][first+j] = selected
                    output[f'selected_max_distance{k}'][first+j] = np.linalg.norm(seeds[selected]-centers[j], axis=1).max()
                lo = hi
            output[f'effective_radius{k}'][first:last] = effective
            output[f'expanded{k}'][first:last] = expanded
        if progress:
            progress({'completed': last, 'total': len(rows), 'seconds': time.perf_counter()-started})
    return output


def encode_bundles(curves, seeds, center_ids, neighbors, candidate, device='cpu'):
    """One Task4-c FMT token per original seed, from instantaneous v-u curves only."""
    import torch
    from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import resample
    from FMT_Utils.Task4C_FixedDatasetFMT_2_1 import normalize_bundle
    from FMT_Utils.Task4C_FPS16_1_1 import encode
    from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt
    ids = np.column_stack((center_ids, neighbors))
    k = candidate['neighbors']
    if ids.shape[1] != k+1 or any(len(np.unique(row)) != k+1 for row in ids):
        raise ValueError('Need one center and k distinct other sample IDs')
    with torch.no_grad():
        lines = torch.as_tensor(np.array(curves[ids]), device=device)
        points = torch.as_tensor(np.array(seeds[ids]), device=device)
        lines = resample(lines, 32, 'uniform')
        geometry, points, counts = normalize_bundle(lines, points)
        tokens, used = encode(geometry, points, counts,
                              torch.zeros(len(ids), dtype=torch.long, device=device), candidate)
        expected = torch.arange(1, k+1, device=device)[None].expand(len(ids), -1)
        if not torch.equal(used.sort(1).values, expected):
            raise RuntimeError('Frozen encoder changed the ball-query neighbor set')
        values = tokens.cpu().numpy()
        if candidate['base_method'] == 'p35_h0':
            values[..., :141] = transform_fmt(values[..., :141], True)
    if values.shape != (len(ids), 1, 142) or not np.isfinite(values).all():
        raise RuntimeError('Invalid FMT features')
    return values
