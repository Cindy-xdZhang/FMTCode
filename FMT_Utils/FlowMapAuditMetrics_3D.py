"""Independent Euclidean-distance replay with all pairs and less allocation."""
import numpy as np
from scipy.spatial.distance import pdist

def measured_trajectories(prediction, truth, radius, chunk=8):
    """Bounded-memory physical metrics; no removal of bad predicted trajectories."""
    p, t = np.asarray(prediction), np.asarray(truth)
    r = np.asarray(radius, float)
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Predictions and physical truth must be finite and correspond')
    n, q, times, _ = t.shape
    sums = {k: 0. for k in ('squared', 'seen_squared', 'unseen_squared', 'physical', 'position', 'shape', 'pairs')}
    time_error = np.zeros(times)
    bundle_error = []
    pair = np.triu_indices(q, 1)
    for i in range(0, n, chunk):
        a, b = p[i:i + chunk].astype(float), t[i:i + chunk].astype(float)
        rr = r[i:i + chunk, None, None]
        error = np.linalg.norm(a - b, axis=-1)
        e = error / rr
        sums['squared'] += float((e[:, :, 1:] ** 2).sum())
        sums['seen_squared'] += float((e[:, :, 2::2] ** 2).sum())
        sums['unseen_squared'] += float((e[:, :, 1::2] ** 2).sum())
        sums['physical'] += float(error[:, :, 1:].sum())
        sums['position'] += float(e[:, :, 1:].sum())
        se = np.linalg.norm((a - a.mean(1, keepdims=True)) - (b - b.mean(1, keepdims=True)), axis=-1) / rr
        sums['shape'] += float(se[:, :, 1:].sum())
        for region in range(len(a)):
            for moment in range(1, times):
                pa = pdist(a[region, :, moment], metric='euclidean')
                pb = pdist(b[region, :, moment], metric='euclidean')
                sums['pairs'] += float(np.abs(pa-pb).sum() / r[i+region])
        time_error += e.sum((0, 1))
        bundle_error.extend(e[:, :, 1:].mean((1, 2)).tolist())
    count = n * q * (times - 1)
    return dict(position_nrmse=float(np.sqrt(sums['squared'] / count)),
        supervised_time_nrmse=float(np.sqrt(sums['seen_squared'] / (n * q * len(range(2, times, 2))))),
        unseen_time_nrmse=float(np.sqrt(sums['unseen_squared'] / (n * q * len(range(1, times, 2))))),
        mean_position_error=sums['position'] / count, physical_mean_position_error=sums['physical'] / count,
        centered_shape_error=sums['shape'] / count, pair_distance_error=sums['pairs'] / (n * len(pair[0]) * (times - 1)),
        endpoint_error=float(time_error[-1] / (n * q)), time_position_error=(time_error / (n * q)).tolist(),
        bundle_position_error=bundle_error, bundles=n, particles=q, times=times,
        normalization='initial primitive radius; all noninitial particles/times; odd indices withheld from supervision')
