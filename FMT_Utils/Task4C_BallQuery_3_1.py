"""Bounded-memory, exact 3h ball query for the dense Task4-c v3 dataset."""
import time
import json
import numpy as np
from numba import njit
from scipy.spatial import cKDTree


@njit(cache=True)
def fps(points, center, ids, k):
    minimum = np.empty(len(ids), np.float64)
    for i in range(len(ids)):
        j = ids[i]
        dx, dy, dz = points[j, 0]-center[0], points[j, 1]-center[1], points[j, 2]-center[2]
        minimum[i] = dx*dx+dy*dy+dz*dz
    result = np.empty(k, np.int64)
    for t in range(k):
        pick = np.argmax(minimum)
        selected = ids[pick]
        result[t] = selected
        for i in range(len(ids)):
            j = ids[i]
            dx = points[j, 0]-points[selected, 0]
            dy = points[j, 1]-points[selected, 1]
            dz = points[j, 2]-points[selected, 2]
            minimum[i] = min(minimum[i], dx*dx+dy*dy+dz*dz)
        minimum[pick] = -np.inf
    return result


def ball_tables(seeds, radius, ks=(6, 16), workers=4, chunk=2048, progress=False):
    seeds = np.asarray(seeds, np.float64)
    n = len(seeds)
    radius = np.broadcast_to(np.asarray(radius, np.float64), (n,))
    assert seeds.shape == (n, 3) and np.isfinite(seeds).all()
    assert np.isfinite(radius).all() and (radius > 0).all() and n > max(ks)
    tree = cKDTree(seeds)
    output = {'initial_count': np.empty(n, np.int64)}
    for k in ks:
        output.update({f'order{k}': np.empty((n, k), np.int64),
            f'effective_radius{k}': np.empty(n, np.float64), f'expanded{k}': np.empty(n, bool),
            f'selected_max_distance{k}': np.empty(n, np.float64)})
    started = time.perf_counter()
    for first in range(0, n, chunk):
        last = min(first+chunk, n)
        centers = seeds[first:last]
        distances, nearest = tree.query(centers, k=max(ks)+1, workers=workers)
        assert (distances[:, 1] > 0).all()
        np.testing.assert_array_equal(nearest[:, 0], np.arange(first, last))
        count = tree.query_ball_point(centers, np.nextafter(radius[first:last], np.inf),
                                      return_length=True, workers=workers)-1
        output['initial_count'][first:last] = count
        # Only this small block materializes candidate lists; never n global Python lists.
        for k in sorted(ks, reverse=True):
            effective = np.maximum(radius[first:last], distances[:, k])
            expanded = count < k
            candidates = tree.query_ball_point(centers, np.nextafter(effective, np.inf), workers=workers, return_sorted=True)
            order = output[f'order{k}'][first:last]
            for local, pool in enumerate(candidates):
                i = first+local
                ids = np.asarray([p for p in pool if p != i], np.int64)
                if expanded[local]:
                    ids = np.unique(np.r_[ids, nearest[local, 1:k+1]])
                distance = np.linalg.norm(seeds[ids]-seeds[i], axis=1)
                tolerance = 8*np.finfo(float).eps*max(1., effective[local])
                ids = ids[distance <= effective[local]+tolerance]
                assert len(ids) >= k
                order[local] = fps(seeds, seeds[i], ids, k) if len(ids) > k else ids
            selected = np.linalg.norm(seeds[order]-centers[:, None], axis=2)
            assert (selected <= effective[:, None]+8*np.finfo(float).eps*np.maximum(1., effective[:, None])).all()
            assert (order != np.arange(first, last)[:, None]).all()
            output[f'effective_radius{k}'][first:last] = effective
            output[f'expanded{k}'][first:last] = expanded
            output[f'selected_max_distance{k}'][first:last] = selected.max(1)
        if progress and (first == 0 or last == n or first//chunk % 50 == 0):
            print(json.dumps(dict(stage='neighbors', completed=last, total=n, seconds=time.perf_counter()-started)), flush=True)
    return output
