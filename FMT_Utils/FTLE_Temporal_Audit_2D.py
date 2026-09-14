"""Certify disjoint integration windows and numerical source-frame support."""
from itertools import combinations

import netCDF4
import numpy as np


def certify(meta, splits, tau):
    times = None
    if meta['name'] != 'doublegyre2d':
        # Coordinates only: this check does not inspect velocity or target values.
        with netCDF4.Dataset(meta['source']) as ds:
            times = np.asarray(ds[meta['axis_names']['t']][:], dtype=np.float64)
    windows, support = {}, {}
    for split, starts in splits.items():
        windows[split], support[split] = [], set()
        for start in starts:
            end = start + tau
            if not meta['tmin'] <= start < end <= meta['tmax'] + 1e-10:
                raise ValueError('Integration window exceeds original source time range')
            row = {'t0': start, 't1': end}
            if times is not None:
                first = max(0, int(np.searchsorted(times, start, side='right')) - 1)
                last = min(len(times) - 1, int(np.searchsorted(times, end, side='left')))
                if times[first] > start or times[last] < end - 1e-10:
                    raise ValueError('Incomplete source-frame bracketing')
                support[split].update(range(first, last + 1))
                row.update(source_first_index=first, source_last_index=last,
                           source_first_time=float(times[first]), source_last_time=float(times[last]))
            windows[split].append(row)
    for a, b in combinations(windows, 2):
        for x in windows[a]:
            for y in windows[b]:
                if max(x['t0'], y['t0']) <= min(x['t1'], y['t1']):
                    raise ValueError(f'Overlapping integration windows: {a}/{b}')
        if support[a] & support[b]:
            raise ValueError(f'Shared interpolation source frames: {a}/{b}')
    return {'status': 'PASS', 'tau': tau, 'windows': windows,
            'source_frame_indices': {k: sorted(v) for k, v in support.items()},
            'meaning': 'No cross-split integration time or interpolation source-frame overlap; within-split overlap allowed.'}
