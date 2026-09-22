"""Native-cell candidate membership, independent of GT and region labels."""
import numpy as np


def candidate_at_seeds(axes, lam, oyf, seeds, threshold):
    indices = [np.searchsorted(a, seeds[:, d], side='right')-1 for d, a in enumerate(axes)]
    assert all(np.all((p >= 0) & (p < len(a)-1)) for p, a in zip(indices, axes))
    x, y, z = indices
    good = np.ones(len(seeds), bool); average = np.zeros(len(seeds), np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                good &= lam[z+dz, y+dy, x+dx] < threshold
                average += oyf[z+dz, y+dy, x+dx]/8
    return good & (average > 0)
