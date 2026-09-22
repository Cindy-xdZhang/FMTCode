"""Physical grid scale and final connected-core arclength filtering."""
import numpy as np


def mean_grid_spacing(axes_xyz):
    """Use each full axis extent / (node count - 1), including nonuniform axes."""
    if len(axes_xyz) != 3:
        raise ValueError('Exactly three coordinate axes are required')
    spacing = []
    for axis in axes_xyz:
        axis = np.asarray(axis, dtype=np.float64)
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError('Each axis must contain at least two finite coordinates')
        step = float((axis.max() - axis.min()) / (len(axis) - 1))
        if step <= 0:
            raise ValueError('Each axis must have positive physical extent')
        spacing.append(step)
    return np.asarray(spacing)


def filter_core_lengths(cores, h, minimum_h=10.0):
    """Filter complete connected fragments; equality is retained, with no tolerance."""
    if not np.isfinite(h) or h <= 0 or minimum_h <= 0:
        raise ValueError('The scale and minimum length multiplier must be positive')
    lengths = []
    for core in cores:
        points = np.asarray(core, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('A core must have at least two finite xyz vertices')
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    lengths = np.asarray(lengths, dtype=np.float64)
    keep = lengths >= minimum_h * h
    return [core for core, valid in zip(cores, keep) if valid], lengths, keep
