"""Local-neighborhood binary labels derived from a 3D IVD volume."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import percentile_filter, uniform_filter


def _validate_neighborhood_size(neighborhood_size):
    size = int(neighborhood_size)
    if size < 1 or size % 2 == 0:
        raise ValueError("neighborhood_size must be a positive odd integer")
    return size


def local_ivd_threshold_3d(ivd_volume, neighborhood_size, mode="mean_fraction",
                           value=0.9):
    """Compute a local threshold volume without using class labels.

    ``mean_fraction`` means ``value * local_mean(IVD)``. ``percentile`` means
    the local ``value``-th percentile, for example ``value=90``.
    """
    ivd = np.asarray(ivd_volume, dtype=np.float32)
    if ivd.ndim != 3 or not np.isfinite(ivd).all():
        raise ValueError("ivd_volume must be a finite 3-D array")
    size = _validate_neighborhood_size(neighborhood_size)
    if mode == "mean_fraction":
        fraction = float(value)
        if not 0.0 < fraction:
            raise ValueError("mean_fraction value must be positive")
        threshold = fraction * uniform_filter(ivd, size=size, mode="nearest")
    elif mode == "percentile":
        percentile = float(value)
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile value must be in [0,100]")
        threshold = percentile_filter(ivd, percentile=percentile, size=size,
                                      mode="nearest")
    else:
        raise ValueError("mode must be 'mean_fraction' or 'percentile'")
    return np.asarray(threshold, dtype=np.float32)


def local_ivd_labels_3d(ivd_volume, neighborhood_size, mode="mean_fraction",
                        value=0.9):
    """Return ``IVD > local_threshold`` on the full voxel grid."""
    ivd = np.asarray(ivd_volume, dtype=np.float32)
    threshold = local_ivd_threshold_3d(ivd, neighborhood_size, mode, value)
    return ivd > threshold, threshold


def sample_local_ivd_labels_3d(ivd_volume, axes_xyz, seeds_xyz, neighborhood_size,
                               mode="mean_fraction", value=0.9):
    """Interpolate IVD and its local threshold to arbitrary physical seed points."""
    ivd = np.asarray(ivd_volume, dtype=np.float32)
    threshold = local_ivd_threshold_3d(ivd, neighborhood_size, mode, value)
    xs, ys, zs = (np.asarray(axis, dtype=np.float64) for axis in axes_xyz)
    sample_zyx = np.asarray(seeds_xyz, dtype=np.float64)[:, [2, 1, 0]]
    ivd_at_seeds = RegularGridInterpolator(
        (zs, ys, xs), ivd, bounds_error=True
    )(sample_zyx).astype(np.float32)
    threshold_at_seeds = RegularGridInterpolator(
        (zs, ys, xs), threshold, bounds_error=True
    )(sample_zyx).astype(np.float32)
    return ivd_at_seeds > threshold_at_seeds, ivd_at_seeds, threshold_at_seeds
