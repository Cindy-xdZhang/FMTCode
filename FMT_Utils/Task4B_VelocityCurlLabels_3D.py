"""Task4-b 4.1: strict coverage threshold and exhaustive velocity-curl labels."""

import numpy as np

CLASS_NAMES = ("ordinary_streamwise", "ordinary_spanwise", "hairpin_head", "hairpin_leg")


def coverage_threshold(hairpin_ivd):
    values = np.asarray(hairpin_ivd, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Hairpin IVD must be nonempty, finite and nonnegative")
    minimum = float(values.min())
    if minimum == 0:
        raise ValueError("No nonnegative a can satisfy strict IVD > a for a zero-IVD annotation")
    a = float(np.nextafter(minimum, -np.inf))
    return {"minimum_hairpin_ivd": minimum, "a": a, "vortex_threshold": 0.9 * a}


def velocity_curl_labels(velocity, curl, hairpin):
    velocity, curl = np.asarray(velocity, dtype=np.float64), np.asarray(curl, dtype=np.float64)
    hairpin = np.asarray(hairpin, dtype=bool)
    if velocity.shape != curl.shape or velocity.shape != (len(hairpin), 3):
        raise ValueError("Expected matching [N,3] vectors and [N] membership")
    speed, magnitude = np.linalg.norm(velocity, axis=1), np.linalg.norm(curl, axis=1)
    valid = np.isfinite(velocity).all(1) & np.isfinite(curl).all(1) & (speed > 0) & (magnitude > 0)
    cosine = np.full(len(hairpin), np.nan)
    cosine[valid] = np.clip(np.abs(np.sum(velocity[valid] * curl[valid], axis=1)) / (speed[valid] * magnitude[valid]), 0, 1)
    # Squared cosine avoids arccos rounding at the exact 45-degree tie.
    parallel = cosine * cosine >= 0.5 - 4 * np.finfo(np.float64).eps
    labels = np.where(hairpin, np.where(parallel, 3, 2), np.where(parallel, 0, 1)).astype(np.int8)
    labels[~valid] = -1
    return labels, cosine, valid


def spatial_mean(vector_zyx3, axes_zyx, periodic_xy=False):
    """Volume mean using trapezoidal weights; periodic axes use equal spacing."""
    result = np.asarray(vector_zyx3, dtype=np.float64)
    for axis, coordinates in zip((2, 1, 0), axes_zyx[::-1]):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if len(coordinates) != result.shape[axis] or np.any(np.diff(coordinates) <= 0):
            raise ValueError("Invalid physical grid coordinates")
        if periodic_xy and axis in (1, 2):
            if not np.allclose(np.diff(coordinates), np.diff(coordinates).mean(), rtol=1e-3):
                raise ValueError("Periodic axes must be uniformly spaced")
            weights = np.ones(len(coordinates))
        else:
            intervals = np.diff(coordinates)
            weights = np.r_[intervals[0], intervals[:-1] + intervals[1:], intervals[-1]] / 2
        result = np.average(result, axis=axis, weights=weights)
    return result
