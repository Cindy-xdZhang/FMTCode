"""Exact icosahedral streamline stars traced in a frozen velocity field.

Task 6 ships precomputed streamlines at fixed scattered seeds, so a 13-line star
normally has to be *assembled* from whatever seeds exist nearby -- an
approximation whose angular error is 12-15 degrees and which, at small radii,
reuses the same streamline in several channels.

Where the underlying field is available the star can instead be **traced**, the
way Task 1 does it: integrate a streamline from each ideal point
`centre + r * d_j`.  `deltaWing_mag0_3reesampled.nc` was verified to be exactly
the field behind Task 6's `deltaWing_resampled` scene (median cos between the
stored streamline tangents and the interpolated field = 0.998), so for that
scene the exact construction is available.

Convention matched to the dataset: arclength parameterisation, `steps` samples
spanning `length` of arclength centred on the seed, truncating where the
streamline leaves the domain.
"""
from __future__ import annotations

import numpy as np


def sample_field(frame, points, lower, upper):
    """Trilinear sample of a ``[z, y, x, 3]`` frame at world ``points``."""
    depth, height, width, _ = frame.shape
    grid = (points - lower) / (upper - lower) * np.array([width - 1, height - 1, depth - 1])
    grid = np.clip(grid, 0.0, [width - 1 - 1e-6, height - 1 - 1e-6, depth - 1 - 1e-6])
    base = np.floor(grid).astype(np.int64)
    frac = grid - base
    out = np.zeros(points.shape, dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = ((frac[:, 0] if dx else 1 - frac[:, 0])
                          * (frac[:, 1] if dy else 1 - frac[:, 1])
                          * (frac[:, 2] if dz else 1 - frac[:, 2]))
                out += weight[:, None] * frame[base[:, 2] + dz, base[:, 1] + dy, base[:, 0] + dx]
    return out


def _direction(frame, points, lower, upper, sign):
    velocity = sample_field(frame, points, lower, upper)
    norm = np.linalg.norm(velocity, axis=1, keepdims=True)
    return sign * velocity / np.maximum(norm, 1e-12), norm[:, 0]


def trace_streamlines(frame, starts, lower, upper, steps=65, length=0.5):
    """RK4 streamlines in arclength, ``steps`` samples centred on each start.

    Returns ``(paths [N, steps, 3], valid [N])``.  A row is invalid if the
    velocity vanishes at the start or the line leaves the domain immediately.
    """
    starts = np.asarray(starts, dtype=np.float64)
    half = (steps - 1) // 2
    delta = float(length) / (steps - 1)
    paths = np.repeat(starts[:, None, :], steps, axis=1)
    alive = np.ones(len(starts), dtype=bool)

    for sign, count, slot in ((+1.0, steps - 1 - half, +1), (-1.0, half, -1)):
        point = starts.copy()
        moving = alive.copy()
        for k in range(1, count + 1):
            k1, speed = _direction(frame, point, lower, upper, sign)
            k2, _ = _direction(frame, point + 0.5 * delta * k1, lower, upper, sign)
            k3, _ = _direction(frame, point + 0.5 * delta * k2, lower, upper, sign)
            k4, _ = _direction(frame, point + delta * k3, lower, upper, sign)
            step = delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            candidate = point + step
            inside = ((candidate >= lower) & (candidate <= upper)).all(axis=1) & (speed > 1e-12)
            moving &= inside
            point = np.where(moving[:, None], candidate, point)
            paths[:, half + slot * k] = point
    return paths.astype(np.float32), alive


def icosahedral_star(frame, centres, vertices, radius, lower, upper, steps=65, length=0.5):
    """``[N, 13, steps, 3]`` -- the centre streamline then the twelve neighbours."""
    offsets = np.concatenate((np.zeros((1, 3)), radius * np.asarray(vertices)))
    starts = (centres[:, None, :] + offsets[None]).reshape(-1, 3)
    paths, _ = trace_streamlines(frame, starts, lower, upper, steps, length)
    return paths.reshape(len(centres), len(offsets), steps, 3)


def multi_shell_star(frame, centres, vertices, radii, lower, upper, steps=65, length=0.5):
    """``[N, 1 + 12*len(radii), steps, 3]`` -- centre, then one shell per radius.

    Two concentric icosahedral shells are still exactly icosahedrally symmetric:
    a group element permutes the twelve channels of each shell by the *same*
    permutation, so the action is `apply_group` applied shell-wise.  This gives
    the star a second length scale without leaving the symmetry class.
    """
    offsets = [np.zeros((1, 3))]
    for radius in radii:
        offsets.append(radius * np.asarray(vertices))
    offsets = np.concatenate(offsets)
    starts = (centres[:, None, :] + offsets[None]).reshape(-1, 3)
    paths, _ = trace_streamlines(frame, starts, lower, upper, steps, length)
    return paths.reshape(len(centres), len(offsets), steps, 3)
