"""Deterministic spherical neighborhoods and unchanged Task6 streamline integration."""
import hashlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
from FMT_Utils.Task6Streamlines_3D import candidate_cells
from FMT_Utils.Task6AdaptiveStreamlines_3D import integrate_samples_adaptive
from FMT_Utils.Task6Refinement_1_3 import accepted_refinement


def directions(kind):
    """Unit directions, ordered lexicographically; center is a separate zero row."""
    phi = (1 + np.sqrt(5.)) / 2
    vertices = np.array([(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)] +
                        [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)] +
                        [(b * phi, 0, a) for a in (-1, 1) for b in (-1, 1)], float)
    if kind == 'icosahedron_face_centers':
        points = vertices[ConvexHull(vertices).simplices].mean(axis=1)
        assert len(points) == 20
    elif kind == 'icosahedron_vertices':
        points = vertices
    else:
        raise ValueError('Neighbor geometry must be explicitly confirmed')
    points /= np.linalg.norm(points, axis=1)[:, None]
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    return np.vstack([np.zeros(3), points])


def frame_seed(key, base=96611):
    return int.from_bytes(hashlib.sha256(f'{base}:{key}'.encode()).digest()[:8], 'little')


class Candidate:
    """Uniform physical-volume proposals, exact trilinear IVD > 0.8 maximum."""
    def __init__(self, ivd, axes_xyz, offsets, seed, threshold_fraction=.8):
        self.axes = [np.asarray(a, float) for a in axes_xyz]
        if not 0 < threshold_fraction < 1: raise ValueError('Threshold fraction must be in (0,1)')
        self.threshold = threshold_fraction * float(np.max(ivd))
        self.interpolate = RegularGridInterpolator(tuple(self.axes[::-1]), ivd,
                                                   bounds_error=False, fill_value=-np.inf)
        self.rng = np.random.default_rng(seed)
        # A trilinear field cannot exceed its eight corner values.
        cells = candidate_cells(ivd > self.threshold)[:, ::-1]
        lower = np.array([a[0] for a in self.axes]) - np.min(offsets, axis=0)
        upper = np.array([a[-1] for a in self.axes]) - np.max(offsets, axis=0)
        self.low = np.maximum(np.column_stack([a[cells[:, j]] for j, a in enumerate(self.axes)]), lower)
        self.high = np.minimum(np.column_stack([a[cells[:, j] + 1] for j, a in enumerate(self.axes)]), upper)
        keep = np.all(self.high > self.low, axis=1)
        self.low, self.high = self.low[keep], self.high[keep]
        weights = np.prod(self.high - self.low, axis=1)
        if not len(weights) or weights.sum() <= 0:
            raise ValueError('No candidate cell can contain the requested full neighborhood')
        self.cdf = np.cumsum(weights); self.cdf /= self.cdf[-1]
        self.proposed = 0; self.candidate_accepted = 0

    def draw(self, count, proposal_limit=20000000):
        parts = []; remaining = count; start = self.proposed
        while remaining:
            n = max(2048, min(100000, remaining * 4))
            ids = np.searchsorted(self.cdf, self.rng.random(n), side='right')
            p = self.low[ids] + self.rng.random((n, 3)) * (self.high[ids] - self.low[ids])
            valid = self.interpolate(p[:, ::-1]) > self.threshold
            self.proposed += n; self.candidate_accepted += int(valid.sum())
            accepted = p[valid][:remaining]; parts.append(accepted); remaining -= len(accepted)
            if remaining and self.proposed - start >= proposal_limit:
                raise RuntimeError('Candidate rejection budget exhausted; threshold was not relaxed')
        return np.ascontiguousarray(np.concatenate(parts))


def integrate(field, centers, axes_xyz, h, unit_offsets):
    """Three frozen numerical levels; retain h/16, 65 points, length .25 each side."""
    lower = np.array([a[0] for a in axes_xyz], float)
    spacing = np.array([(float(a[-1])-float(a[0]))/(len(a)-1) for a in axes_xyz])
    for a, step in zip(axes_xyz, spacing):
        if not np.allclose(np.diff(a), step, rtol=2e-4, atol=step*2e-4):
            raise ValueError('Frozen interpolator requires a regular physical grid')
    seeds = np.ascontiguousarray((centers[:, None, :] + h * unit_offsets).reshape(-1, 3))
    coarse, _, cr = integrate_samples_adaptive(field, seeds, lower, spacing, .25, h/8, h*1e-7, 65)
    fine, lengths, fr = integrate_samples_adaptive(field, seeds, lower, spacing, .25, h/16, h*1e-8, 65)
    finest, _, rr = integrate_samples_adaptive(field, seeds, lower, spacing, .25, h/32, h*1e-9, 65)
    good, error = accepted_refinement(coarse, fine, finest, (cr, fr, rr), lengths, h)
    n, k = len(centers), len(unit_offsets)
    return dict(curves=fine.reshape(n, k, 65, 3), half_lengths=lengths.reshape(n, k, 2),
                termination=fr.reshape(n, k, 2), errors=error.reshape(n, k),
                valid=good.reshape(n, k).all(axis=1))


def integrate_filtered(field, centers, axes_xyz, h, unit_offsets, order):
    """Same accepted bundles; stop evaluating a bundle after its first invalid line."""
    n, k = len(centers), len(unit_offsets)
    if sorted(order) != list(range(k)): raise ValueError('Invalid evaluation permutation')
    curves = np.empty((n, k, 65, 3), np.float32)
    lengths = np.empty((n, k, 2), np.float32)
    reasons = np.empty((n, k, 2), np.int8)
    errors = np.empty((n, k), np.float32)
    active = np.arange(n)
    for j in order:
        if not len(active): break
        # Each offset remains attached to its original stored position.
        result = integrate(field, centers[active], axes_xyz, h, unit_offsets[j:j+1])
        curves[active, j] = result['curves'][:, 0]
        lengths[active, j] = result['half_lengths'][:, 0]
        reasons[active, j] = result['termination'][:, 0]
        errors[active, j] = result['errors'][:, 0]
        active = active[result['valid']]
    valid = np.zeros(n, bool); valid[active] = True
    return dict(curves=curves, half_lengths=lengths, termination=reasons, errors=errors, valid=valid)
