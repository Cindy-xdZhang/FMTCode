"""Material-point data and physical evaluation for Tasks 6, 7 and 8, v1.1."""
from __future__ import annotations

from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit, prange
from FLowUtils.flowlineIntegral import _interp4_quadrilinear


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def cross_offsets():
    return np.concatenate([np.zeros((1, 3)), np.eye(3).repeat(2, axis=0)
                           * np.tile([1, -1], 3)[:, None]])


def material_offsets():
    """Union of the original seven-line cross and the 24 material shell seeds."""
    corners = np.array(list(product((-1., 1.), repeat=3))) / np.sqrt(3.)
    return np.concatenate([cross_offsets()] + [corners * r for r in (1., 1.5, 2.)])


def query_offsets():
    # Eight different seeds inside the cross convex hull; first four for volume
    # are selected as 0,1,2,4 (noncoplanar), never inferred from a feature.
    return np.array(list(product((-1., 1.), repeat=3))) * .22


def randomized_queries(count, seed):
    """Different interior query coordinates per bundle, independent of velocity."""
    rng = np.random.default_rng(seed)
    rotation, upper = np.linalg.qr(rng.normal(size=(count, 3, 3)))
    rotation *= np.sign(np.diagonal(upper, axis1=1, axis2=2))[:, None, :]
    rotation[:, :, 0] *= np.linalg.det(rotation)[:, None]
    return np.einsum("nij,qj->nqi", rotation, query_offsets()) * rng.uniform(.8, 1.1, (count, 1, 1))


def select_windows(times, available, count=8, frames=9, minimum_time=None):
    """Use real contiguous source frames; no split shares an interpolation frame."""
    t = np.asarray(times, dtype=float)
    if len(t) < frames or not np.all(np.diff(t) > 0):
        raise ValueError("Need increasing physical time coordinates")
    available = set(int(i) for i in available)
    candidates, i = [], 0
    while i + frames <= len(t):
        subset = t[i:i + frames]
        okay = all(j in available for j in range(i, i + frames))
        okay &= minimum_time is None or subset[0] >= minimum_time - 1e-7
        okay &= np.allclose(np.diff(subset), np.diff(subset)[0], rtol=1e-4, atol=1e-8)
        if okay:
            candidates.append(i)
            i += frames
        else:
            i += 1
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} disjoint valid windows; need {count}")
    chosen = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int)
    return [candidates[i] for i in chosen]


@njit(cache=True)
def _inside(p, lo, hi):
    return np.isfinite(p).all() and (p >= lo).all() and (p <= hi).all()


@njit(parallel=True, cache=True)
def _rk4(field, lo, hi, spacing, seeds, start, duration, steps, field_tmin, field_dt):
    """Fixed-clock RK4; reject boundary crossings at every internal stage."""
    out = np.full((len(seeds), steps + 1, 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(seeds), dtype=np.bool_)
    dt = duration / steps
    nt, nz, ny, nx, _ = field.shape
    for j in prange(len(seeds)):
        p = seeds[j].copy()
        if not _inside(p, lo, hi):
            continue
        out[j, 0] = p
        valid[j] = True
        for k in range(steps):
            time = start + dt * k
            a = np.array(_interp4_quadrilinear(field, lo, spacing, nx, ny, nz,
                field_tmin, field_dt, nt, p[0], p[1], p[2], time))
            p2 = p + .5 * dt * a
            if not _inside(p2, lo, hi):
                valid[j] = False
                break
            b = np.array(_interp4_quadrilinear(field, lo, spacing, nx, ny, nz,
                field_tmin, field_dt, nt, p2[0], p2[1], p2[2], time + dt / 2))
            p3 = p + .5 * dt * b
            if not _inside(p3, lo, hi):
                valid[j] = False
                break
            c = np.array(_interp4_quadrilinear(field, lo, spacing, nx, ny, nz,
                field_tmin, field_dt, nt, p3[0], p3[1], p3[2], time + dt / 2))
            p4 = p + dt * c
            if not _inside(p4, lo, hi):
                valid[j] = False
                break
            d = np.array(_interp4_quadrilinear(field, lo, spacing, nx, ny, nz,
                field_tmin, field_dt, nt, p4[0], p4[1], p4[2], time + dt))
            p = p + dt / 6 * (a + 2 * b + 2 * c + d)
            if not _inside(p, lo, hi):
                valid[j] = False
                break
            out[j, k + 1] = p
    return out, valid


def integrate(field, seeds, start, duration, steps, chunk=4096):
    shape = np.asarray(seeds).shape[:-1]
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    if start < field.tmin - 1e-8 or start + duration > field.tmax + 1e-7 or steps < 1:
        raise ValueError("Integration interval outside observed source window")
    out, flags = [], []
    for i in range(0, len(seeds), chunk):
        x, v = _rk4(np.asarray(field.field), np.asarray(field.domainMinBoundary, float),
                    np.asarray(field.domainMaxBoundary, float), np.asarray(field.gridInterval, float),
                    seeds[i:i + chunk], float(start), float(duration), int(steps),
                    float(field.tmin), float(field.timeInterval))
        out.append(x)
        flags.append(v)
    return np.concatenate(out).reshape(*shape, steps + 1, 3), np.concatenate(flags).reshape(shape)


def build_window(field, centers, radius, duration, steps=62, query_seed=9106):
    """Build common cohorts before feature extraction. No labels are read.

    Task6: two independent short-map support/query pairs.
    Task7: six outside supports; the center support is NOT an input.
    Task8: true long trajectories, with stage-two support placed using only
    the *visible support center* endpoint and visible support spread.
    """
    centers = np.asarray(centers, float)
    n = len(centers)
    offsets = material_offsets()
    seeds0 = centers[:, None] + radius * offsets
    dense0 = centers[:, None] + radius * randomized_queries(n, query_seed)
    support0, v0 = integrate(field, seeds0, 0., duration, steps)
    long_target, vl = integrate(field, dense0, 0., 2 * duration, 2 * steps)
    context_centers = centers[:, None] + 3. * radius * cross_offsets()[1:]
    context_seeds = context_centers[:, :, None] + radius * offsets
    context, vc = integrate(field, context_seeds, 0., duration, steps)

    # Missing hidden points cannot reappear among a context's neighbors.
    minimum_gap = np.linalg.norm(context_seeds[:, :, :, None] - dense0[:, None, None], axis=-1).min()
    if minimum_gap <= radius:
        raise ValueError("Task7 input violates the hidden-region buffer")
    base_valid = v0.all(1) & vl.all(1) & vc.all((1, 2))
    idx = np.flatnonzero(base_valid)
    if not len(idx):
        raise ValueError("No complete common material bundles in this window")
    support0, long_target, context = support0[idx], long_target[idx], context[idx]
    center1 = support0[:, 0, -1]
    radius1 = np.maximum(radius, np.linalg.norm(support0[:, :7, -1] - center1[:, None], axis=-1).max(1) * 1.25)
    support1, v1 = integrate(field, center1[:, None] + radius1[:, None, None] * offsets,
                            duration, duration, steps)
    target1, vt1 = integrate(field, center1[:, None] + radius1[:, None, None] * randomized_queries(len(idx), query_seed + 10000),
                            duration, duration, steps)
    # Queries must fall inside the stage-two cross's convex hull at composition.
    arrival = (long_target[:, :, steps] - center1[:, None]) / radius1[:, None, None]
    covered = (np.abs(arrival).sum(-1) <= 1.).all(1)
    valid = v1.all(1) & vt1.all(1) & covered
    selected = idx[valid]
    if not len(selected):
        raise ValueError("No complete, covered second-stage supports")
    take = lambda x: np.asarray(x[valid], np.float32)
    result = dict(
        support0=take(support0[:, :, ::2]), support1=take(support1[:, :, ::2]),
        context=take(context[:, :, :, ::2]), target0=take(long_target[:, :, :steps + 1]),
        target1=take(target1), target_long=take(long_target),
        origin0=centers[selected].astype(np.float32), origin1=take(center1),
        radius0=np.full(len(selected), radius, np.float32), radius1=take(radius1),
        context_origins=context_centers[selected].astype(np.float32),
        duration=np.full(len(selected), duration, np.float32), material_bundle_ids=selected,
    )
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite retained material data")
    audit = dict(candidate_bundles=n, retained_bundles=len(selected), query_seed=query_seed,
                 excluded_first_stage=int((~base_valid).sum()),
                 excluded_second_stage=int((~(v1.all(1) & vt1.all(1))).sum()),
                 excluded_composition_coverage=int((~covered).sum()),
                 task7_minimum_seed_gap=float(minimum_gap), minimum_required_gap=float(radius),
                 support_samples=steps // 2 + 1, query_samples=steps + 1,
                 composition_samples=2 * steps + 1)
    return result, audit


def affine_query(support, origins, radii, queries, tau):
    """Untrained local-affine interpolation from original seven material paths."""
    x = (np.asarray(support[:, :7], float) - origins[:, None, None]) / radii[:, None, None, None]
    design = np.concatenate([np.ones((len(x), 7, 1)), x[:, :, 0]], axis=-1)
    weights = np.linalg.pinv(design)
    coeff = np.einsum("nck,nktd->nctd", weights, x)
    loc = np.asarray(tau) * (x.shape[2] - 1)
    lo = np.floor(loc).astype(int)
    hi = np.minimum(lo + 1, x.shape[2] - 1)
    frac = loc - lo
    values = coeff[:, :, lo] * (1 - frac)[None, None, :, None] + coeff[:, :, hi] * frac[None, None, :, None]
    query = np.concatenate([np.ones((*queries.shape[:2], 1)), queries], axis=-1)
    return np.einsum("nqc,nctd->nqtd", query, values) * radii[:, None, None, None] + origins[:, None, None]


def trajectory_metrics(pred, truth, radius, input_stride=2):
    """Evaluate corresponding particles/times; no feature-space self-scoring."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    radius = np.asarray(radius, float)
    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError("Prediction must be finite and preserve material/time identities")
    d = (pred - truth) / radius[:, None, None, None]
    e = np.linalg.norm(d, axis=-1)
    pairs = np.triu_indices(truth.shape[1], 1)
    def distances(x):
        return np.linalg.norm(x[:, pairs[0]] - x[:, pairs[1]], axis=-1) / radius[:, None, None]
    pe = np.abs(distances(pred) - distances(truth))
    def volume(x):
        a = x[:, [1, 2, 4]] - x[:, :1]
        return np.abs(np.linalg.det(a.transpose(0, 2, 1, 3))) / (6 * radius[:, None] ** 3)
    true_vol = volume(truth)
    ve = np.abs(volume(pred) - true_vol)
    pcenter, tcenter = pred.mean(1, keepdims=True), truth.mean(1, keepdims=True)
    shape_e = np.linalg.norm(((pred - pcenter) - (truth - tcenter)) / radius[:, None, None, None], axis=-1)
    # Equal weight per query time, excluding the exact supplied initial state.
    return dict(position_nrmse=float(np.sqrt(np.mean(np.sum(d[:, :, 1:] ** 2, -1)))),
                mean_position_error=float(e[:, :, 1:].mean()), endpoint_error=float(e[:, :, -1].mean()),
                late_half_error=float(e[:, :, e.shape[2] // 2:].mean()),
                between_sample_error=float(e[:, :, 1::input_stride].mean()),
                pair_distance_error=float(pe[:, :, 1:].mean()),
                centered_shape_error=float(shape_e[:, :, 1:].mean()),
                tetra_volume_error=float(ve[:, 1:].mean()),
                time_position_error=e.mean((0, 1)).tolist(), time_pair_distance_error=pe.mean((0, 1)).tolist(),
                bundle_position_error=e[:, :, 1:].mean((1, 2)).tolist(),
                bundles=len(pred), particles=int(pred.shape[1]), times=int(pred.shape[2]),
                normalization="initial primitive radius; equal particle/time weights, initial state excluded")
