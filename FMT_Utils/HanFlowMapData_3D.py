"""Dense independent material queries and explicit evaluation roles, version 1.1."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc

from FMT_Utils.FlowMapData_3D import cross_offsets, integrate
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d

ROLES = ('fit', 'query_validation', 'query_test', 'primitive_validation',
         'primitive_test', 'time_validation', 'time_test')
TEST_ROLES = ('query_test', 'primitive_test', 'time_test')


def sobol_queries(count, seed, extent=.95):
    """Uniform-volume queries in a cross's octahedral hull, not a spherical shell."""
    power = int(np.ceil(np.log2(max(64, 16 * count))))
    points = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(power) * 2 - 1
    valid = np.abs(points).sum(1) < extent
    valid &= np.linalg.norm(points[:, None] - cross_offsets()[None], axis=-1).min(1) > 1e-5
    points = points[valid]
    if len(points) < count:
        raise ValueError('Registered Sobol candidate buffer is insufficient')
    return points[:count]


def dense_window(field, centers, radius, duration, settings, seed):
    """Retain a method-independent physical cohort; never filter by model error."""
    centers = np.asarray(centers, float)
    steps = settings['steps_per_segment']
    if steps != 62:
        raise ValueError('Frozen FMT uses 32 samples; this version requires 62 steps')
    qcount = sum(settings[k] for k in ('train_queries', 'validation_queries', 'test_queries'))
    candidates = settings['query_candidates']
    if candidates < qcount:
        raise ValueError('Need enough query candidates')
    chunks, exclusions = [], {'invalid_visible_support': 0, 'insufficient_queries': 0}
    query_counts = dict(candidate_queries=0, invalid_long_queries=0,
                        outside_second_hull=0, invalid_second_queries=0)
    minimum_gap = np.inf
    for first in range(0, len(centers), settings['integration_bundle_chunk']):
        ids = np.arange(first, min(first + settings['integration_bundle_chunk'], len(centers)))
        center = centers[ids]
        n = len(ids)
        support_seeds = center[:, None] + radius * cross_offsets()
        context_origins = center[:, None] + 3 * radius * cross_offsets()[1:]
        context_seeds = context_origins[:, :, None] + radius * cross_offsets()
        support0, v0 = integrate(field, support_seeds, 0., duration, steps)
        context, vc = integrate(field, context_seeds, 0., duration, steps)
        visible = v0.all(1) & vc.all((1, 2))
        exclusions['invalid_visible_support'] += int((~visible).sum())
        if not visible.any():
            continue
        ids, center, support0, context, context_origins = (
            x[visible] for x in (ids, center, support0, context, context_origins))
        center1 = support0[:, 0, -1]
        spread = np.linalg.norm(support0[:, :, -1] - center1[:, None], axis=-1).max(1)
        radius1 = np.maximum(radius, settings['second_radius_factor'] * spread)
        support1, vs1 = integrate(field, center1[:, None] + radius1[:, None, None] * cross_offsets(),
                                 duration, duration, steps)
        valid_support = vs1.all(1)
        exclusions['invalid_visible_support'] += int((~valid_support).sum())
        ids, center, support0, context, context_origins, center1, radius1, support1 = (
            x[valid_support] for x in (ids, center, support0, context, context_origins, center1, radius1, support1))
        if not len(ids):
            continue
        local0 = np.stack([sobol_queries(candidates, seed + int(i) * 17, settings['query_extent']) for i in ids])
        local1 = np.stack([sobol_queries(candidates, seed + int(i) * 17 + 1000003, settings['query_extent']) for i in ids])
        q0 = center[:, None] + radius * local0
        q1 = center1[:, None] + radius1[:, None, None] * local1
        long, vl = integrate(field, q0, 0., 2 * duration, 2 * steps)
        target1, vt1 = integrate(field, q1, duration, duration, steps)
        arrived = (long[:, :, steps] - center1[:, None]) / radius1[:, None, None]
        covered = np.abs(arrived).sum(-1) <= settings['query_extent']
        query_counts['candidate_queries'] += int(vl.size)
        query_counts['invalid_long_queries'] += int((~vl).sum())
        query_counts['outside_second_hull'] += int((vl & ~covered).sum())
        query_counts['invalid_second_queries'] += int((~vt1).sum())
        for j, identity in enumerate(ids):
            ix0 = np.flatnonzero(vl[j] & covered[j])[:qcount]
            ix1 = np.flatnonzero(vt1[j])[:qcount]
            if len(ix0) < qcount or len(ix1) < qcount:
                exclusions['insufficient_queries'] += 1
                continue
            visible_seeds = context_origins[j, :, None] + radius * cross_offsets()
            gap = np.linalg.norm(visible_seeds[:, :, None] - q0[j, ix0][None, None], axis=-1).min()
            if gap <= radius:
                raise ValueError('A hidden query violates the Task7 visible-input buffer')
            minimum_gap = min(minimum_gap, float(gap))
            chunks.append(dict(support0=support0[j, :, ::2], support1=support1[j, :, ::2],
                context=context[j, :, :, ::2], target0=long[j, ix0, :steps + 1],
                target1=target1[j, ix1], target_long=long[j, ix0], origin0=center[j], origin1=center1[j],
                radius0=np.asarray(radius), radius1=np.asarray(radius1[j]), duration=np.asarray(duration),
                context_origins=context_origins[j], material_bundle_ids=np.asarray(identity, np.int64),
                query_ids0=ix0.astype(np.int64), query_ids1=ix1.astype(np.int64)))
    if not chunks:
        raise ValueError(f'No complete dense-query bundles: {exclusions}')
    data = {k: np.stack([c[k] for c in chunks]) for k in chunks[0]}
    data = {k: v.astype(np.int64 if 'ids' in k else np.float32) for k, v in data.items()}
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError('Nonfinite material data')
    return data, dict(candidate_bundles=len(centers), retained_bundles=len(chunks),
        exclusions=exclusions, query_counts=query_counts, task7_minimum_seed_gap=minimum_gap,
        minimum_required_gap=radius, second_radius_uses='visible first support spread only',
        cohort_policy='source boundaries and true second-hull coverage before any model; all methods share the cohort')


def support_features(data):
    out = {}
    for name in ('support0', 'support1', 'context'):
        x = data[name]
        flat = x.reshape(-1, 7, 32, 3)
        local = flat - flat[:, :1, :1]
        radius = data['radius1'] if name == 'support1' else data['radius0']
        if name == 'context':
            radius = np.repeat(radius, 6)
        raw = (local / radius[:, None, None, None]).reshape(len(flat), -1)
        fmt = np.asarray(pathline_dft_features_3d(torch.from_numpy(local.copy())), np.float32)
        if fmt.shape != (len(flat), 161) or not np.isfinite(fmt).all():
            raise ValueError('Frozen full FMT dimension or finiteness changed')
        lead = x.shape[:-3]
        out[f'fmt_all__{name}'] = fmt.reshape(*lead, 161)
        out[f'raw_positions__{name}'] = raw.reshape(*lead, 672).astype(np.float32)
    return out


def partition_roles(data, source_role, settings):
    """Initial region assignment precedes integration; query sets never share IDs."""
    nt, nv, ne = (settings[k] for k in ('train_queries', 'validation_queries', 'test_queries'))
    fit_q, val_q, test_q = slice(0, nt), slice(nt, nt + nv), slice(nt + nv, nt + nv + ne)
    if source_role == 'train':
        fold = data['material_bundle_ids'] % 8
        groups = [('fit', fold < 5, fit_q), ('query_validation', fold < 5, val_q),
                  ('query_test', fold < 5, test_q), ('primitive_validation', fold == 5, test_q),
                  ('primitive_test', fold >= 6, test_q)]
    else:
        groups = [('time_' + source_role, np.ones(len(data['origin0']), bool), test_q)]
    result = {}
    for role, choose, queries in groups:
        if not choose.any():
            raise ValueError(f'No retained regions in registered role {role}')
        part = {k: v[choose] for k, v in data.items()}
        for name in ('target0', 'target1', 'target_long', 'query_ids0', 'query_ids1'):
            part[name] = part[name][:, queries]
        result[role] = part
    return result


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
        pa = np.linalg.norm(a[:, pair[0]] - a[:, pair[1]], axis=-1)
        pb = np.linalg.norm(b[:, pair[0]] - b[:, pair[1]], axis=-1)
        sums['pairs'] += float((np.abs(pa - pb) / rr)[:, :, 1:].sum())
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
