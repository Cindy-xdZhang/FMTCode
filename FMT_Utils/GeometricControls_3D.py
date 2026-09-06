"""Validity-checked seven-line physical geometry and paired corruptions.

This module does not modify the frozen FMT encoders. Geometric IVD here means
deviation from the sampled-seed mean, not the whole-volume label definition.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np


def array_hash(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def validate_cache(cache, *, fixed_steps=48, fixed_dt_scale=0.25):
    """Reject invalid sentinels and mismatched original-grid vs retained rows.

    Zero coordinates are legitimate; only an explicit boolean/0-1 mask and
    complete seven-line lengths establish validity. Never filter per method.
    """
    raw = np.asarray(cache['raw_features'])
    if raw.ndim != 2 or raw.shape[1] != 672 or not np.isfinite(raw).all():
        raise ValueError('cache raw must be finite [N,672]')
    n = len(raw)
    flags = np.asarray(cache['valid_mask'])
    if flags.ndim != 1 or not np.isin(flags, [False, True]).all():
        raise ValueError('valid_mask must be boolean or explicit 0/1, never -1 sentinels')
    flags = flags.astype(bool)
    lengths = np.asarray(cache['line_lengths'])
    if lengths.shape != (len(flags), 7) or int(flags.sum()) != n:
        raise ValueError('original-grid valid_mask/line_lengths disagree with retained rows')
    meta = json.loads(str(cache['metadata_json']))
    if int(meta.get('valid_primitives', n)) != n:
        raise ValueError('metadata valid count mismatch')
    steps = np.asarray(cache['integration_steps'] if 'integration_steps' in cache
                       else np.full(n, fixed_steps), dtype=np.int64)
    if steps.shape != (n,) or not np.all(lengths[flags] == steps[:, None] + 1):
        raise ValueError('retained primitive contains an incomplete line')
    dt = np.asarray(cache['physical_dt'] if 'physical_dt' in cache else
                    np.full(n, float(meta['source_time_step']) * fixed_dt_scale), dtype=np.float64)
    if dt.shape != (n,) or np.any(dt <= 0) or not np.isfinite(dt).all():
        raise ValueError('missing/invalid physical time metadata')
    times = np.rint(steps[:, None] * np.linspace(0, 1, 32)) * dt[:, None]
    if np.any(np.diff(times, axis=1) <= 0):
        raise ValueError('sample times must increase strictly')
    xyz = np.ascontiguousarray(raw.reshape(n, 7, 32, 3), dtype=np.float32)
    offsets = np.linalg.norm(xyz[:, 1:, 0] - xyz[:, :1, 0], axis=-1)
    if np.any(offsets <= 0) or not np.isfinite(offsets).all():
        raise ValueError('invalid initial neighbour cross')
    pairs = np.stack([xyz[:, 1, 0]-xyz[:, 2, 0], xyz[:, 3, 0]-xyz[:, 4, 0],
                      xyz[:, 5, 0]-xyz[:, 6, 0]], axis=-1).astype(np.float64)
    singular = np.linalg.svd(pairs, compute_uv=False)
    if np.any(singular[:, -1] <= 1e-6 * singular[:, 0]):
        raise ValueError('singular initial neighbour cross')
    seeds = np.asarray(cache['seeds'])
    if seeds.shape != (n, 3) or not np.isfinite(seeds).all():
        raise ValueError('seed identity mismatch')
    return {
        'raw': xyz, 'times': times, 'offset': np.median(offsets, axis=1),
        'scale_id': np.asarray(cache['scale_id'] if 'scale_id' in cache else np.zeros(n), dtype=np.int64),
        'labels': np.asarray(cache['reference'], dtype=np.float32),
        'metadata': meta, 'cached_fmt': np.asarray(cache['fmt_features'], dtype=np.float32),
        'identity': array_hash(np.flatnonzero(flags)) + ':' + array_hash(seeds),
        'certificate': {'total': len(flags), 'valid': n, 'invalid': int((~flags).sum()),
                        'mask_sha256': array_hash(flags), 'lengths_sha256': array_hash(lengths),
                        'raw_sha256': array_hash(raw), 'times_sha256': array_hash(times)},
    }


def geometric_sequences(raw, times, scale_ids):
    """Ddot @ pinv(D), with physical nonuniform time differences.

    All rows share the same physical seed time. Later samples are grouped by
    scale before subtracting means because mixed-scale sample times differ.
    Returns IVD-like, strain norm, absolute divergence and Q-like sequences.
    """
    x = np.asarray(raw, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    d = np.stack((x[:, 1]-x[:, 2], x[:, 3]-x[:, 4], x[:, 5]-x[:, 6]), axis=-1)
    derivative = np.empty_like(d)
    derivative[:, 0] = (d[:, 1]-d[:, 0]) / (t[:, 1]-t[:, 0])[:, None, None]
    derivative[:, -1] = (d[:, -1]-d[:, -2]) / (t[:, -1]-t[:, -2])[:, None, None]
    # Three-point nonuniform central derivative, unlike a uniform-index FFT.
    a, b = t[:, 1:-1]-t[:, :-2], t[:, 2:]-t[:, 1:-1]
    derivative[:, 1:-1] = (
        (-b/(a*(a+b)))[..., None, None]*d[:, :-2]
        + ((b-a)/(a*b))[..., None, None]*d[:, 1:-1]
        + (a/(b*(a+b)))[..., None, None]*d[:, 2:])
    gradient = derivative @ np.linalg.pinv(d, rcond=1e-6)
    omega = np.stack((gradient[..., 2, 1]-gradient[..., 1, 2],
                      gradient[..., 0, 2]-gradient[..., 2, 0],
                      gradient[..., 1, 0]-gradient[..., 0, 1]), axis=-1)
    deviation = np.empty_like(omega)
    for group in np.unique(scale_ids):
        mask = np.asarray(scale_ids) == group
        deviation[mask] = omega[mask] - omega[mask].mean(axis=0, keepdims=True)
    deviation[:, 0] = omega[:, 0] - omega[:, 0].mean(axis=0, keepdims=True)
    ivd = np.linalg.norm(deviation, axis=-1)
    strain = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
    strain_norm = np.linalg.norm(strain, axis=(-2, -1))
    divergence = np.abs(np.trace(gradient, axis1=-2, axis2=-1))
    result = np.stack((ivd, strain_norm, divergence, 0.25*ivd**2-0.5*strain_norm**2), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite geometric features; do not drop rows per method')
    return result.astype(np.float32)


def perturb(record, condition, repeat_seed):
    """One shared coordinate array and physical time grid for every arm.

    Dropout keeps endpoints, interpolating in physical time. Truncation uses
    only the retained prefix (no interpolation from discarded future points).
    No corrupted-row filtering or post-noise re-centering is performed.
    """
    raw, times = record['raw'], record['times']
    kind, level = condition['kind'], float(condition['level'])
    changed, new_times = raw.copy(), times.copy()
    rng = np.random.default_rng(stable_seed(record.get('context', ''), record['identity'], condition['id'], repeat_seed))
    if kind == 'clean':
        if level != 0:
            raise ValueError('clean level must be zero')
    elif kind == 'gaussian':
        changed += (rng.normal(size=raw.shape) * level * record['offset'][:, None, None, None]).astype(np.float32)
    elif kind in {'frame_dropout', 'short_track'}:
        for i in range(len(raw)):
            if kind == 'frame_dropout':
                dropped = rng.choice(np.arange(1, 31), size=round(level*30), replace=False)
                keep = np.setdiff1d(np.arange(32), dropped)
                target = times[i]
            else:
                count = max(3, int(np.floor(31*level))+1)
                keep = np.arange(count)
                target = np.linspace(times[i, 0], times[i, count-1], 32)
                new_times[i] = target
            for line in range(7):
                for channel in range(3):
                    changed[i, line, :, channel] = np.interp(
                        target, times[i, keep], raw[i, line, keep, channel])
    else:
        raise ValueError(kind)
    return changed, new_times


def representations(record, task, raw=None, times=None, device='cpu'):
    """Frozen task-specific FMT plus predeclared physical/Fourier controls."""
    import torch
    from FMT_Utils.DFT_FMT_3D import (
        pathline_dft_features_3d, pathline_velocity_gradient_dft_features_3d,
        pathline_anchored_kinematic_dft_features_3d, time_local_gram_dft_features_3d,
    )
    from FMT_Utils.RobustnessFeatures_3D import plain_pathline_dft_features_3d
    raw = record['raw'] if raw is None else raw
    times = record['times'] if times is None else times
    geometry = geometric_sequences(raw, times, record['scale_id'])
    tensor = torch.from_numpy(raw).to(device)
    if task == 'Task3':
        full = pathline_anchored_kinematic_dft_features_3d(
            tensor, num_freq=1, window=3, channels=(0,), anchor_names=(), include_dft=True)
    else:
        # Recompute the frozen recipe on the same raw coordinates as all arms.
        # Clean caches have backend-dependent roundoff; do not overwrite them.
        spectral = pathline_dft_features_3d(tensor, num_freq=6, neighbor_weight=1.0,
            neighbor_scale=1.0, neighbor_pool='sort', mode='gram', include_chirality=True)
        kinetic = pathline_velocity_gradient_dft_features_3d(tensor, num_freq=4 if task == 'Task1' else 6)
        parts = [spectral]
        if task == 'Task5':
            parts.append(time_local_gram_dft_features_3d(tensor, num_freq=2))
        full = np.concatenate([*parts, kinetic], axis=1)
    result = {
        'geometry_ivd': geometry[:, 0, :1],
        'plain_dft': plain_pathline_dft_features_3d(raw, 6, mode='complex'),
        'plain_magnitude': plain_pathline_dft_features_3d(raw, 6, mode='magnitude'),
        'geometry_no_fourier': geometry.reshape(len(raw), -1),
        'fmt': full,
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{name} produced nonfinite features')
    return result


def pad_auxiliary(values, width=268):
    """Equal nominal network capacity without discarding any input feature."""
    value = np.asarray(values, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] > width:
        raise ValueError(f'cannot losslessly pad {value.shape} to {width}')
    return np.pad(value, ((0, 0), (0, width-value.shape[1])))


def stressed_seed_ivd(record, *, lag=1, smooth_window=1, rcond=1e-6,
                      mean_weight=1.0, feature_scale=1.0):
    """Explicitly misconfigured seed-time geometry for diagnostic sweeps.

    Defaults reproduce the original seed-time scalar. Nondefaults are NOT
    replacements for that baseline. Times and row validity remain unchanged.
    Smoothing uses an edge-padded moving mean of pair-separation matrices.
    A cutoff above one intentionally discards every singular direction.
    """
    lag, window = int(lag), int(smooth_window)
    if not 1 <= lag < 32 or window not in range(1, 32, 2):
        raise ValueError('lag must be 1..31 and smoothing an odd width 1..31')
    if not np.isfinite([rcond, mean_weight, feature_scale]).all() or rcond < 0:
        raise ValueError('invalid diagnostic geometry parameters')
    x = np.asarray(record['raw'], dtype=np.float64)
    times = record['times']
    d = np.stack((x[:, 1]-x[:, 2], x[:, 3]-x[:, 4], x[:, 5]-x[:, 6]), axis=-1)
    if window > 1:
        half = window // 2
        padded = np.pad(d, ((0, 0), (half, half), (0, 0), (0, 0)), mode='edge')
        d = sum(padded[:, j:j+32] for j in range(window)) / window
    derivative = (d[:, lag]-d[:, 0]) / (times[:, lag]-times[:, 0])[:, None, None]
    gradient = derivative @ np.linalg.pinv(d[:, 0], rcond=float(rcond))
    omega = np.stack((gradient[:, 2, 1]-gradient[:, 1, 2],
                      gradient[:, 0, 2]-gradient[:, 2, 0],
                      gradient[:, 1, 0]-gradient[:, 0, 1]), axis=-1)
    result = float(feature_scale)*np.linalg.norm(
        omega-float(mean_weight)*omega.mean(axis=0, keepdims=True), axis=1)
    if not np.isfinite(result).all():
        raise ValueError('diagnostic geometry is nonfinite; no row removal permitted')
    return result.astype(np.float32)[:, None]
