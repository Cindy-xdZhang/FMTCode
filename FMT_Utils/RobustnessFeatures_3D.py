"""Non-FMT baselines, fixed-width FMT ablations, and pathline corruptions.

All helpers are label-free.  They operate on the cached seven-line 3D
primitive contract ``[N, 7, L, 3]`` and never inspect IVD labels.
"""

from __future__ import annotations

import numpy as np

from FMT_Utils.DFT_FMT_3D import fmt_feature_indices_3d
from FMT_Utils.RawPathline_3D import raw_pathline_representation
from FMT_Utils.Task12Data_3D import feature_matrix


FMT_ALL_WIDTH = 161
KIN4_WIDTH = 28
FMT_KIN4_WIDTH = FMT_ALL_WIDTH + KIN4_WIDTH


def reshape_cached_primitives(raw: np.ndarray) -> np.ndarray:
    """Return a validated ``[N,7,L,3]`` view of cached raw pathlines."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 4:
        primitives = raw
    elif raw.ndim == 2 and raw.shape[1] % (7 * 3) == 0:
        primitives = raw.reshape(len(raw), 7, -1, 3)
    else:
        raise ValueError(
            "raw pathlines must be [N,7,L,3] or flattened [N,7*L*3], "
            f"got {raw.shape}"
        )
    if primitives.shape[1] != 7 or primitives.shape[2] < 2:
        raise ValueError(f"unexpected primitive shape: {primitives.shape}")
    if not np.isfinite(primitives).all():
        raise ValueError("pathline primitives contain non-finite coordinates")
    return primitives


def _center_relative_sequences(primitives: np.ndarray) -> np.ndarray:
    """Keep center motion while expressing six neighbours relative to it."""
    xyz = reshape_cached_primitives(primitives)
    center = xyz[:, :1]
    return np.concatenate((center, xyz[:, 1:] - center), axis=1)


def plain_pathline_dft_features_3d(
    raw: np.ndarray,
    num_freq: int = 6,
    *,
    mode: str = "complex",
) -> np.ndarray:
    """Ordinary coordinate-wise DFT baseline without FMT invariants.

    The input uses the same center/relative decomposition as the Raw baseline.
    ``complex`` retains real coefficients and non-DC imaginary coefficients;
    ``magnitude`` retains coefficient magnitudes.  Unlike FMT, this baseline is
    coordinate-axis dependent and has no chirality or neighbour aggregation.
    """
    sequences = _center_relative_sequences(raw)
    independent = sequences.shape[2] // 2 + 1
    num_freq = int(num_freq)
    if not 1 <= num_freq <= independent:
        raise ValueError(f"num_freq={num_freq} outside [1,{independent}]")
    spectrum = np.fft.rfft(sequences, axis=2)[:, :, :num_freq]
    if mode == "complex":
        parts = [spectrum.real]
        if num_freq > 1:
            parts.append(spectrum[:, :, 1:].imag)
        result = np.concatenate(parts, axis=2)
    elif mode == "magnitude":
        result = np.abs(spectrum)
    else:
        raise ValueError("mode must be 'complex' or 'magnitude'")
    result = result.reshape(len(sequences), -1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("ordinary DFT baseline produced non-finite values")
    return result


def pair_distance_dft_features_3d(
    raw: np.ndarray,
    num_freq: int = 6,
) -> np.ndarray:
    """Pairwise-distance temporal DFT baseline.

    It uses all 21 unordered line pairs, removes each pair's initial distance,
    and retains real plus non-DC imaginary coefficients.  This is a strong
    rigid-motion-invariant baseline, but it lacks FMT's vector Fourier Gram,
    chirality, semantic center block, and sorted-neighbour construction.
    """
    xyz = reshape_cached_primitives(raw)
    pair_index = np.asarray(
        [(left, right) for left in range(7) for right in range(left + 1, 7)],
        dtype=np.int64,
    )
    distance = np.linalg.norm(
        xyz[:, pair_index[:, 0]] - xyz[:, pair_index[:, 1]], axis=-1
    )
    distance = distance - distance[:, :, :1]
    independent = distance.shape[2] // 2 + 1
    num_freq = int(num_freq)
    if not 1 <= num_freq <= independent:
        raise ValueError(f"num_freq={num_freq} outside [1,{independent}]")
    spectrum = np.fft.rfft(distance, axis=2)[:, :, :num_freq]
    result = np.concatenate(
        (spectrum.real, spectrum[:, :, 1:].imag), axis=2
    ).reshape(len(distance), -1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("pair-distance DFT produced non-finite values")
    return result


def pathline_geometric_quantities_3d(
    raw: np.ndarray,
    *,
    relative_eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Per-line speed, curvature and signed torsion sequences.

    The unit time step between samples is absorbed into the finite
    differences, so ``speed`` is the per-step displacement norm.  Curvature is
    ``|v x a| / |v|^3`` evaluated at the ``L-2`` interior samples with ``v``
    taken as the mean of the two adjacent displacement vectors.  Torsion is
    ``((v x a) . j) / |v x a|^2`` at the ``L-3`` samples where the jerk ``j``
    exists.  Denominators receive ``relative_eps`` times the primitive's mean
    speed (cubed or squared accordingly) so stagnant lines give zero rather
    than non-finite values.  All three quantities are invariant to proper
    rigid motions; the signed torsion flips under reflections.
    """
    xyz = reshape_cached_primitives(raw).astype(np.float64)
    velocity = np.diff(xyz, axis=2)                      # [N,7,L-1,3]
    speed = np.linalg.norm(velocity, axis=-1)            # [N,7,L-1]
    acceleration = np.diff(velocity, axis=2)             # [N,7,L-2,3]
    jerk = np.diff(acceleration, axis=2)                 # [N,7,L-3,3]
    scale = speed.mean(axis=(1, 2), keepdims=True)       # [N,1,1]
    eps = float(relative_eps) * scale + 1e-300
    v_mid = 0.5 * (velocity[:, :, :-1] + velocity[:, :, 1:])          # [N,7,L-2,3]
    cross = np.cross(v_mid, acceleration)                              # [N,7,L-2,3]
    cross_norm = np.linalg.norm(cross, axis=-1)
    v_norm = np.linalg.norm(v_mid, axis=-1)
    curvature = cross_norm / (v_norm ** 3 + eps ** 3)                  # [N,7,L-2]
    cross_mid = 0.5 * (cross[:, :, :-1] + cross[:, :, 1:])             # [N,7,L-3,3]
    torsion = np.einsum("nktc,nktc->nkt", cross_mid, jerk) / (
        np.sum(cross_mid * cross_mid, axis=-1) + eps ** 4
    )                                                                  # [N,7,L-3]
    out = {
        "speed": speed.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "torsion": torsion.astype(np.float32),
    }
    for name, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f"geometric quantity {name} produced non-finite values")
    return out


GEOMETRIC_STATISTICS_PER_LINE = 14


def pathline_geometric_statistics_3d(raw: np.ndarray) -> np.ndarray:
    """Curvature, torsion and speed statistics baseline (14 values per line).

    Per line: speed mean/std/min/max, path length, net displacement,
    curvature mean/std/max, signed torsion mean/std, absolute torsion mean/max,
    and the fraction of samples with positive torsion.  With seven lines the
    representation is ``7 x 14 = 98`` wide, label-free, and invariant to
    proper rigid motions.
    """
    xyz = reshape_cached_primitives(raw).astype(np.float64)
    q = pathline_geometric_quantities_3d(raw)
    speed = q["speed"].astype(np.float64)
    curvature = q["curvature"].astype(np.float64)
    torsion = q["torsion"].astype(np.float64)
    displacement = np.linalg.norm(xyz[:, :, -1] - xyz[:, :, 0], axis=-1)
    columns = [
        speed.mean(axis=2), speed.std(axis=2), speed.min(axis=2), speed.max(axis=2),
        speed.sum(axis=2), displacement,
        curvature.mean(axis=2), curvature.std(axis=2), curvature.max(axis=2),
        torsion.mean(axis=2), torsion.std(axis=2),
        np.abs(torsion).mean(axis=2), np.abs(torsion).max(axis=2),
        (torsion > 0.0).mean(axis=2),
    ]
    assert len(columns) == GEOMETRIC_STATISTICS_PER_LINE
    result = np.stack(columns, axis=-1).reshape(len(xyz), -1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("geometric statistics produced non-finite values")
    return result


def pathline_geometric_sequences_3d(raw: np.ndarray) -> np.ndarray:
    """Full per-sample speed, curvature and torsion sequences per line.

    Curvature and torsion are squashed with ``arcsinh`` so heavy tails at
    near-straight samples do not dominate the train-only standardisation.  With
    ``L`` samples the width is ``7 x ((L-1) + (L-2) + (L-3))``.
    """
    q = pathline_geometric_quantities_3d(raw)
    parts = (
        q["speed"],
        np.arcsinh(q["curvature"]),
        np.arcsinh(q["torsion"]),
    )
    result = np.concatenate(
        [part.reshape(len(part), -1) for part in parts], axis=1
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("geometric sequences produced non-finite values")
    return result


def non_fmt_feature_matrix(
    raw: np.ndarray,
    name: str,
    *,
    num_freq: int = 6,
) -> np.ndarray:
    """Build one registered non-FMT baseline representation."""
    name = str(name)
    if name == "plain_dft_complex":
        return plain_pathline_dft_features_3d(raw, num_freq, mode="complex")
    if name == "plain_dft_magnitude":
        return plain_pathline_dft_features_3d(raw, num_freq, mode="magnitude")
    if name == "pair_distance_dft":
        return pair_distance_dft_features_3d(raw, num_freq)
    if name == "geometric_statistics":
        return pathline_geometric_statistics_3d(raw)
    if name == "geometric_sequences":
        return pathline_geometric_sequences_3d(raw)
    return raw_pathline_representation(raw, name)


def fmt_kin4_ablation_mask(name: str) -> np.ndarray:
    """Return a 189-wide information mask for the canonical FMT+kin4 input.

    Zero masking keeps the downstream VAE input width and trainable parameter
    count exactly unchanged.  The full representation is ordered as the
    cached 161-dimensional FMT block followed by the 28-dimensional kin4
    block.
    """
    name = str(name)
    mask = np.ones(FMT_KIN4_WIDTH, dtype=np.float32)
    if name == "full":
        return mask
    if name == "without_chirality":
        mask[fmt_feature_indices_3d("chirality_all")] = 0.0
    elif name == "without_cosine":
        mask[fmt_feature_indices_3d("cosine_all")] = 0.0
    elif name == "without_neighbor_fourier":
        mask[fmt_feature_indices_3d("real_imag_cosine_chirality_neighbor")] = 0.0
    elif name == "without_center_fourier":
        mask[fmt_feature_indices_3d("real_imag_cosine_chirality_center")] = 0.0
    elif name == "without_kinematic":
        mask[FMT_ALL_WIDTH:] = 0.0
    elif name == "kinematic_only":
        mask[:FMT_ALL_WIDTH] = 0.0
    else:
        raise ValueError(f"unknown FMT component ablation: {name!r}")
    return mask


def masked_fmt_kin4_feature_matrix(record: dict, name: str, device="cpu") -> np.ndarray:
    """Build the canonical full feature and remove only registered blocks."""
    full = feature_matrix(record, "fmt_all+kin4", device)
    if full.shape[1] != FMT_KIN4_WIDTH:
        raise ValueError(
            f"expected canonical FMT+kin4 width {FMT_KIN4_WIDTH}, got {full.shape}"
        )
    return np.ascontiguousarray(full * fmt_kin4_ablation_mask(name)[None, :])


def _linear_interpolation_matrix(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or len(source) < 2:
        raise ValueError("interpolation grids must be one-dimensional")
    if np.any(np.diff(source) <= 0):
        raise ValueError("source interpolation positions must increase")
    if target.min() < source[0] or target.max() > source[-1]:
        raise ValueError("target interpolation positions leave source range")
    right = np.searchsorted(source, target, side="right")
    right = np.clip(right, 1, len(source) - 1)
    left = right - 1
    denominator = source[right] - source[left]
    alpha = (target - source[left]) / denominator
    matrix = np.zeros((len(target), len(source)), dtype=np.float32)
    rows = np.arange(len(target))
    matrix[rows, left] = (1.0 - alpha).astype(np.float32)
    matrix[rows, right] += alpha.astype(np.float32)
    return matrix


def corrupt_pathline_primitives_3d(
    raw: np.ndarray,
    kind: str,
    level: float,
    *,
    spatial_scale: float,
    random_state: int,
) -> np.ndarray:
    """Apply one deterministic test-time corruption to pathline coordinates.

    ``gaussian`` uses isotropic i.i.d. coordinate noise with standard
    deviation ``level * spatial_scale``.  ``frame_dropout`` removes one common
    random set of temporal frames (endpoints retained) and linearly fills the
    original time grid.  ``short_track`` retains the leading ``level`` fraction
    of the trajectory and resamples that shorter physical window back to L.
    """
    primitives = reshape_cached_primitives(raw)
    kind = str(kind)
    level = float(level)
    if kind == "clean":
        if level != 0.0:
            raise ValueError("clean corruption requires level=0")
        return primitives.copy()
    rng = np.random.default_rng(int(random_state))
    if kind == "gaussian":
        if level < 0.0 or not np.isfinite(spatial_scale) or spatial_scale <= 0.0:
            raise ValueError("gaussian corruption requires nonnegative level and scale")
        noise = rng.normal(
            0.0, level * float(spatial_scale), size=primitives.shape
        ).astype(np.float32)
        return np.ascontiguousarray(primitives + noise)
    length = primitives.shape[2]
    original_grid = np.arange(length, dtype=np.float64)
    if kind == "frame_dropout":
        if not 0.0 <= level < 1.0:
            raise ValueError("frame_dropout level must be in [0,1)")
        drop_count = min(int(round(level * (length - 2))), length - 2)
        interior = np.arange(1, length - 1, dtype=np.int64)
        dropped = (
            np.asarray(rng.choice(interior, size=drop_count, replace=False))
            if drop_count else np.empty(0, dtype=np.int64)
        )
        keep = np.setdiff1d(np.arange(length), dropped, assume_unique=True)
        weights = _linear_interpolation_matrix(original_grid[keep], original_grid)
        return np.einsum(
            "ts,nksc->nktc", weights, primitives[:, :, keep], optimize=True
        ).astype(np.float32)
    if kind == "short_track":
        if not 0.0 < level <= 1.0:
            raise ValueError("short_track level must be in (0,1]")
        last = max(1.0, (length - 1) * level)
        target = np.linspace(0.0, last, length, dtype=np.float64)
        weights = _linear_interpolation_matrix(original_grid, target)
        return np.einsum(
            "ts,nksc->nktc", weights, primitives, optimize=True
        ).astype(np.float32)
    raise ValueError(f"unknown pathline corruption: {kind!r}")

