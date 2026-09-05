"""Proxy-label utilities for the channel-flow Task4-b experiment.

The four-class protocol deliberately excludes non-vortex voxels from the
multiclass target:

``0`` ordinary streamwise, ``1`` ordinary spanwise/perpendicular,
``2`` hairpin head, and ``3`` hairpin limb (legs plus necks).

The manual ``VortexIds`` field marks hairpin support and instance identity but
does not contain anatomical part labels.  Head/limb labels are therefore a
deterministic physics proxy rather than manual ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    normalized_mutual_info_score,
)

from FMT_Utils.Task4A_TraditionalBaseline_3D import (
    velocity_curl_orientation_classes,
)
from FMT_Utils.Task4A_StreamlineClustering_3D import (
    SPANWISE_CLASS,
    STREAMWISE_CLASS,
)


IGNORE_LABEL = -1
ORDINARY_STREAMWISE = 0
ORDINARY_SPANWISE = 1
HAIRPIN_HEAD = 2
HAIRPIN_LIMB = 3
CLASS_IDS = (
    ORDINARY_STREAMWISE,
    ORDINARY_SPANWISE,
    HAIRPIN_HEAD,
    HAIRPIN_LIMB,
)
CLASS_NAMES = (
    "ordinary_streamwise",
    "ordinary_spanwise",
    "hairpin_head",
    "hairpin_limb",
)


@dataclass(frozen=True)
class IVDSelection:
    """Frozen result of one recall-constrained IVD threshold search."""

    mode: str
    percentile: float
    threshold: float
    train_hairpin_recall: float
    train_vortex_equal_recall: float
    candidate_fraction: float


def vorticity_deviation_volume(
    vorticity_zyx3: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a vorticity-deviation norm and the subtracted mean field.

    ``whole_domain`` is standard instantaneous vorticity deviation (IVD): one
    spatial mean vector is subtracted from the full volume.  Channel flow is
    inhomogeneous in wall-normal ``z``; ``wall_normal_plane`` instead subtracts
    the x-y plane mean at every z and is explicitly named a channel-conditioned
    proxy rather than standard IVD.
    """

    omega = np.asarray(vorticity_zyx3, dtype=np.float64)
    if omega.ndim != 4 or omega.shape[-1] != 3:
        raise ValueError("vorticity must have shape [Z,Y,X,3]")
    if not np.isfinite(omega).all():
        raise ValueError("vorticity contains non-finite values")
    mode = str(mode)
    if mode == "whole_domain":
        mean = omega.mean(axis=(0, 1, 2), keepdims=True)
    elif mode == "wall_normal_plane":
        mean = omega.mean(axis=(1, 2), keepdims=True)
    else:
        raise ValueError(
            "mode must be 'whole_domain' or 'wall_normal_plane'"
        )
    deviation = np.linalg.norm(omega - mean, axis=-1).astype(np.float32)
    return deviation, mean.astype(np.float32)


def pad_periodic_xy(volume_zyx: np.ndarray) -> np.ndarray:
    """Append repeated x/y seam planes for channel interpolation."""

    volume = np.asarray(volume_zyx)
    if volume.ndim < 3:
        raise ValueError("volume must have at least [Z,Y,X] dimensions")
    padded = np.concatenate((volume, volume[:, :, :1]), axis=2)
    return np.concatenate((padded, padded[:, :1]), axis=1)


def voxel_center_axes(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical center coordinates in x, y, z order."""

    axes = []
    for axis, count in enumerate(grid.resolution_xyz):
        axes.append(
            grid.domain_min_xyz[axis]
            + (np.arange(int(count), dtype=np.float64) + 0.5)
            * grid.voxel_size_xyz[axis]
        )
    return tuple(axes)


def sample_scalar_on_voxel_grid(
    axes_zyx: tuple[np.ndarray, np.ndarray, np.ndarray],
    scalar_zyx: np.ndarray,
    grid,
    *,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Interpolate one scalar flow volume at every independent cube center."""

    scalar = np.asarray(scalar_zyx)
    expected = tuple(len(axis) for axis in axes_zyx)
    if scalar.shape != expected:
        raise ValueError(
            f"scalar shape {scalar.shape} does not match axes {expected}"
        )
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    interpolator = RegularGridInterpolator(
        axes_zyx, scalar, bounds_error=False, fill_value=np.nan
    )
    xs, ys, zs = voxel_center_axes(grid)
    nz, ny, nx = grid.shape_zyx
    output = np.empty(nz * ny * nx, dtype=np.float32)
    for start in range(0, len(output), chunk_size):
        stop = min(len(output), start + chunk_size)
        flat = np.arange(start, stop, dtype=np.int64)
        iz, remainder = np.divmod(flat, ny * nx)
        iy, ix = np.divmod(remainder, nx)
        query_zyx = np.column_stack((zs[iz], ys[iy], xs[ix]))
        output[start:stop] = interpolator(query_zyx).astype(np.float32)
    return output.reshape(grid.shape_zyx)


def split_vortex_ids(
    vortex_ids: np.ndarray,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 7068,
) -> dict[str, np.ndarray]:
    """Split whole positive VortexIds without inspecting proxy part labels."""

    unique = np.unique(np.asarray(vortex_ids, dtype=np.int32))
    unique = unique[unique > 0]
    if len(unique) < 5:
        raise ValueError("at least five positive VortexIds are required")
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0,1)")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test split")
    shuffled = np.random.default_rng(int(seed)).permutation(unique)
    train_count = max(1, int(round(train_fraction * len(shuffled))))
    validation_count = max(1, int(round(validation_fraction * len(shuffled))))
    if train_count + validation_count >= len(shuffled):
        validation_count = len(shuffled) - train_count - 1
    return {
        "train": np.sort(shuffled[:train_count]),
        "validation": np.sort(
            shuffled[train_count : train_count + validation_count]
        ),
        "test": np.sort(shuffled[train_count + validation_count :]),
    }


def split_vortex_ids_by_buffered_x(
    vortex_ids_zyx: np.ndarray,
    x_centers: np.ndarray,
    *,
    x_intervals: dict[str, Iterable[float]],
    period_x: float,
    domain_min_x: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Assign complete VortexIds only when their full x support fits one split.

    IDs touching an excluded buffer are deliberately omitted.  This lets the
    same buffered x slabs govern both annotated hairpins and ordinary cubes.
    """

    ids = np.asarray(vortex_ids_zyx, dtype=np.int32)
    xs = np.asarray(x_centers, dtype=np.float64).reshape(-1)
    if ids.ndim != 3 or ids.shape[2] != len(xs):
        raise ValueError("x centers must match the x axis of VortexIds")
    period_x = float(period_x)
    if not np.isfinite(period_x) or period_x <= 0.0:
        raise ValueError("period_x must be positive and finite")
    fractions = np.mod(xs - float(domain_min_x), period_x) / period_x
    positive = ids > 0
    _, _, positive_ix = np.nonzero(positive)
    positive_ids = ids[positive]
    unique = np.unique(positive_ids)
    splits: dict[str, list[int]] = {name: [] for name in x_intervals}
    excluded: list[int] = []
    parsed: dict[str, tuple[float, float]] = {}
    for name, bounds in x_intervals.items():
        lower, upper = (float(value) for value in bounds)
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(f"invalid x interval for {name}: {bounds}")
        parsed[str(name)] = (lower, upper)
    for vortex_id in unique:
        member_x = fractions[positive_ix[positive_ids == vortex_id]]
        matches = [
            name
            for name, (lower, upper) in parsed.items()
            if float(np.min(member_x)) >= lower
            and float(np.max(member_x)) < upper
        ]
        if len(matches) == 1:
            splits[matches[0]].append(int(vortex_id))
        elif not matches:
            excluded.append(int(vortex_id))
        else:
            raise ValueError(f"VortexId {int(vortex_id)} fits multiple x intervals")
    arrays = {
        name: np.asarray(sorted(values), dtype=np.int32)
        for name, values in splits.items()
    }
    if any(len(values) == 0 for values in arrays.values()):
        raise ValueError(f"every split requires at least one complete VortexId: {arrays}")
    return arrays, np.asarray(sorted(excluded), dtype=np.int32)


def scan_ivd_thresholds(
    reference_volume: np.ndarray,
    sampled_ivd_zyx: np.ndarray,
    vortex_ids_zyx: np.ndarray,
    train_vortex_ids: Iterable[int],
    percentiles: Iterable[float],
) -> list[dict]:
    """Scan thresholds using only training-instance hairpin recall.

    Manual hairpin support is a subset of all vortices, so voxels outside the
    annotation are not counted as false positives.  Candidate fraction and
    Dice-to-annotation are descriptive compactness diagnostics only.
    """

    reference = np.asarray(reference_volume, dtype=np.float64)
    sampled = np.asarray(sampled_ivd_zyx, dtype=np.float64)
    ids = np.asarray(vortex_ids_zyx, dtype=np.int32)
    if sampled.shape != ids.shape or sampled.ndim != 3:
        raise ValueError("sampled IVD and VortexIds must be same-shape 3D")
    finite = np.isfinite(sampled)
    train_mask = np.isin(ids, np.asarray(list(train_vortex_ids), dtype=np.int32))
    if not np.any(train_mask):
        raise ValueError("training VortexIds select no annotated voxels")
    all_hairpin = ids > 0
    train_ids = np.unique(ids[train_mask])
    rows: list[dict] = []
    for percentile in percentiles:
        percentile = float(percentile)
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("IVD percentiles must lie in [0,100]")
        threshold = float(np.percentile(reference, percentile))
        candidate = finite & (sampled >= threshold)
        train_instance_recalls = np.asarray(
            [float(np.mean(candidate[ids == vortex_id])) for vortex_id in train_ids],
            dtype=np.float64,
        )
        intersection = int(np.count_nonzero(candidate & all_hairpin))
        denominator = int(np.count_nonzero(candidate)) + int(
            np.count_nonzero(all_hairpin)
        )
        rows.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "train_hairpin_recall": float(
                    np.mean(candidate[train_mask])
                ),
                "train_vortex_equal_recall": float(
                    np.mean(train_instance_recalls)
                ),
                "train_min_vortex_recall": float(
                    np.min(train_instance_recalls)
                ),
                "all_hairpin_recall_descriptive": float(
                    np.mean(candidate[all_hairpin])
                ),
                "candidate_count": int(np.count_nonzero(candidate)),
                "candidate_fraction": float(np.mean(candidate[finite])),
                "annotated_support_fraction_not_precision": float(
                    intersection / max(1, np.count_nonzero(candidate))
                ),
                "dice_to_hairpin_annotation_descriptive": float(
                    2.0 * intersection / max(1, denominator)
                ),
            }
        )
    return rows


def select_recall_constrained_ivd(
    rows_by_mode: dict[str, list[dict]],
    *,
    minimum_train_hairpin_recall: float,
    minimum_train_vortex_equal_recall: float | None = None,
) -> tuple[IVDSelection, list[dict]]:
    """Choose the most compact candidate mask meeting frozen train recall."""

    target = float(minimum_train_hairpin_recall)
    if not 0.0 < target <= 1.0:
        raise ValueError("minimum_train_hairpin_recall must lie in (0,1]")
    macro_target = (
        target
        if minimum_train_vortex_equal_recall is None
        else float(minimum_train_vortex_equal_recall)
    )
    if not 0.0 < macro_target <= 1.0:
        raise ValueError(
            "minimum_train_vortex_equal_recall must lie in (0,1]"
        )
    eligible: list[dict] = []
    annotated: list[dict] = []
    for mode, rows in rows_by_mode.items():
        for row in rows:
            item = {"mode": str(mode), **row}
            item["meets_recall_constraint"] = bool(
                item["train_hairpin_recall"] >= target
                and item["train_vortex_equal_recall"] >= macro_target
            )
            annotated.append(item)
            if item["meets_recall_constraint"]:
                eligible.append(item)
    if not eligible:
        best = max(
            annotated,
            key=lambda item: (
                item["train_hairpin_recall"],
                item["train_vortex_equal_recall"],
                -item["candidate_fraction"],
                item["percentile"],
            ),
        )
        raise RuntimeError(
            "no IVD threshold met the requested training hairpin recall; "
            f"best={best}"
        )
    chosen = min(
        eligible,
        key=lambda item: (
            item["candidate_fraction"],
            -item["percentile"],
            -item["train_hairpin_recall"],
            item["mode"],
        ),
    )
    for item in annotated:
        item["selected"] = bool(
            item["mode"] == chosen["mode"]
            and item["percentile"] == chosen["percentile"]
        )
    return (
        IVDSelection(
            mode=chosen["mode"],
            percentile=float(chosen["percentile"]),
            threshold=float(chosen["threshold"]),
            train_hairpin_recall=float(chosen["train_hairpin_recall"]),
            train_vortex_equal_recall=float(
                chosen["train_vortex_equal_recall"]
            ),
            candidate_fraction=float(chosen["candidate_fraction"]),
        ),
        annotated,
    )


def build_four_vortex_type_labels(
    velocity_xyz: np.ndarray,
    vorticity_xyz: np.ndarray,
    omega_y_prime: np.ndarray,
    vortex_ids: np.ndarray,
    *,
    streamwise_max_angle_degrees: float = 45.0,
    minimum_vector_norm: float = 1e-9,
) -> dict:
    """Assign four proxy vortex types to an already filtered candidate set.

    Outside manual hairpins, the frozen velocity--curl proxy names ordinary
    streamwise versus spanwise/perpendicular vortices.  Inside hairpins, a head
    must be perpendicular, have positive spanwise vorticity fluctuation, and
    have that component dominate x and z.  All remaining annotated voxels are
    limb, deliberately folding vertical necks into the leg class requested by
    the four-class taxonomy.
    """

    velocity = np.asarray(velocity_xyz, dtype=np.float64)
    omega = np.asarray(vorticity_xyz, dtype=np.float64)
    omega_y_prime = np.asarray(omega_y_prime, dtype=np.float64).reshape(-1)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    if velocity.shape != omega.shape or velocity.shape != (len(ids), 3):
        raise ValueError("velocity/vorticity must be [N,3] and match IDs")
    if len(omega_y_prime) != len(ids):
        raise ValueError("omega_y_prime must match candidate count")
    orientation, angles, valid = velocity_curl_orientation_classes(
        velocity,
        omega,
        streamwise_max_angle_degrees=float(streamwise_max_angle_degrees),
        minimum_vector_norm=float(minimum_vector_norm),
    )
    labels = np.full(len(ids), IGNORE_LABEL, dtype=np.int8)
    ordinary = (ids <= 0) & valid
    labels[ordinary & (orientation == STREAMWISE_CLASS)] = ORDINARY_STREAMWISE
    labels[ordinary & (orientation == SPANWISE_CLASS)] = ORDINARY_SPANWISE

    hairpin = (ids > 0) & valid
    head = (
        hairpin
        & (orientation == SPANWISE_CLASS)
        & (omega_y_prime > 0.0)
        & (np.abs(omega_y_prime) >= np.abs(omega[:, 0]))
        & (np.abs(omega_y_prime) >= np.abs(omega[:, 2]))
    )
    labels[hairpin] = HAIRPIN_LIMB
    labels[head] = HAIRPIN_HEAD
    return {
        "labels": labels,
        "orientation_classes": orientation,
        "orientation_angle_degrees": angles,
        "valid_orientation": valid,
        "head_rule_mask": head,
    }


def _unwrap_periodic(values: np.ndarray, period: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or period <= 0.0:
        return values.copy()
    angle = 2.0 * np.pi * values / period
    center = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
    center_value = period * center / (2.0 * np.pi)
    return center_value + np.mod(values - center_value + 0.5 * period, period) - 0.5 * period


def hairpin_physics_quality_rows(
    seeds_xyz: np.ndarray,
    vorticity_xyz: np.ndarray,
    omega_y_prime: np.ndarray,
    vortex_ids: np.ndarray,
    labels: np.ndarray,
    *,
    periods_xy: tuple[float, float],
) -> list[dict]:
    """Report, but do not optimize against, the requested hairpin constraints."""

    seeds = np.asarray(seeds_xyz, dtype=np.float64)
    omega = np.asarray(vorticity_xyz, dtype=np.float64)
    omega_y_prime = np.asarray(omega_y_prime, dtype=np.float64).reshape(-1)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if seeds.shape != omega.shape or seeds.shape != (len(ids), 3):
        raise ValueError("seeds/vorticity must be [N,3] and match IDs")
    if not (len(omega_y_prime) == len(labels) == len(ids)):
        raise ValueError("physics-quality arrays must have equal length")
    rows: list[dict] = []
    for vortex_id in np.unique(ids[ids > 0]):
        member = ids == vortex_id
        local_labels = labels[member]
        local_seed = seeds[member].copy()
        local_omega = omega[member]
        local_prime = omega_y_prime[member]
        local_seed[:, 0] = _unwrap_periodic(local_seed[:, 0], periods_xy[0])
        local_seed[:, 1] = _unwrap_periodic(local_seed[:, 1], periods_xy[1])
        head = local_labels == HAIRPIN_HEAD
        limb = local_labels == HAIRPIN_LIMB
        leg = limb & (np.abs(local_omega[:, 0]) >= np.abs(local_omega[:, 2]))
        neck = limb & ~leg
        # Looking downstream along +x, this dataset's left side is positive
        # periodic y displacement from the head center.  The convention is
        # recorded explicitly because a camera reversal swaps these names.
        center_source = head if np.any(head) else limb
        y_center = (
            float(np.mean(local_seed[center_source, 1]))
            if np.any(center_source)
            else np.nan
        )
        relative_y = local_seed[:, 1] - y_center
        left_leg = leg & (relative_y > 0.0)
        right_leg = leg & (relative_y < 0.0)
        left_neck = neck & (relative_y > 0.0)
        right_neck = neck & (relative_y < 0.0)

        def mean_at(mask: np.ndarray, column: int) -> float | None:
            return float(np.mean(local_seed[mask, column])) if np.any(mask) else None

        def median_omega(mask: np.ndarray, column: int) -> float | None:
            return float(np.median(local_omega[mask, column])) if np.any(mask) else None

        hx, hz = mean_at(head, 0), mean_at(head, 2)
        nx, nz = mean_at(neck, 0), mean_at(neck, 2)
        lx, lz = mean_at(leg, 0), mean_at(leg, 2)
        position_evaluable = all(
            value is not None for value in (hx, hz, nx, nz, lx, lz)
        )
        left_leg_wx = median_omega(left_leg, 0)
        right_leg_wx = median_omega(right_leg, 0)
        left_neck_wz = median_omega(left_neck, 2)
        right_neck_wz = median_omega(right_neck, 2)
        sign_evaluable = all(
            value is not None
            for value in (left_leg_wx, right_leg_wx, left_neck_wz, right_neck_wz)
        )
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "voxel_count": int(np.count_nonzero(member)),
                "head_count": int(np.count_nonzero(head)),
                "limb_count": int(np.count_nonzero(limb)),
                "leg_proxy_count": int(np.count_nonzero(leg)),
                "neck_proxy_count": int(np.count_nonzero(neck)),
                "head_fraction": float(np.mean(head)),
                "head_all_positive_omega_y_prime": bool(
                    np.all(local_prime[head] > 0.0)
                ) if np.any(head) else False,
                "head_exists": bool(np.any(head)),
                "position_order_evaluable": bool(position_evaluable),
                "head_x": hx,
                "neck_x": nx,
                "leg_x": lx,
                "head_z": hz,
                "neck_z": nz,
                "leg_z": lz,
                "position_order_hx_gt_nx_gt_lx": bool(
                    position_evaluable and hx > nx > lx
                ),
                "position_order_hz_gt_nz_gt_lz": bool(
                    position_evaluable and hz > nz > lz
                ),
                "full_xz_position_order_success": bool(
                    position_evaluable and hx > nx > lx and hz > nz > lz
                ),
                "side_signs_evaluable": bool(sign_evaluable),
                "left_leg_median_omega_x": left_leg_wx,
                "right_leg_median_omega_x": right_leg_wx,
                "left_neck_median_omega_z": left_neck_wz,
                "right_neck_median_omega_z": right_neck_wz,
                "left_leg_omega_x_negative": bool(
                    left_leg_wx is not None and left_leg_wx < 0.0
                ),
                "right_leg_omega_x_positive": bool(
                    right_leg_wx is not None and right_leg_wx > 0.0
                ),
                "left_neck_omega_z_negative": bool(
                    left_neck_wz is not None and left_neck_wz < 0.0
                ),
                "right_neck_omega_z_positive": bool(
                    right_neck_wz is not None and right_neck_wz > 0.0
                ),
                "left_right_sign_rule_success": bool(
                    sign_evaluable
                    and left_leg_wx < 0.0
                    and right_leg_wx > 0.0
                    and left_neck_wz < 0.0
                    and right_neck_wz > 0.0
                ),
            }
        )
    return rows


def summarize_hairpin_physics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("hairpin physics summary requires rows")

    def rate(key: str, eligible_key: str | None = None) -> dict:
        eligible = rows if eligible_key is None else [r for r in rows if r[eligible_key]]
        return {
            "success_count": int(sum(bool(r[key]) for r in eligible)),
            "eligible_count": int(len(eligible)),
            "rate": float(np.mean([bool(r[key]) for r in eligible]))
            if eligible
            else None,
        }

    return {
        "vortex_id_count": len(rows),
        "head_exists": rate("head_exists"),
        "head_positive_omega_y_prime": rate(
            "head_all_positive_omega_y_prime", "head_exists"
        ),
        "x_position_order": rate(
            "position_order_hx_gt_nx_gt_lx", "position_order_evaluable"
        ),
        "z_position_order": rate(
            "position_order_hz_gt_nz_gt_lz", "position_order_evaluable"
        ),
        "full_xz_position_order": rate(
            "full_xz_position_order_success", "position_order_evaluable"
        ),
        "left_leg_omega_x_negative": rate(
            "left_leg_omega_x_negative", "side_signs_evaluable"
        ),
        "right_leg_omega_x_positive": rate(
            "right_leg_omega_x_positive", "side_signs_evaluable"
        ),
        "left_neck_omega_z_negative": rate(
            "left_neck_omega_z_negative", "side_signs_evaluable"
        ),
        "right_neck_omega_z_positive": rate(
            "right_neck_omega_z_positive", "side_signs_evaluable"
        ),
        "left_right_sign_rule": rate(
            "left_right_sign_rule_success", "side_signs_evaluable"
        ),
    }


def best_cluster_permutation(
    true_labels: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    class_ids: tuple[int, ...] = CLASS_IDS,
    cluster_ids_universe: tuple[int, ...] | None = None,
) -> tuple[dict[int, int], float]:
    """Find the validation macro-F1 mapping for arbitrary cluster IDs."""

    truth = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    if len(truth) != len(clusters):
        raise ValueError("true labels and clusters must have equal length")
    observed_clusters = tuple(sorted(int(value) for value in np.unique(clusters)))
    unique_clusters = (
        observed_clusters
        if cluster_ids_universe is None
        else tuple(sorted(int(value) for value in cluster_ids_universe))
    )
    if not set(observed_clusters).issubset(unique_clusters):
        raise ValueError("observed cluster lies outside cluster_ids_universe")
    if len(unique_clusters) != len(class_ids):
        raise ValueError("cluster and class counts must match for permutation")
    best_mapping: dict[int, int] | None = None
    best_score = -np.inf
    for semantic in permutations(class_ids):
        mapping = dict(zip(unique_clusters, semantic))
        predicted = np.asarray([mapping[int(value)] for value in clusters])
        score = float(
            f1_score(truth, predicted, labels=class_ids, average="macro", zero_division=0)
        )
        if score > best_score + 1e-15:
            best_mapping, best_score = mapping, score
    assert best_mapping is not None
    return best_mapping, best_score


def apply_cluster_mapping(
    cluster_ids: np.ndarray, mapping: dict[int, int]
) -> np.ndarray:
    clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    missing = sorted(set(int(value) for value in np.unique(clusters)) - set(mapping))
    if missing:
        raise ValueError(f"cluster mapping misses IDs {missing}")
    return np.asarray([mapping[int(value)] for value in clusters], dtype=np.int8)


def four_class_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    class_ids: tuple[int, ...] = CLASS_IDS,
) -> dict:
    """Return class-balanced four-class segmentation metrics."""

    truth = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
    if len(truth) != len(predicted) or len(truth) == 0:
        raise ValueError("non-empty true/predicted arrays must have equal length")
    per_f1 = f1_score(
        truth, predicted, labels=class_ids, average=None, zero_division=0
    )
    per_iou = jaccard_score(
        truth, predicted, labels=class_ids, average=None, zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=class_ids)
    return {
        "sample_count": int(len(truth)),
        "macro_f1": float(np.mean(per_f1)),
        "macro_iou": float(np.mean(per_iou)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(truth, predicted)
        ),
        "per_class_f1": {
            CLASS_NAMES[index]: float(per_f1[index])
            for index in range(len(class_ids))
        },
        "per_class_iou": {
            CLASS_NAMES[index]: float(per_iou[index])
            for index in range(len(class_ids))
        },
        "support": {
            CLASS_NAMES[index]: int(np.count_nonzero(truth == label))
            for index, label in enumerate(class_ids)
        },
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
    }
