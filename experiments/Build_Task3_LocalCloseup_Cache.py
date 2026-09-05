"""Build prediction-ready Task3 caches on an IVD-selected local seed grid.

The global dense evaluation remains unchanged.  This cache is visualization
only: it places a fresh regular seed grid inside a crop selected exclusively
from frozen global IVD labels, integrates real pathline primitives there, and
feeds the frozen Task3 checkpoints without interpolation or duplicated marks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
import torch

from Build_Task2_Universality_Cache import (
    _raw_local_features,
    resolve_primitive_offset,
)
from Build_Channel_Killing_Cache import load_channel_vtk
from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_3D_pipeline import (
    compute_ivd_reference_3d,
    integrate_cross_primitives_3d,
)
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from FMT_Utils.KillingObserver3D import (
    compose_steady_to_unsteady,
    integrate_killing_frame,
    smooth_channel_observer,
)
from Visualize_Task3_F22Anchored import _vortex_roi


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expand_bounds_to_mixed_labels(
    roi, domain_bounds, seeds, reference,
    expansion_factors=(1.0, 1.5, 2.25, 3.375, 5.0, 8.0),
):
    """Expand one fixed IVD crop only as far as needed to show its boundary."""
    roi = np.asarray(roi, dtype=np.float64)
    domain_bounds = np.asarray(domain_bounds, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.float64)
    reference = np.asarray(reference, dtype=bool)
    center = roi.mean(axis=0)
    base_span = roi[1] - roi[0]
    domain_span = domain_bounds[1] - domain_bounds[0]
    for factor in expansion_factors:
        span = np.minimum(base_span * float(factor), domain_span)
        lower = np.clip(
            center - 0.5 * span,
            domain_bounds[0],
            domain_bounds[1] - span,
        )
        candidate = np.stack((lower, lower + span))
        visible = np.all(
            (seeds >= candidate[0]) & (seeds <= candidate[1]), axis=1
        )
        labels = reference[visible]
        if np.any(labels) and np.any(~labels):
            return candidate, float(factor)
    raise RuntimeError(
        "could not form a local IVD crop containing both label classes"
    )


def _dataset_entry(config, dataset):
    entries = [item for item in config.datasets if str(item["id"]) == dataset]
    if len(entries) != 1:
        raise RuntimeError(f"{dataset}: expected one source entry, found {len(entries)}")
    path = Path(entries[0]["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return entries[0], path


def _global_ivd_crop(dataset, visual_spec):
    local = visual_spec["local_sampling"]
    prediction_path = (
        ROOT / local["roi_source_prediction_root"]
        / f"{dataset}_task3_dense5x_predictions.npz"
    )
    audit_suffix = str(local.get(
        "roi_source_audit_suffix", "task3_diagnostic"
    ))
    audit_path = (
        ROOT / local["roi_source_audit_root"]
        / f"{dataset}_{audit_suffix}.json"
    )
    if not prediction_path.exists() or not audit_path.exists():
        raise FileNotFoundError(
            f"{dataset}: missing frozen global prediction or audit"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if _sha256(prediction_path) != audit["prediction_artifact_sha256"]:
        raise RuntimeError(f"{dataset}: frozen global prediction hash changed")
    with np.load(prediction_path) as source:
        seeds = np.asarray(source["seeds"], dtype=np.float64)
        reference = np.asarray(source["reference"], dtype=bool)
        source_metadata = json.loads(str(source["metadata_json"]))
    bounds = np.asarray(audit["full_bounds"], dtype=np.float64)
    full_span = np.maximum(bounds[1] - bounds[0], 1e-12)
    roi_focus = str(local.get("roi_focus", "ivd_dense_positive"))
    if roi_focus == "frozen_audit_bounds":
        bounds_key = str(local.get("roi_source_bounds_key", "close_up_bounds"))
        if bounds_key not in audit:
            raise KeyError(
                f"{dataset}: source audit does not contain {bounds_key!r}"
            )
        roi = np.asarray(audit[bounds_key], dtype=np.float64)
        if roi.shape != (2, 3) or np.any(roi[0] >= roi[1]):
            raise ValueError(
                f"{dataset}: {bounds_key} must have shape (2,3) and "
                "positive spans"
            )
        if np.any(roi[0] < bounds[0]) or np.any(roi[1] > bounds[1]):
            raise ValueError(
                f"{dataset}: frozen audit bounds exceed the physical domain"
            )
        return roi, bounds, prediction_path, audit_path, source_metadata
    side = (
        float(local["physical_cube_longest_axis_fraction"])
        * float(np.max(full_span))
    )
    loaded_shape = np.asarray(
        source_metadata.get("loaded_shape_TZYXC", []), dtype=np.int64
    )
    if loaded_shape.shape == (5,) and np.all(loaded_shape[1:4] >= 2):
        # Stored vector fields use [T,Z,Y,X,C].  A close-up smaller than the
        # source grid cannot reveal additional flow structure; it only makes
        # the evaluated seeds look like a point lattice.  Keep every local
        # crop physically resolvable before generating the denser Sobol set.
        shape_xyz = loaded_shape[[3, 2, 1]].astype(np.float64)
        source_spacing = full_span / (shape_xyz - 1.0)
        side = max(
            side,
            float(local.get("minimum_source_voxels", 4.0))
            * float(np.max(source_spacing)),
        )
    if roi_focus == "ivd_boundary":
        positive_point, negative_point = _ivd_boundary_pair(
            seeds, reference, bounds
        )
        focus = 0.5 * (positive_point + negative_point)
        side = max(
            side,
            1.05 * float(np.max(np.abs(positive_point - negative_point))),
        )
        target_span_fraction = np.minimum(side / full_span, 1.0)
        span = target_span_fraction * full_span
        lower = np.clip(focus - 0.5 * span, bounds[0], bounds[1] - span)
        roi = np.stack((lower, lower + span))
    elif roi_focus in {"ivd_dense_positive", "ivd_dense_boundary"}:
        target_span_fraction = np.minimum(side / full_span, 1.0)
        dense_roi = _vortex_roi(
            bounds,
            seeds,
            reference,
            positive_fraction=float(local["positive_fraction"]),
            minimum_points=int(local["minimum_points"]),
            padding_factor=1.0,
            minimum_span_fraction=float(local["minimum_span_fraction"]),
            target_span_fraction=target_span_fraction,
            exact_target_span=True,
            center_on_selected_neighbourhood=True,
        )
        if roi_focus == "ivd_dense_boundary":
            # Keep the component selected above, but move the small physical
            # box to the nearest IVD boundary of that component.  The global
            # seed grid only chooses the midpoint; its coarse point spacing
            # must not enlarge the final local field of view.
            all_positive_indices = np.flatnonzero(reference)
            if not len(all_positive_indices):
                raise RuntimeError(f"{dataset}: no global IVD-positive seed")
            dense_center = 0.5 * (dense_roi[0] + dense_roi[1])
            center_distance = np.linalg.norm(
                (seeds[all_positive_indices] - dense_center) / full_span,
                axis=1,
            )
            candidate_count = min(256, len(all_positive_indices))
            nearest_to_component = np.argpartition(
                center_distance, candidate_count - 1
            )[:candidate_count]
            positive_indices = all_positive_indices[nearest_to_component]
            negative_points = seeds[~reference]
            nearest_distance, nearest_negative = cKDTree(
                negative_points
            ).query(seeds[positive_indices], k=1)
            pair_midpoints = 0.5 * (
                seeds[positive_indices]
                + negative_points[np.asarray(nearest_negative, dtype=np.int64)]
            )
            normalized_center_distance = np.linalg.norm(
                (pair_midpoints - dense_center) / full_span,
                axis=1,
            )
            order = np.lexsort((normalized_center_distance, nearest_distance))
            focus = pair_midpoints[int(order[0])]
            span = target_span_fraction * full_span
            lower = np.clip(
                focus - 0.5 * span,
                bounds[0],
                bounds[1] - span,
            )
            roi = np.stack((lower, lower + span))
        else:
            roi = dense_roi
    else:
        raise ValueError(
            "local_sampling.roi_focus must be ivd_boundary, "
            "ivd_dense_positive, ivd_dense_boundary, or "
            "frozen_audit_bounds"
        )
    if (
        roi_focus == "ivd_dense_positive"
        and not np.any(
            reference
            & np.all((seeds >= roi[0]) & (seeds <= roi[1]), axis=1)
        )
    ):
        raise RuntimeError(f"{dataset}: IVD-selected crop contains no positive seed")
    if roi_focus == "ivd_dense_positive" and bool(
        local.get("require_mixed_global_labels", False)
    ):
        roi, _ = _expand_bounds_to_mixed_labels(
            roi, bounds, seeds, reference
        )
    return roi, bounds, prediction_path, audit_path, source_metadata


def _ivd_boundary_pair(seeds, reference, bounds):
    """Select the nearest physical pair with opposite global IVD labels."""
    seeds = np.asarray(seeds, dtype=np.float64)
    reference = np.asarray(reference, dtype=bool)
    bounds = np.asarray(bounds, dtype=np.float64)
    if not np.any(reference) or np.all(reference):
        raise RuntimeError("IVD boundary focus requires both label classes")
    positive_points = seeds[reference]
    negative_points = seeds[~reference]
    nearest_distance, nearest_negative_index = cKDTree(
        negative_points
    ).query(positive_points, k=1)
    normalized = (positive_points - bounds[0]) / np.maximum(
        bounds[1] - bounds[0], 1e-12
    )
    positive_centroid = normalized.mean(axis=0)
    centroid_distance = np.linalg.norm(
        normalized - positive_centroid, axis=1
    )
    # Physical opposite-label distance is authoritative because the displayed
    # crop is an equal-sided physical cube.  The centroid term resolves ties.
    order = np.lexsort((centroid_distance, nearest_distance))
    positive_index = int(order[0])
    negative_index = int(nearest_negative_index[positive_index])
    return positive_points[positive_index], negative_points[negative_index]


def _ivd_boundary_focus(seeds, reference, bounds):
    """Return the midpoint of a prediction-independent IVD boundary pair."""
    positive_point, negative_point = _ivd_boundary_pair(
        seeds, reference, bounds
    )
    return 0.5 * (positive_point + negative_point)


def _local_grid(bounds, grid_shape, offset):
    bounds = np.asarray(bounds, dtype=np.float64)
    shape = np.asarray(grid_shape, dtype=np.int64)
    if shape.shape != (3,) or np.any(shape < 2):
        raise ValueError("local_sampling.seed_grid_shape must contain three values >=2")
    margin = float(offset) * 1.01
    low = bounds[0] + margin
    high = bounds[1] - margin
    if np.any(low >= high):
        raise ValueError(
            "local close-up is too small for the seven-line primitive offset"
        )
    axes = [np.linspace(low[i], high[i], int(shape[i])) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    seeds = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1)
    return np.ascontiguousarray(seeds, dtype=np.float64)


def _scaled_local_bounds(bounds, scale):
    """Shrink a local sampling box about its center without moving the focus."""
    bounds = np.asarray(bounds, dtype=np.float64)
    scale = float(scale)
    if bounds.shape != (2, 3) or np.any(bounds[0] >= bounds[1]):
        raise ValueError("bounds must have shape (2,3) with positive spans")
    if not 0.0 < scale <= 1.0:
        raise ValueError("local_sampling.sampling_box_scale must be in (0,1]")
    center = 0.5 * (bounds[0] + bounds[1])
    half_span = 0.5 * scale * (bounds[1] - bounds[0])
    return np.stack((center - half_span, center + half_span))


def _local_sobol_points(bounds, count, seed, offset, domain_bounds):
    """Generate deterministic non-lattice seeds inside a physical close-up.

    Primitive neighbours may leave the displayed crop, but must remain inside
    the full flow domain. The crop is a camera window, not an integration wall.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    domain_bounds = np.asarray(domain_bounds, dtype=np.float64)
    count = int(count)
    if count < 2:
        raise ValueError("local_sampling.seed_count must be at least 2")
    if domain_bounds.shape != (2, 3):
        raise ValueError("domain_bounds must have shape (2,3)")
    margin = 1.01 * float(offset)
    low = np.maximum(bounds[0], domain_bounds[0] + margin)
    high = np.minimum(bounds[1], domain_bounds[1] - margin)
    if np.any(low >= high):
        raise ValueError(
            "local close-up cannot keep primitive neighbours inside the domain"
        )
    sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
    power = int(np.ceil(np.log2(count)))
    unit = sampler.random_base2(power)[:count]
    seeds = qmc.scale(unit, low, high)
    return np.ascontiguousarray(seeds, dtype=np.float64)


def _strong_ivd_boundary_index(ivd_volume, threshold, spacing_xyz):
    """Return a high-gradient voxel lying nearest the requested IVD level."""
    ivd = np.asarray(ivd_volume, dtype=np.float64)
    finite = np.isfinite(ivd)
    if ivd.ndim != 3 or not np.any(finite):
        raise ValueError("ivd_volume must be a finite three-dimensional array")
    residual = np.abs(ivd - float(threshold))
    cutoff = float(np.quantile(residual[finite], 0.01))
    dx, dy, dz = (float(value) for value in spacing_xyz)
    grad_z, grad_y, grad_x = np.gradient(ivd, dz, dy, dx, edge_order=1)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    candidate = finite & (residual <= cutoff)
    if min(ivd.shape) > 2:
        interior = np.zeros(ivd.shape, dtype=bool)
        interior[1:-1, 1:-1, 1:-1] = True
        if np.any(candidate & interior):
            candidate &= interior
    score = np.where(candidate, gradient, -np.inf)
    if not np.isfinite(score).any():
        raise RuntimeError("could not locate an IVD threshold boundary voxel")
    return np.asarray(np.unravel_index(np.nanargmax(score), ivd.shape))


def _strong_ivd_level_crossing(ivd_volume, threshold, axes_xyz):
    """Locate the strongest sub-voxel edge crossing of an IVD level."""
    ivd = np.asarray(ivd_volume, dtype=np.float64)
    xs, ys, zs = (np.asarray(axis, dtype=np.float64) for axis in axes_xyz)
    if ivd.shape != (len(zs), len(ys), len(xs)):
        raise ValueError("IVD volume and physical axes have inconsistent shapes")
    axes_zyx = (zs, ys, xs)
    candidates = []
    for volume_axis, coordinates in enumerate(axes_zyx):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[volume_axis] = slice(0, -1)
        second_slice[volume_axis] = slice(1, None)
        first = ivd[tuple(first_slice)]
        second = ivd[tuple(second_slice)]
        residual_first = first - float(threshold)
        residual_second = second - float(threshold)
        crossing = (
            np.isfinite(first)
            & np.isfinite(second)
            & (residual_first * residual_second <= 0.0)
            & (first != second)
        )
        # Prefer an interior crossing so every seven-line primitive remains
        # comfortably inside the physical flow domain.
        index_grid = np.indices(first.shape)
        interior = np.ones(first.shape, dtype=bool)
        for index_axis, size in enumerate(ivd.shape):
            upper = size - 2 if index_axis == volume_axis else size - 1
            interior &= (index_grid[index_axis] >= 1)
            interior &= (index_grid[index_axis] < upper)
        crossing &= interior
        if not np.any(crossing):
            continue
        spacing = np.diff(coordinates)
        edge_index = index_grid[volume_axis]
        score = np.where(
            crossing,
            np.abs(second - first) / spacing[edge_index],
            -np.inf,
        )
        flat_index = int(np.nanargmax(score))
        index = np.asarray(np.unravel_index(flat_index, score.shape))
        candidates.append((float(score[tuple(index)]), volume_axis, index))
    if not candidates:
        raise RuntimeError("could not locate an interior IVD level crossing")
    _, volume_axis, index = max(candidates, key=lambda item: item[0])
    first_index = index.copy()
    second_index = index.copy()
    second_index[volume_axis] += 1
    first_value = float(ivd[tuple(first_index)])
    second_value = float(ivd[tuple(second_index)])
    fraction = np.clip(
        (float(threshold) - first_value) / (second_value - first_value),
        0.0,
        1.0,
    )
    first_point = np.asarray([
        xs[first_index[2]], ys[first_index[1]], zs[first_index[0]]
    ])
    second_point = np.asarray([
        xs[second_index[2]], ys[second_index[1]], zs[second_index[0]]
    ])
    return first_point + float(fraction) * (second_point - first_point)


def _nearest_ivd_level_crossing(
    ivd_volume, threshold, axes_xyz, target_point,
):
    """Locate the sub-voxel IVD crossing nearest one frozen component.

    The global IVD labels identify the vortex component.  At strong close-up
    scales their coarse positive/negative midpoint need not lie on the
    continuous p95 level.  Search all three edge directions and choose the
    exact linear edge crossing nearest that component instead.
    """
    ivd = np.asarray(ivd_volume, dtype=np.float64)
    xs, ys, zs = (np.asarray(axis, dtype=np.float64) for axis in axes_xyz)
    target = np.asarray(target_point, dtype=np.float64)
    if ivd.shape != (len(zs), len(ys), len(xs)):
        raise ValueError("IVD volume and physical axes have inconsistent shapes")
    if target.shape != (3,):
        raise ValueError("target_point must contain x, y, z")
    coordinates_zyx = (zs, ys, xs)
    domain_span_xyz = np.maximum(
        np.asarray([np.ptp(xs), np.ptp(ys), np.ptp(zs)]), 1e-12
    )
    candidates = []
    for volume_axis, coordinates in enumerate(coordinates_zyx):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[volume_axis] = slice(0, -1)
        second_slice[volume_axis] = slice(1, None)
        first = ivd[tuple(first_slice)]
        second = ivd[tuple(second_slice)]
        residual_first = first - float(threshold)
        residual_second = second - float(threshold)
        crossing = (
            np.isfinite(first)
            & np.isfinite(second)
            & (residual_first * residual_second <= 0.0)
            & (first != second)
        )
        indices = np.argwhere(crossing)
        if not len(indices):
            continue
        first_values = first[tuple(indices.T)]
        second_values = second[tuple(indices.T)]
        fractions = np.clip(
            (float(threshold) - first_values)
            / (second_values - first_values),
            0.0,
            1.0,
        )
        points_zyx = np.column_stack([
            coordinates_zyx[axis][indices[:, axis]]
            for axis in range(3)
        ]).astype(np.float64)
        edge_indices = indices[:, volume_axis]
        points_zyx[:, volume_axis] = (
            coordinates[edge_indices]
            + fractions * (
                coordinates[edge_indices + 1] - coordinates[edge_indices]
            )
        )
        points_xyz = points_zyx[:, [2, 1, 0]]
        distance = np.linalg.norm(
            (points_xyz - target) / domain_span_xyz, axis=1
        )
        edge_length = np.maximum(
            coordinates[edge_indices + 1] - coordinates[edge_indices],
            1e-12,
        )
        gradient = np.abs(second_values - first_values) / edge_length
        order = np.lexsort((-gradient, distance))
        index = int(order[0])
        candidates.append((
            float(distance[index]), -float(gradient[index]), points_xyz[index]
        ))
    if not candidates:
        raise RuntimeError("could not locate an IVD level crossing")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _ivd_volume_boundary_crop_near(
    field, local_spec, target_point, expected_threshold,
):
    """Center a resolvable source box on the selected component boundary."""
    empty = np.empty((0, 3), dtype=np.float64)
    ivd_volume, _, (xs, ys, zs) = compute_ivd_reference_3d(
        field, 0.0, empty
    )
    threshold = float(np.percentile(
        ivd_volume[np.isfinite(ivd_volume)],
        float(local_spec["ivd_percentile"]),
    ))
    if not np.isclose(
        threshold, float(expected_threshold), rtol=1e-7,
        atol=1e-7 * max(1.0, abs(float(expected_threshold))),
    ):
        raise RuntimeError(
            "frozen and recomputed IVD thresholds differ "
            f"({expected_threshold} vs {threshold})"
        )
    focus = _nearest_ivd_level_crossing(
        ivd_volume, threshold, (xs, ys, zs), target_point
    )
    bounds = np.stack((
        np.asarray(field.domainMinBoundary, dtype=np.float64),
        np.asarray(field.domainMaxBoundary, dtype=np.float64),
    ))
    full_span = np.maximum(bounds[1] - bounds[0], 1e-12)
    side = (
        float(local_spec["physical_cube_longest_axis_fraction"])
        * float(np.max(full_span))
    )
    minimum_source_voxels = float(
        local_spec.get("minimum_source_voxels", 4.0)
    )
    side = max(
        side,
        minimum_source_voxels * float(np.max(field.gridInterval)),
    )
    side = min(side, float(np.min(full_span)))
    span = np.repeat(side, 3)
    lower = np.clip(focus - 0.5 * span, bounds[0], bounds[1] - span)
    return np.stack((lower, lower + span)), threshold


def _ivd_volume_boundary_crop(field, local_spec):
    empty = np.empty((0, 3), dtype=np.float64)
    ivd_volume, _, (xs, ys, zs) = compute_ivd_reference_3d(
        field, 0.0, empty
    )
    threshold = float(np.percentile(
        ivd_volume[np.isfinite(ivd_volume)],
        float(local_spec["ivd_percentile"]),
    ))
    try:
        focus = _strong_ivd_level_crossing(
            ivd_volume, threshold, (xs, ys, zs)
        )
    except RuntimeError:
        # Degenerate synthetic/test fields can have no strict interior edge
        # crossing. Preserve the older nearest-threshold fallback for them;
        # production audits record nonzero clipped surface triangles.
        iz, iy, ix = _strong_ivd_boundary_index(
            ivd_volume, threshold, field.gridInterval
        )
        focus = np.asarray([xs[ix], ys[iy], zs[iz]], dtype=np.float64)
    bounds = np.stack((
        np.asarray(field.domainMinBoundary, dtype=np.float64),
        np.asarray(field.domainMaxBoundary, dtype=np.float64),
    ))
    full_span = np.maximum(bounds[1] - bounds[0], 1e-12)
    side = (
        float(local_spec["physical_cube_longest_axis_fraction"])
        * float(np.max(full_span))
    )
    minimum_source_voxels = float(
        local_spec.get("minimum_source_voxels", 4.0)
    )
    if minimum_source_voxels < 2.0:
        raise ValueError(
            "local_sampling.minimum_source_voxels must be at least 2.0"
        )
    side = max(
        side,
        minimum_source_voxels * float(np.max(field.gridInterval)),
    )
    side = min(side, float(np.min(full_span)))
    span = np.repeat(side, 3)
    lower = np.clip(focus - 0.5 * span, bounds[0], bounds[1] - span)
    return np.stack((lower, lower + span)), threshold


def _load_channel_window(config, path, source_index):
    interpolator, points, axes, source_min, source_max, vtk_metadata = (
        load_channel_vtk(
            path,
            int(config.sampling.max_spatial_dim),
            float(config.channel_observer.output_crop_fraction),
        )
    )
    total_frames = int(config.channel_observer.total_frames)
    duration = float(config.channel_observer.duration)
    times = np.linspace(0.0, duration, total_frames)
    dt_source = float(times[1] - times[0])
    parameters = smooth_channel_observer(
        times / duration, source_min, source_max
    )
    parameters /= duration
    rotation, displacement = integrate_killing_frame(parameters, dt_source)
    future_intervals = int(np.ceil(
        float(config.pathlines.dt_scale)
        * int(config.pathlines.integration_steps)
    ))
    frame_count = future_intervals + 2
    if not 0 <= int(source_index) <= total_frames - frame_count:
        raise IndexError("channel source index cannot supply the future window")
    selection = slice(int(source_index), int(source_index) + frame_count)
    flat_field = compose_steady_to_unsteady(
        points,
        interpolator,
        parameters[selection],
        rotation[selection],
        displacement[selection],
        bounds_min=source_min,
        bounds_max=source_max,
    )
    ox, oy, oz = axes
    shape_zyx = (len(oz), len(oy), len(ox))
    field_data = flat_field.reshape(frame_count, *shape_zyx, 3)
    dmin = np.array([ox[0], oy[0], oz[0]], dtype=np.float32)
    dmax = np.array([ox[-1], oy[-1], oz[-1]], dtype=np.float32)
    field = UnsteadyVectorField3D(
        len(ox), len(oy), len(oz), frame_count,
        dmin, dmax, 0.0, dt_source * (frame_count - 1),
    )
    field.field = np.ascontiguousarray(field_data, dtype=np.float32)
    metadata = {
        "source_start_index": int(source_index),
        "source_time": float(times[int(source_index)]),
        "source_time_step": dt_source,
        "frame_count": frame_count,
        "loaded_shape_TZYXC": list(field_data.shape),
        "channel_vtk": vtk_metadata,
    }
    return field, metadata


def build_local_closeup_cache(config, dataset, visual_spec, overwrite=False):
    """Build one local cache while preserving the frozen source/time protocol."""
    _, path = _dataset_entry(config, dataset)
    global_roi, full_bounds, source_prediction, source_audit, source_metadata = (
        _global_ivd_crop(dataset, visual_spec)
    )
    source_index = int(visual_spec["source_indices"][dataset])
    if int(source_metadata["source_start_index"]) != source_index:
        raise RuntimeError(f"{dataset}: global crop source index changed")
    output_dir = Path(config.output.cache_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"slice_00_index_{source_index:04d}.npz"
    if output_path.exists() and not overwrite:
        return output_dir

    started = time.time()
    if dataset == "channel":
        field, load_metadata = _load_channel_window(
            config, path, source_index
        )
    else:
        future_intervals = int(np.ceil(
            float(config.pathlines.dt_scale)
            * int(config.pathlines.integration_steps)
        ))
        frame_count = future_intervals + 2
        field, load_metadata = load_netcdf_window_3d(
            path,
            source_index,
            frame_count,
            int(config.sampling.max_spatial_dim),
        )
    local_spec = visual_spec["local_sampling"]
    roi_focus = str(local_spec.get("roi_focus", "ivd_boundary"))
    if roi_focus == "frozen_audit_bounds":
        # Re-integrate densely inside a previously audited, prediction-
        # independent physical crop.  The optional sampling_box_scale below
        # performs the requested stronger close-up without reducing the number
        # of evaluated primitives shown in the final figure.
        roi = np.asarray(global_roi, dtype=np.float64)
        volume_threshold = float(source_metadata["ivd_threshold"])
    elif roi_focus == "ivd_dense_positive":
        # The global, frozen IVD labels select a compact positive component.
        # Do not silently replace that component with the strongest gradient
        # point from the whole volume; doing so made every flow show an
        # unrelated, nearly planar sub-voxel patch.
        roi = np.asarray(global_roi, dtype=np.float64)
        volume_threshold = float(source_metadata["ivd_threshold"])
    elif roi_focus == "ivd_dense_boundary":
        # Preserve the component selected from the frozen global labels, but
        # recenter the source box on its nearest continuous p95 crossing. This
        # matters once the display is tighter than one source voxel.
        roi, volume_threshold = _ivd_volume_boundary_crop_near(
            field,
            {
                **local_spec,
                "ivd_percentile": float(config.reference.percentile),
            },
            np.asarray(global_roi, dtype=np.float64).mean(axis=0),
            float(source_metadata["ivd_threshold"]),
        )
    else:
        roi, volume_threshold = _ivd_volume_boundary_crop(
            field, {
                **local_spec,
                "ivd_percentile": float(config.reference.percentile),
            },
        )
    offset = resolve_primitive_offset(
        field.gridInterval,
        float(config.pathlines.offset_grid_scale),
        str(getattr(config.pathlines, "offset_mode", "min")),
    )
    source_roi = np.asarray(roi, dtype=np.float64)
    scale_by_dataset = local_spec.get("sampling_box_scale_by_dataset", {})
    sampling_box_scale = float(
        scale_by_dataset.get(
            dataset, local_spec.get("sampling_box_scale", 1.0)
        )
    )
    roi = _scaled_local_bounds(
        source_roi, sampling_box_scale
    )
    sampling_pattern = str(local_spec.get("sampling_pattern", "regular_grid"))
    local_shape = local_spec.get("seed_grid_shape")
    if sampling_pattern == "regular_grid":
        if local_shape is None:
            raise ValueError("regular_grid sampling requires seed_grid_shape")
        seeds = _local_grid(roi, local_shape, offset)
    elif sampling_pattern == "sobol":
        seeds = _local_sobol_points(
            roi,
            int(local_spec["seed_count"]),
            int(local_spec.get("sampling_seed", 0)),
            offset,
            np.stack((
                np.asarray(field.domainMinBoundary, dtype=np.float64),
                np.asarray(field.domainMaxBoundary, dtype=np.float64),
            )),
        )
    else:
        raise ValueError(
            "local_sampling.sampling_pattern must be regular_grid or sobol"
        )
    dt = float(field.timeInterval) * float(config.pathlines.dt_scale)
    primitives, valid_mask, lengths = integrate_cross_primitives_3d(
        field,
        seeds,
        0.0,
        dt,
        int(config.pathlines.integration_steps),
        int(config.pathlines.sampled_steps),
        offset,
        method=str(config.pathlines.method),
        chunk_size=int(config.pathlines.chunk_size),
    )
    seeds_valid = seeds[valid_mask]
    # The cache contract stores seed coordinates as float32.  Compute the
    # reference labels at those exact serialized coordinates; otherwise a seed
    # lying on the IVD threshold can change class after reload even though the
    # field and percentile are unchanged.
    seeds_serialized = np.ascontiguousarray(seeds_valid, dtype=np.float32)
    if len(seeds_valid) < 100:
        raise RuntimeError(
            f"{dataset}: only {len(seeds_valid)} valid local primitives"
        )
    raw_features = _raw_local_features(primitives)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmt_features = pathline_dft_features_3d(
        torch.from_numpy(primitives).to(device),
        num_freq=int(config.encoder.num_freq),
        neighbor_weight=1.0,
        neighbor_scale=1.0,
        neighbor_pool=str(config.encoder.neighbor_pool),
        mode=str(config.encoder.mode),
        include_chirality=bool(config.encoder.include_chirality),
    ).astype(np.float32)
    ivd_volume, ivd_at_seeds, _ = compute_ivd_reference_3d(
        field, 0.0, seeds_serialized
    )
    threshold = float(np.percentile(
        ivd_volume[np.isfinite(ivd_volume)],
        float(config.reference.percentile),
    ))
    if roi_focus in {"ivd_dense_positive", "ivd_dense_boundary"} and not np.isclose(
        threshold,
        volume_threshold,
        rtol=1e-7,
        atol=1e-7 * max(1.0, abs(volume_threshold)),
    ):
        raise RuntimeError(
            f"{dataset}: frozen and recomputed IVD thresholds differ "
            f"({volume_threshold} vs {threshold})"
        )
    reference = ivd_at_seeds >= threshold
    metadata = {
        "dataset": dataset,
        "ordinal": 0,
        **load_metadata,
        "valid_primitives": int(len(seeds_valid)),
        "total_primitives": int(len(seeds)),
        "ivd_threshold": threshold,
        "roi_ivd_threshold": volume_threshold,
        "ivd_positive_count": int(reference.sum()),
        "ivd_positive_fraction": float(reference.mean()),
        "primitive_offset": float(offset),
        "primitive_offset_mode": str(
            getattr(config.pathlines, "offset_mode", "min")
        ),
        "local_sampling_pattern": sampling_pattern,
        "local_seed_grid_shape": (
            None if local_shape is None else [int(value) for value in local_shape]
        ),
        "local_seed_count": int(len(seeds)),
        "local_sampling_seed": (
            int(local_spec.get("sampling_seed", 0))
            if sampling_pattern == "sobol" else None
        ),
        "local_sampling_box_scale": sampling_box_scale,
        "source_local_sampling_bounds": source_roi.tolist(),
        "fixed_close_up_bounds": roi.tolist(),
        "full_domain_bounds": full_bounds.tolist(),
        "roi_selected_from": (
            str(local_spec.get(
                "roi_source_description",
                "a frozen audited close-up, subsequently resampled densely",
            ))
            if roi_focus == "frozen_audit_bounds" else
            "nearest continuous p95 edge crossing to the frozen densest "
            "global-IVD component; no model prediction used"
            if roi_focus == "ivd_dense_boundary" else
            "densest compact component of frozen global IVD-positive labels; "
            "no model prediction used"
            if roi_focus == "ivd_dense_positive" else
            "strongest interior sub-voxel p95 edge crossing in the source IVD "
            "volume; no model prediction used"
        ),
        "minimum_source_voxels": float(
            visual_spec["local_sampling"].get("minimum_source_voxels", 4.0)
        ),
        "roi_source_prediction_sha256": _sha256(source_prediction),
        "roi_source_audit_sha256": _sha256(source_audit),
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(
        output_path,
        raw_features=raw_features,
        fmt_features=fmt_features,
        seeds=seeds_serialized,
        reference=reference,
        valid_mask=valid_mask,
        line_lengths=lengths,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    manifest = {
        "experiment": str(config.experiment),
        "dataset": dataset,
        "visualization_only": True,
        "selected_time_indices": [source_index],
        "slices": [metadata],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
