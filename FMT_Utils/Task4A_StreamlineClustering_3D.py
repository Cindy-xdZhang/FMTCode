"""Label-free Task4-a streamline/FMT clustering utilities.

The positive ``VortexIds`` mask only defines vortex instances.  It is never used
as a streamwise/spanwise class target.  Class names are attached post hoc from
the clusters' center-streamline alignment with the channel x axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d


STREAMWISE_CLASS = 1
SPANWISE_CLASS = 2


@dataclass
class ChannelVelocityField3D:
    """Separable steady channel field with periodic x/y interpolation."""

    axes_zyx: tuple[np.ndarray, np.ndarray, np.ndarray]
    velocity_zyx3: np.ndarray
    wall_bounds_z: tuple[float, float]
    periods_xy: tuple[float, float]
    vorticity_zyx3: np.ndarray | None = None

    def __post_init__(self):
        zs, ys, xs = self.axes_zyx
        self._interpolator = RegularGridInterpolator(
            (zs, ys, xs),
            self.velocity_zyx3,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._vorticity_interpolator = None
        if self.vorticity_zyx3 is not None:
            self._vorticity_interpolator = RegularGridInterpolator(
                (zs, ys, xs),
                self.vorticity_zyx3,
                bounds_error=False,
                fill_value=np.nan,
            )

    @classmethod
    def from_vtk(cls, path: str | Path) -> tuple["ChannelVelocityField3D", dict]:
        """Load ``channel.vtk`` without constructing a full coordinate mesh."""

        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        reader = vtk.vtkStructuredGridReader()
        reader.SetFileName(str(path))
        reader.ReadAllVectorsOn()
        reader.Update()
        grid = reader.GetOutput()
        dimensions = [0, 0, 0]
        grid.GetDimensions(dimensions)
        nx, ny, nz = (int(value) for value in dimensions)
        if min(nx, ny, nz) < 2:
            raise ValueError(f"channel grid must be 3D, got dimensions={dimensions}")

        points = np.asarray(vtk_to_numpy(grid.GetPoints().GetData())).reshape(
            nz, ny, nx, 3
        )
        xs = np.asarray(points[0, 0, :, 0], dtype=np.float64)
        ys = np.asarray(points[0, :, 0, 1], dtype=np.float64)
        zs = np.asarray(points[:, 0, 0, 2], dtype=np.float64)
        sample_z = np.unique(np.linspace(0, nz - 1, min(nz, 7), dtype=int))
        sample_y = np.unique(np.linspace(0, ny - 1, min(ny, 7), dtype=int))
        sample_x = np.unique(np.linspace(0, nx - 1, min(nx, 7), dtype=int))
        if not all(np.all(np.diff(axis) > 0) for axis in (xs, ys, zs)):
            raise ValueError("channel x/y/z axes must be strictly increasing")
        if not np.allclose(points[np.ix_(sample_z, sample_y, sample_x)][..., 0],
                           xs[sample_x][None, None, :], rtol=0, atol=1e-6):
            raise ValueError("channel x coordinates are not separable")
        if not np.allclose(points[np.ix_(sample_z, sample_y, sample_x)][..., 1],
                           ys[sample_y][None, :, None], rtol=0, atol=1e-6):
            raise ValueError("channel y coordinates are not separable")
        if not np.allclose(points[np.ix_(sample_z, sample_y, sample_x)][..., 2],
                           zs[sample_z][:, None, None], rtol=0, atol=1e-6):
            raise ValueError("channel z coordinates are not separable")

        velocity_array = grid.GetPointData().GetArray("velocity")
        if velocity_array is None or velocity_array.GetNumberOfComponents() != 3:
            raise ValueError("channel VTK misses 3-component point array 'velocity'")
        velocity = np.asarray(vtk_to_numpy(velocity_array), dtype=np.float32).reshape(
            nz, ny, nx, 3
        )
        vorticity_array = grid.GetPointData().GetArray("vorticity")
        if vorticity_array is None or vorticity_array.GetNumberOfComponents() != 3:
            raise ValueError("channel VTK misses 3-component point array 'vorticity'")
        vorticity = np.asarray(
            vtk_to_numpy(vorticity_array), dtype=np.float32
        ).reshape(nz, ny, nx, 3)

        # The stored periodic axes omit the repeated endpoint.  Add one seam plane
        # so interpolation remains continuous when an unwrapped streamline crosses it.
        dx = float(np.median(np.diff(xs)))
        dy = float(np.median(np.diff(ys)))
        periodic_x = float(xs[-1] - xs[0] + dx)
        periodic_y = float(ys[-1] - ys[0] + dy)
        padded = np.concatenate((velocity, velocity[:, :, :1]), axis=2)
        padded = np.concatenate((padded, padded[:, :1]), axis=1)
        padded_vorticity = np.concatenate((vorticity, vorticity[:, :, :1]), axis=2)
        padded_vorticity = np.concatenate(
            (padded_vorticity, padded_vorticity[:, :1]), axis=1
        )
        padded_xs = np.append(xs, xs[-1] + dx)
        padded_ys = np.append(ys, ys[-1] + dy)

        field = cls(
            axes_zyx=(zs, padded_ys, padded_xs),
            velocity_zyx3=np.ascontiguousarray(padded, dtype=np.float32),
            wall_bounds_z=(float(zs[0]), float(zs[-1])),
            periods_xy=(periodic_x, periodic_y),
            vorticity_zyx3=np.ascontiguousarray(padded_vorticity, dtype=np.float32),
        )
        component_mean_abs = np.mean(np.abs(velocity), axis=(0, 1, 2))
        vorticity_mean_abs = np.mean(np.abs(vorticity), axis=(0, 1, 2))
        metadata = {
            "path": str(path.resolve()),
            "dimensions_xyz": dimensions,
            "stored_bounds_xyz": [
                [float(xs[0]), float(ys[0]), float(zs[0])],
                [float(xs[-1]), float(ys[-1]), float(zs[-1])],
            ],
            "periods_xy": [periodic_x, periodic_y],
            "median_spacing_xyz": [dx, dy, float(np.median(np.diff(zs)))],
            "minimum_spacing_z": float(np.min(np.diff(zs))),
            "mean_absolute_velocity_xyz": component_mean_abs.astype(float).tolist(),
            "mean_absolute_vorticity_xyz": vorticity_mean_abs.astype(float).tolist(),
            "streamwise_axis": "x",
            "spanwise_axis": "y",
            "wall_normal_axis": "z",
        }
        return field, metadata

    def velocity(self, points_xyz: np.ndarray) -> np.ndarray:
        """Interpolate at unwrapped xyz points; x/y are mapped periodically."""

        points = np.asarray(points_xyz, dtype=np.float64)
        original_shape = points.shape
        if original_shape[-1] != 3:
            raise ValueError(f"points must end in xyz, got shape={original_shape}")
        flat = points.reshape(-1, 3).copy()
        _, ys, xs = self.axes_zyx
        flat[:, 0] = xs[0] + np.mod(flat[:, 0] - xs[0], self.periods_xy[0])
        flat[:, 1] = ys[0] + np.mod(flat[:, 1] - ys[0], self.periods_xy[1])
        values = self._interpolator(flat[:, [2, 1, 0]])
        return np.asarray(values, dtype=np.float32).reshape(original_shape)

    def vorticity(self, points_xyz: np.ndarray) -> np.ndarray:
        """Interpolate physical vorticity at unwrapped xyz points."""

        if self._vorticity_interpolator is None:
            raise RuntimeError("this channel field does not contain vorticity")
        points = np.asarray(points_xyz, dtype=np.float64)
        original_shape = points.shape
        if original_shape[-1] != 3:
            raise ValueError(f"points must end in xyz, got shape={original_shape}")
        flat = points.reshape(-1, 3).copy()
        _, ys, xs = self.axes_zyx
        flat[:, 0] = xs[0] + np.mod(flat[:, 0] - xs[0], self.periods_xy[0])
        flat[:, 1] = ys[0] + np.mod(flat[:, 1] - ys[0], self.periods_xy[1])
        values = self._vorticity_interpolator(flat[:, [2, 1, 0]])
        return np.asarray(values, dtype=np.float32).reshape(original_shape)


def cross_offsets_3d(offset: float) -> np.ndarray:
    """Return center, x-/x+, y-/y+, z-/z+ offsets."""

    offset = float(offset)
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError("cross offset must be positive and finite")
    return offset * np.asarray(
        [
            [0, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def _unit_direction(
    field: ChannelVelocityField3D, points: np.ndarray, minimum_speed: float
) -> np.ndarray:
    velocity = field.velocity(points)
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    valid = np.isfinite(velocity).all(axis=-1, keepdims=True) & (speed >= minimum_speed)
    return np.where(valid, velocity / np.maximum(speed, minimum_speed), np.nan)


def _parameterized_velocity(
    field: ChannelVelocityField3D,
    points: np.ndarray,
    minimum_speed: float,
    unit_velocity: bool,
) -> np.ndarray:
    velocity = field.velocity(points)
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    valid = np.isfinite(velocity).all(axis=-1, keepdims=True) & (speed >= minimum_speed)
    if unit_velocity:
        velocity = velocity / np.maximum(speed, minimum_speed)
    return np.where(valid, velocity, np.nan)


def _rk4_streamline_step(
    field: ChannelVelocityField3D,
    points: np.ndarray,
    signed_step: float,
    minimum_speed: float,
    unit_velocity: bool = True,
) -> np.ndarray:
    k1 = _parameterized_velocity(field, points, minimum_speed, unit_velocity)
    k2 = _parameterized_velocity(
        field, points + 0.5 * signed_step * k1, minimum_speed, unit_velocity
    )
    k3 = _parameterized_velocity(
        field, points + 0.5 * signed_step * k2, minimum_speed, unit_velocity
    )
    k4 = _parameterized_velocity(
        field, points + signed_step * k3, minimum_speed, unit_velocity
    )
    return points + (signed_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_bidirectional_cross_primitives(
    field: ChannelVelocityField3D,
    seeds_xyz: np.ndarray,
    spatial_step: float,
    steps_per_direction: int,
    offset: float,
    minimum_speed: float = 1e-6,
    chunk_size: int = 8192,
    unit_velocity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate seven-line steady primitives as ``[N,7,2*S+1,3]``.

    ``unit_velocity=True`` preserves the historical equal-arc-length sampling.
    ``False`` uses ``dx/dt=v(x)`` so displacements retain local speed and allow
    velocity-gradient reconstruction from the cross geometry.
    """

    seeds = np.asarray(seeds_xyz, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError(f"seeds_xyz must be [N,3], got {seeds.shape}")
    steps_per_direction = int(steps_per_direction)
    chunk_size = int(chunk_size)
    if steps_per_direction < 1 or chunk_size < 1:
        raise ValueError("steps_per_direction and chunk_size must be positive")
    if spatial_step <= 0 or minimum_speed <= 0:
        raise ValueError("spatial_step and minimum_speed must be positive")

    offsets = cross_offsets_3d(offset)
    output = np.full(
        (len(seeds), 7, 2 * steps_per_direction + 1, 3),
        np.nan,
        dtype=np.float32,
    )
    for start in range(0, len(seeds), chunk_size):
        stop = min(len(seeds), start + chunk_size)
        initial = seeds[start:stop, None, :] + offsets[None, :, :]
        backward = [initial]
        current = initial
        for _ in range(steps_per_direction):
            current = _rk4_streamline_step(
                field,
                current,
                -float(spatial_step),
                float(minimum_speed),
                bool(unit_velocity),
            )
            backward.append(current)
        forward = [initial]
        current = initial
        for _ in range(steps_per_direction):
            current = _rk4_streamline_step(
                field,
                current,
                float(spatial_step),
                float(minimum_speed),
                bool(unit_velocity),
            )
            forward.append(current)
        combined = np.stack(backward[:0:-1] + forward, axis=2)
        output[start:stop] = combined.astype(np.float32, copy=False)
    valid = np.isfinite(output).all(axis=(1, 2, 3))
    return output, valid


def stratified_fit_indices(
    vortex_ids: np.ndarray, max_per_vortex: int, seed: int
) -> np.ndarray:
    """Sample equal-capped KMeans fit populations without using class labels."""

    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    max_per_vortex = int(max_per_vortex)
    if max_per_vortex < 1 or np.any(ids <= 0):
        raise ValueError("vortex IDs must be positive and max_per_vortex >= 1")
    rng = np.random.default_rng(int(seed))
    selected = []
    for vortex_id in np.unique(ids):
        candidates = np.flatnonzero(ids == vortex_id)
        if len(candidates) > max_per_vortex:
            candidates = np.sort(
                rng.choice(candidates, size=max_per_vortex, replace=False)
            )
        selected.append(candidates)
    return np.concatenate(selected).astype(np.int64, copy=False)


def cluster_fmt_primitives(
    primitives: np.ndarray,
    vortex_ids: np.ndarray,
    semantic_orientation_vectors_xyz: np.ndarray,
    *,
    num_freq: int = 6,
    mode: str = "gram",
    include_chirality: bool = True,
    neighbor_pool: str = "sort",
    neighbor_weight: float = 0.5,
    n_init: int = 20,
    fit_max_seeds_per_vortex: int = 128,
    seed: int = 7068,
) -> dict:
    """Encode with pure FMT, fit KMeans(2), then name clusters from vorticity."""

    primitives = np.asarray(primitives, dtype=np.float32)
    vortex_ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    if primitives.ndim != 4 or primitives.shape[0] != len(vortex_ids):
        raise ValueError("primitives must be [N,K,L,3] and match vortex_ids")
    if not np.isfinite(primitives).all():
        raise ValueError("cluster_fmt_primitives received invalid primitives")
    orientation = np.asarray(semantic_orientation_vectors_xyz, dtype=np.float32)
    if orientation.shape != (len(primitives), 3) or not np.isfinite(orientation).all():
        raise ValueError("semantic orientation vectors must be finite [N,3] vorticity")

    raw_features = pathline_dft_features_3d(
        torch.from_numpy(primitives),
        num_freq=int(num_freq),
        neighbor_weight=1.0,
        neighbor_scale=1.0,
        neighbor_pool=str(neighbor_pool),
        mode=str(mode),
        include_chirality=bool(include_chirality),
    )
    if not np.isfinite(raw_features).all():
        raise RuntimeError("FMT produced non-finite features")
    fit_indices = stratified_fit_indices(
        vortex_ids, int(fit_max_seeds_per_vortex), int(seed)
    )
    scaler = StandardScaler().fit(raw_features[fit_indices])
    standardized = scaler.transform(raw_features)
    base_width = int(num_freq) * (1 if mode == "magnitude" else 3) + (
        int(num_freq) - 1 if include_chirality else 0
    )
    if standardized.shape[1] <= base_width:
        raise RuntimeError("FMT feature matrix lacks neighbor feature blocks")
    standardized[:, base_width:] *= float(neighbor_weight)

    kmeans = KMeans(
        n_clusters=2,
        random_state=int(seed),
        n_init=int(n_init),
    ).fit(standardized[fit_indices])
    raw_clusters = kmeans.predict(standardized).astype(np.int8)

    displacement = primitives[:, 0, -1] - primitives[:, 0, 0]
    length = np.linalg.norm(displacement, axis=1, keepdims=True)
    endpoint_alignment = np.abs(displacement) / np.maximum(length, 1e-12)
    endpoint_axis_medians = np.asarray(
        [np.median(endpoint_alignment[raw_clusters == cluster], axis=0)
         for cluster in (0, 1)]
    )
    orientation_length = np.linalg.norm(orientation, axis=1, keepdims=True)
    orientation_alignment = np.abs(orientation) / np.maximum(orientation_length, 1e-12)
    orientation_axis_medians = np.asarray(
        [np.median(orientation_alignment[raw_clusters == cluster], axis=0)
         for cluster in (0, 1)]
    )
    streamwise_cluster = int(np.argmax(orientation_axis_medians[:, 0]))
    semantic = np.where(
        raw_clusters == streamwise_cluster, STREAMWISE_CLASS, SPANWISE_CLASS
    ).astype(np.int8)
    return {
        "raw_features": np.asarray(raw_features, dtype=np.float32),
        "standardized_features": np.asarray(standardized, dtype=np.float32),
        "fit_indices": fit_indices,
        "raw_clusters": raw_clusters,
        "semantic_classes": semantic,
        "centerline_absolute_axis_alignment": endpoint_alignment.astype(np.float32),
        "cluster_endpoint_axis_alignment_medians_xyz": endpoint_axis_medians.astype(
            np.float64
        ),
        "vorticity_absolute_axis_alignment": orientation_alignment.astype(np.float32),
        "cluster_vorticity_axis_alignment_medians_xyz": orientation_axis_medians.astype(
            np.float64
        ),
        "streamwise_raw_cluster": streamwise_cluster,
        "spanwise_raw_cluster": int(1 - streamwise_cluster),
        "feature_dimension": int(raw_features.shape[1]),
    }


def fill_invalid_predictions_within_vortex(
    predicted_classes: np.ndarray,
    valid_mask: np.ndarray,
    vortex_ids: np.ndarray,
    indices_xyz: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Nearest-fill invalid primitives from valid voxels of the same instance."""

    predicted = np.asarray(predicted_classes, dtype=np.int8).reshape(-1).copy()
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    indices = np.asarray(indices_xyz, dtype=np.float64)
    if not (len(predicted) == len(valid) == len(ids) == len(indices)):
        raise ValueError("prediction, validity, IDs, and indices must have equal length")
    filled_count = 0
    missing_ids = []
    for vortex_id in np.unique(ids):
        member = np.flatnonzero(ids == vortex_id)
        source = member[valid[member]]
        target = member[~valid[member]]
        if not len(target):
            continue
        if not len(source):
            missing_ids.append(int(vortex_id))
            predicted[target] = 0
            continue
        nearest = cKDTree(indices[source]).query(indices[target], k=1)[1]
        predicted[target] = predicted[source[np.asarray(nearest, dtype=int)]]
        filled_count += int(len(target))
    return predicted, {
        "invalid_primitive_count": int(np.count_nonzero(~valid)),
        "nearest_filled_voxel_count": filled_count,
        "vortex_ids_without_valid_primitive": missing_ids,
        "unfilled_voxel_count": int(np.count_nonzero(predicted == 0)),
    }


def _component_sizes(mask: np.ndarray, structure: np.ndarray) -> list[int]:
    components, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return []
    sizes = np.bincount(components.ravel())[1:]
    return sorted((int(value) for value in sizes), reverse=True)


def topology_proxy_rows(
    vortex_ids_zyx: np.ndarray,
    predicted_classes_zyx: np.ndarray,
    dominant_min_voxels: int = 3,
    dominant_min_fraction: float = 0.05,
) -> list[dict]:
    """Compute strict 2+1 and soft component-mass proxies per vortex instance."""

    instances = np.asarray(vortex_ids_zyx, dtype=np.int32)
    classes = np.asarray(predicted_classes_zyx, dtype=np.int8)
    if instances.shape != classes.shape or instances.ndim != 3:
        raise ValueError("instance and class volumes must be same-shape 3D arrays")
    structure = np.ones((3, 3, 3), dtype=bool)
    rows = []
    for vortex_id in np.unique(instances[instances > 0]):
        mask = instances == vortex_id
        total = int(np.count_nonzero(mask))
        input_sizes = _component_sizes(mask, structure)
        threshold = max(
            int(dominant_min_voxels), int(np.ceil(float(dominant_min_fraction) * total))
        )
        stream_sizes = _component_sizes(mask & (classes == STREAMWISE_CLASS), structure)
        span_sizes = _component_sizes(mask & (classes == SPANWISE_CLASS), structure)
        stream_dominant = [size for size in stream_sizes if size >= threshold]
        span_dominant = [size for size in span_sizes if size >= threshold]
        stream_count = int(np.count_nonzero(mask & (classes == STREAMWISE_CLASS)))
        span_count = int(np.count_nonzero(mask & (classes == SPANWISE_CLASS)))
        unassigned_count = total - stream_count - span_count
        target_mass = sum(stream_sizes[:2]) + sum(span_sizes[:1])
        mass_fraction = float(target_mass / total) if total else 0.0
        count_error = abs(len(stream_dominant) - 2) + abs(len(span_dominant) - 1)
        soft_score = float(mass_fraction * np.exp(-count_error))
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "voxel_count": total,
                "input_component_count": len(input_sizes),
                "input_single_component": len(input_sizes) == 1,
                "dominant_component_threshold": threshold,
                "streamwise_voxel_count": stream_count,
                "spanwise_voxel_count": span_count,
                "unassigned_voxel_count": unassigned_count,
                "streamwise_fraction": float(stream_count / total),
                "spanwise_fraction": float(span_count / total),
                "streamwise_component_count_all": len(stream_sizes),
                "spanwise_component_count_all": len(span_sizes),
                "streamwise_dominant_component_count": len(stream_dominant),
                "spanwise_dominant_component_count": len(span_dominant),
                "target_component_mass_fraction": mass_fraction,
                "component_count_error": int(count_error),
                "soft_topology_score": soft_score,
                "hard_2plus1_success": bool(
                    len(stream_dominant) == 2 and len(span_dominant) == 1
                ),
            }
        )
    return rows


def summarize_topology(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("topology summary requires at least one vortex")
    connected = [row for row in rows if row["input_single_component"]]
    return {
        "vortex_id_count": len(rows),
        "input_single_component_count": len(connected),
        "input_disconnected_count": len(rows) - len(connected),
        "hard_2plus1_success_count_all": int(
            sum(row["hard_2plus1_success"] for row in rows)
        ),
        "hard_2plus1_success_rate_all": float(
            np.mean([row["hard_2plus1_success"] for row in rows])
        ),
        "hard_2plus1_success_count_single_input_component": int(
            sum(row["hard_2plus1_success"] for row in connected)
        ),
        "hard_2plus1_success_rate_single_input_component": float(
            np.mean([row["hard_2plus1_success"] for row in connected])
        ) if connected else None,
        "mean_soft_topology_score_all": float(
            np.mean([row["soft_topology_score"] for row in rows])
        ),
        "median_soft_topology_score_all": float(
            np.median([row["soft_topology_score"] for row in rows])
        ),
        "mean_soft_topology_score_single_input_component": float(
            np.mean([row["soft_topology_score"] for row in connected])
        ) if connected else None,
    }
