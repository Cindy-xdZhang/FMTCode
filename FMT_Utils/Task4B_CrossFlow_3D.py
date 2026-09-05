"""Cross-flow utilities for Task4-b channel-to-TBL transfer.

The historical channel loader assumes periodic x/y axes and a stored vector
``vorticity`` array.  The TBL file is non-periodic and stores only ``velocity``
plus the spanwise vorticity fluctuation ``oyf``.  This module therefore keeps
the existing channel implementation frozen and provides a separate, explicit
TBL path based on finite-difference curl reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d


def finite_difference_curl_zyx(
    velocity_zyx3: np.ndarray,
    axes_zyx: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Compute ``curl(velocity)`` on a separable, possibly nonuniform grid."""

    velocity = np.asarray(velocity_zyx3, dtype=np.float32)
    zs, ys, xs = (np.asarray(axis, dtype=np.float64) for axis in axes_zyx)
    expected = (len(zs), len(ys), len(xs), 3)
    if velocity.shape != expected:
        raise ValueError(f"velocity shape {velocity.shape} != {expected}")
    if any(len(axis) < 3 or not np.all(np.diff(axis) > 0.0) for axis in (zs, ys, xs)):
        raise ValueError("curl axes must be strictly increasing with at least 3 points")

    curl = np.empty_like(velocity, dtype=np.float32)
    first = np.gradient(velocity[..., 2], ys, axis=1, edge_order=2)
    second = np.gradient(velocity[..., 1], zs, axis=0, edge_order=2)
    np.subtract(first, second, out=curl[..., 0])
    del first, second

    first = np.gradient(velocity[..., 0], zs, axis=0, edge_order=2)
    second = np.gradient(velocity[..., 2], xs, axis=2, edge_order=2)
    np.subtract(first, second, out=curl[..., 1])
    del first, second

    first = np.gradient(velocity[..., 1], xs, axis=2, edge_order=2)
    second = np.gradient(velocity[..., 0], ys, axis=1, edge_order=2)
    np.subtract(first, second, out=curl[..., 2])
    del first, second
    if not np.isfinite(curl).all():
        raise ValueError("finite-difference curl contains non-finite values")
    return curl


def _seam_jump_ratio(values: np.ndarray, axis: int) -> float:
    """Compare the first/last-plane jump with typical adjacent-plane jumps."""

    values = np.asarray(values, dtype=np.float32)
    first = np.take(values, 0, axis=axis)
    last = np.take(values, values.shape[axis] - 1, axis=axis)
    boundary = float(np.mean(np.linalg.norm(last - first, axis=-1)))
    sample = np.unique(
        np.linspace(0, values.shape[axis] - 2, min(17, values.shape[axis] - 1), dtype=int)
    )
    adjacent = []
    for index in sample:
        left = np.take(values, int(index), axis=axis)
        right = np.take(values, int(index) + 1, axis=axis)
        adjacent.append(float(np.mean(np.linalg.norm(right - left, axis=-1))))
    reference = float(np.median(adjacent))
    return boundary / max(reference, 1.0e-12)


def _agreement_statistics(reference: np.ndarray, observed: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    finite = np.isfinite(reference) & np.isfinite(observed)
    reference = reference[finite]
    observed = observed[finite]
    if len(reference) < 2:
        raise ValueError("not enough finite values for oyf agreement")
    difference = reference - observed
    active = (np.abs(reference) + np.abs(observed)) > 1.0e-8
    return {
        "sample_count": int(len(reference)),
        "pearson_correlation": float(np.corrcoef(reference, observed)[0, 1]),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "mae": float(np.mean(np.abs(difference))),
        "mean_bias_derived_minus_stored": float(np.mean(difference)),
        "stored_standard_deviation": float(np.std(observed)),
        "active_sign_agreement": float(
            np.mean(np.signbit(reference[active]) == np.signbit(observed[active]))
        ),
    }


@dataclass
class NonPeriodicStructuredVelocityField3D:
    """Separable structured velocity field with no artificial seam wrapping."""

    axes_zyx: tuple[np.ndarray, np.ndarray, np.ndarray]
    velocity_zyx3: np.ndarray
    vorticity_zyx3: np.ndarray
    omega_y_prime_zyx: np.ndarray

    def __post_init__(self) -> None:
        expected = tuple(len(axis) for axis in self.axes_zyx)
        if self.velocity_zyx3.shape != expected + (3,):
            raise ValueError("velocity shape does not match structured axes")
        if self.vorticity_zyx3.shape != expected + (3,):
            raise ValueError("vorticity shape does not match structured axes")
        if self.omega_y_prime_zyx.shape != expected:
            raise ValueError("omega_y_prime shape does not match structured axes")
        self._velocity_interpolator = RegularGridInterpolator(
            self.axes_zyx,
            self.velocity_zyx3,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._vorticity_interpolator = RegularGridInterpolator(
            self.axes_zyx,
            self.vorticity_zyx3,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._omega_y_prime_interpolator = RegularGridInterpolator(
            self.axes_zyx,
            self.omega_y_prime_zyx,
            bounds_error=False,
            fill_value=np.nan,
        )

    @classmethod
    def from_vtk(
        cls,
        path: str | Path,
        *,
        velocity_array_name: str = "velocity",
        omega_y_prime_array_name: str = "oyf",
        minimum_oyf_correlation: float = 0.90,
    ) -> tuple["NonPeriodicStructuredVelocityField3D", dict]:
        """Load TBL velocity and reconstruct its missing vector curl."""

        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        reader = vtk.vtkStructuredGridReader()
        reader.SetFileName(str(path))
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        grid = reader.GetOutput()
        dimensions = [0, 0, 0]
        grid.GetDimensions(dimensions)
        nx, ny, nz = (int(value) for value in dimensions)
        if min(nx, ny, nz) < 3:
            raise ValueError(f"TBL grid must be three-dimensional: {dimensions}")

        points = np.asarray(vtk_to_numpy(grid.GetPoints().GetData())).reshape(
            nz, ny, nx, 3
        )
        xs = np.asarray(points[0, 0, :, 0], dtype=np.float64).copy()
        ys = np.asarray(points[0, :, 0, 1], dtype=np.float64).copy()
        zs = np.asarray(points[:, 0, 0, 2], dtype=np.float64).copy()
        sample_z = np.unique(np.linspace(0, nz - 1, min(nz, 7), dtype=int))
        sample_y = np.unique(np.linspace(0, ny - 1, min(ny, 7), dtype=int))
        sample_x = np.unique(np.linspace(0, nx - 1, min(nx, 7), dtype=int))
        if not all(np.all(np.diff(axis) > 0.0) for axis in (xs, ys, zs)):
            raise ValueError("TBL x/y/z axes must be strictly increasing")
        sampled_points = points[np.ix_(sample_z, sample_y, sample_x)]
        if not np.allclose(
            sampled_points[..., 0], xs[sample_x][None, None, :], rtol=0, atol=1e-5
        ):
            raise ValueError("TBL x coordinates are not separable")
        if not np.allclose(
            sampled_points[..., 1], ys[sample_y][None, :, None], rtol=0, atol=1e-5
        ):
            raise ValueError("TBL y coordinates are not separable")
        if not np.allclose(
            sampled_points[..., 2], zs[sample_z][:, None, None], rtol=0, atol=1e-5
        ):
            raise ValueError("TBL z coordinates are not separable")

        point_data = grid.GetPointData()
        velocity_array = point_data.GetArray(str(velocity_array_name))
        if velocity_array is None or velocity_array.GetNumberOfComponents() != 3:
            raise ValueError("TBL VTK misses three-component point array 'velocity'")
        oyf_array = point_data.GetArray(str(omega_y_prime_array_name))
        if oyf_array is None or oyf_array.GetNumberOfComponents() != 1:
            raise ValueError("TBL VTK misses scalar point array 'oyf'")
        velocity = np.array(
            vtk_to_numpy(velocity_array), dtype=np.float32, copy=True
        ).reshape(nz, ny, nx, 3)
        stored_oyf = np.array(
            vtk_to_numpy(oyf_array), dtype=np.float32, copy=True
        ).reshape(nz, ny, nx)
        del sampled_points, points, grid, reader

        axes_zyx = (zs, ys, xs)
        vorticity = finite_difference_curl_zyx(velocity, axes_zyx)
        derived_oyf = vorticity[..., 1] - vorticity[..., 1].mean(
            axis=(1, 2), keepdims=True, dtype=np.float64
        ).astype(np.float32)
        agreement = _agreement_statistics(derived_oyf, stored_oyf)
        if agreement["pearson_correlation"] < float(minimum_oyf_correlation):
            raise RuntimeError(
                "stored oyf is inconsistent with the reconstructed spanwise "
                f"vorticity fluctuation: {agreement}"
            )

        field = cls(
            axes_zyx=axes_zyx,
            velocity_zyx3=velocity,
            vorticity_zyx3=vorticity,
            # The stored field is preferred for the head proxy; it is the source
            # quantity and avoids silently replacing it with a numerical derivative.
            omega_y_prime_zyx=stored_oyf,
        )
        metadata = {
            "path": str(path.resolve()),
            "dimensions_xyz": dimensions,
            "stored_bounds_xyz": [
                [float(xs[0]), float(ys[0]), float(zs[0])],
                [float(xs[-1]), float(ys[-1]), float(zs[-1])],
            ],
            "median_spacing_xyz": [
                float(np.median(np.diff(xs))),
                float(np.median(np.diff(ys))),
                float(np.median(np.diff(zs))),
            ],
            "minimum_spacing_xyz": [
                float(np.min(np.diff(xs))),
                float(np.min(np.diff(ys))),
                float(np.min(np.diff(zs))),
            ],
            "boundary_condition_xyz": ["nonperiodic", "nonperiodic", "wall/nonperiodic"],
            "velocity_seam_jump_ratio_x": _seam_jump_ratio(velocity, axis=2),
            "velocity_seam_jump_ratio_y": _seam_jump_ratio(velocity, axis=1),
            "vorticity_source": "second_order_finite_difference_curl_of_velocity",
            "omega_y_prime_source": str(omega_y_prime_array_name),
            "derived_omega_y_prime_vs_stored_oyf": agreement,
            "streamwise_axis": "x",
            "spanwise_axis": "y",
            "wall_normal_axis": "z",
        }
        return field, metadata

    @staticmethod
    def _query_points(points_xyz: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        points = np.asarray(points_xyz, dtype=np.float64)
        shape = points.shape
        if not shape or shape[-1] != 3:
            raise ValueError(f"points must end in xyz, got {shape}")
        return points.reshape(-1, 3)[:, [2, 1, 0]], shape

    def velocity(self, points_xyz: np.ndarray) -> np.ndarray:
        query, shape = self._query_points(points_xyz)
        return np.asarray(self._velocity_interpolator(query), dtype=np.float32).reshape(shape)

    def vorticity(self, points_xyz: np.ndarray) -> np.ndarray:
        query, shape = self._query_points(points_xyz)
        return np.asarray(self._vorticity_interpolator(query), dtype=np.float32).reshape(shape)

    def omega_y_prime(self, points_xyz: np.ndarray) -> np.ndarray:
        query, shape = self._query_points(points_xyz)
        return np.asarray(self._omega_y_prime_interpolator(query), dtype=np.float32).reshape(
            shape[:-1]
        )

    def wall_normal_vorticity_deviation(self) -> np.ndarray:
        """Return boundary-layer-profile vorticity deviation on the flow grid."""

        mean = self.vorticity_zyx3.mean(axis=(1, 2), keepdims=True, dtype=np.float64)
        deviation = self.vorticity_zyx3.astype(np.float64) - mean
        # ``oyf`` is the source-provided y fluctuation and replaces only the
        # numerically reconstructed y-deviation component.
        deviation[..., 1] = self.omega_y_prime_zyx
        result = np.linalg.norm(deviation, axis=-1).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("TBL profile vorticity deviation contains non-finite values")
        return result


def audit_indexed_vtk_reference(
    reference_path: str | Path,
    field: NonPeriodicStructuredVelocityField3D,
    *,
    require_vorticity: bool,
) -> dict:
    """Audit a sparse VTK file whose ``PointIds`` index the full TBL grid."""

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reference_path = Path(reference_path)
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(reference_path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    data = reader.GetOutput()
    point_data = data.GetPointData()
    ids_array = point_data.GetArray("PointIds")
    if ids_array is None:
        raise ValueError(f"{reference_path} misses point array PointIds")
    point_ids = np.asarray(vtk_to_numpy(ids_array), dtype=np.int64).reshape(-1)
    if len(point_ids) != data.GetNumberOfPoints():
        raise ValueError("reference PointIds must match point count")
    zs, ys, xs = field.axes_zyx
    nx, ny, nz = len(xs), len(ys), len(zs)
    if point_ids.min(initial=0) < 0 or point_ids.max(initial=-1) >= nx * ny * nz:
        raise ValueError("reference PointIds fall outside the TBL structured grid")
    iz, remainder = np.divmod(point_ids, ny * nx)
    iy, ix = np.divmod(remainder, nx)
    expected_points = np.column_stack((xs[ix], ys[iy], zs[iz]))
    stored_points = np.asarray(vtk_to_numpy(data.GetPoints().GetData()), dtype=np.float64)
    point_coordinate_max_abs_error = float(np.max(np.abs(stored_points - expected_points)))

    result = {
        "path": str(reference_path.resolve()),
        "point_count": int(len(point_ids)),
        "unique_point_id_count": int(len(np.unique(point_ids))),
        "duplicated_point_record_count": int(len(point_ids) - len(np.unique(point_ids))),
        "point_id_min": int(np.min(point_ids)),
        "point_id_max": int(np.max(point_ids)),
        "point_coordinate_max_abs_error": point_coordinate_max_abs_error,
    }
    oyf_array = point_data.GetArray("oyf")
    if oyf_array is not None:
        reference_oyf = np.asarray(vtk_to_numpy(oyf_array), dtype=np.float32).reshape(-1)
        source_oyf = field.omega_y_prime_zyx.reshape(-1)[point_ids]
        result["stored_oyf_max_abs_error"] = float(
            np.max(np.abs(reference_oyf - source_oyf))
        )

    vorticity_array = point_data.GetArray("vorticity")
    if require_vorticity and (
        vorticity_array is None or vorticity_array.GetNumberOfComponents() != 3
    ):
        raise ValueError(f"{reference_path} misses three-component point vorticity")
    if vorticity_array is not None:
        reference = np.asarray(vtk_to_numpy(vorticity_array), dtype=np.float64).reshape(-1, 3)
        derived = field.vorticity_zyx3.reshape(-1, 3)[point_ids].astype(np.float64)
        correlations = []
        sign_agreements = []
        for component in range(3):
            correlations.append(float(np.corrcoef(derived[:, component], reference[:, component])[0, 1]))
            active = (np.abs(derived[:, component]) + np.abs(reference[:, component])) > 1.0e-8
            sign_agreements.append(
                float(
                    np.mean(
                        np.signbit(derived[active, component])
                        == np.signbit(reference[active, component])
                    )
                )
            )
        error = derived - reference
        relative_rmse = float(
            np.sqrt(np.mean(np.sum(error * error, axis=1)))
            / max(np.sqrt(np.mean(np.sum(reference * reference, axis=1))), 1.0e-12)
        )
        result["vorticity_component_correlation_xyz"] = correlations
        result["vorticity_component_sign_agreement_xyz"] = sign_agreements
        result["vorticity_vector_relative_rmse"] = relative_rmse
    del data, reader
    return result


def dimensionless_primitive_features(
    primitives: np.ndarray,
    spatial_step: float,
    *,
    num_freq: int,
    neighbor_scale: float,
    neighbor_pool: str,
    mode: str,
    include_chirality: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Create Raw and FMT features in shared integration-step units."""

    primitives = np.asarray(primitives, dtype=np.float32)
    spatial_step = float(spatial_step)
    if primitives.ndim != 4 or primitives.shape[1] != 7 or primitives.shape[-1] != 3:
        raise ValueError("primitives must have shape [N,7,L,3]")
    if not np.isfinite(spatial_step) or spatial_step <= 0.0:
        raise ValueError("spatial_step must be positive and finite")
    normalized = (
        primitives - primitives[:, :1, :1, :]
    ) / np.float32(spatial_step)
    raw = normalized.reshape(len(normalized), -1).astype(np.float32, copy=False)
    fmt = pathline_dft_features_3d(
        torch.from_numpy(normalized),
        num_freq=int(num_freq),
        neighbor_weight=1.0,
        neighbor_scale=float(neighbor_scale),
        neighbor_pool=str(neighbor_pool),
        mode=str(mode),
        include_chirality=bool(include_chirality),
    ).astype(np.float32, copy=False)
    if not np.isfinite(raw).all() or not np.isfinite(fmt).all():
        raise RuntimeError("dimensionless Raw/FMT features contain non-finite values")
    return raw, fmt


def fit_source_normalization(
    source_raw: np.ndarray,
    source_fmt: np.ndarray,
    *,
    sampled_steps: int,
) -> dict[str, np.ndarray]:
    """Fit normalization exclusively on the channel training population."""

    raw = np.asarray(source_raw, dtype=np.float32).reshape(-1, 7, int(sampled_steps), 3)
    fmt = np.asarray(source_fmt, dtype=np.float32)
    return {
        "raw_mean": raw.mean(axis=(0, 1, 2), keepdims=True, dtype=np.float64).astype(np.float32),
        "raw_std": np.maximum(
            raw.std(axis=(0, 1, 2), keepdims=True, dtype=np.float64).astype(np.float32),
            1.0e-6,
        ),
        "fmt_mean": fmt.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32),
        "fmt_std": np.maximum(
            fmt.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32),
            1.0e-6,
        ),
    }


def apply_source_normalization(
    raw_features: np.ndarray,
    fmt_features: np.ndarray,
    normalization: dict[str, np.ndarray],
    *,
    sampled_steps: int,
    fmt_base_width: int,
    neighbor_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen channel statistics to either channel or TBL features."""

    raw = np.asarray(raw_features, dtype=np.float32).reshape(-1, 7, int(sampled_steps), 3)
    fmt = np.asarray(fmt_features, dtype=np.float32)
    raw = ((raw - normalization["raw_mean"]) / normalization["raw_std"]).astype(
        np.float32
    )
    fmt = ((fmt - normalization["fmt_mean"]) / normalization["fmt_std"]).astype(
        np.float32
    )
    fmt[:, int(fmt_base_width) :] *= float(neighbor_weight)
    if not np.isfinite(raw).all() or not np.isfinite(fmt).all():
        raise RuntimeError("source-only normalization produced non-finite values")
    return raw, fmt
