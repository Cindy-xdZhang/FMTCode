"""Create Task2 cache from steady channel VTK under a time-varying Killing observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

from DeepUtils.utils import EasyConfig
from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_3D_pipeline import (
    compute_ivd_reference_3d, generate_seeding_grid_3d, integrate_cross_primitives_3d,
)
from FMT_Utils.KillingObserver3D import (
    compose_steady_to_unsteady, integrate_killing_frame, smooth_channel_observer,
)
from FMT_Utils.NetCDF_window_3D import resolve_time_indices


def load_channel_vtk(path, max_spatial_dim, crop_fraction):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkStructuredGridReader(); reader.SetFileName(str(path))
    reader.ReadAllVectorsOn(); reader.Update(); grid = reader.GetOutput()
    dimensions = [0, 0, 0]; grid.GetDimensions(dimensions)
    x_count, y_count, z_count = dimensions
    points = vtk_to_numpy(grid.GetPoints().GetData()).reshape(z_count, y_count, x_count, 3)
    velocity_array = grid.GetPointData().GetArray("velocity")
    if velocity_array is None or velocity_array.GetNumberOfComponents() != 3:
        raise ValueError("channel VTK misses 3-component point array 'velocity'")
    velocity = vtk_to_numpy(velocity_array).reshape(z_count, y_count, x_count, 3)
    xs = np.asarray(points[0, 0, :, 0], dtype=np.float64)
    ys = np.asarray(points[0, :, 0, 1], dtype=np.float64)
    zs = np.asarray(points[:, 0, 0, 2], dtype=np.float64)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    expected = np.stack((xx, yy, zz), axis=-1)
    if not np.allclose(points, expected, rtol=0, atol=1e-6):
        raise ValueError("channel structured grid is not separable into x/y/z axes")
    grid_interpolator = RegularGridInterpolator(
        (zs, ys, xs), np.asarray(velocity, dtype=np.float32), bounds_error=True
    )
    def interpolator(points_xyz):
        return grid_interpolator(np.asarray(points_xyz)[:, [2, 1, 0]])
    lower = np.array([xs[0], ys[0], zs[0]], dtype=np.float64)
    upper = np.array([xs[-1], ys[-1], zs[-1]], dtype=np.float64)
    cropped_lower = lower + float(crop_fraction) * (upper - lower)
    cropped_upper = upper - float(crop_fraction) * (upper - lower)
    strides = np.maximum(1, np.ceil(np.array(dimensions) / int(max_spatial_dim)).astype(int))
    counts = np.ceil(np.array(dimensions) / strides).astype(int)
    output_axes = tuple(np.linspace(cropped_lower[i], cropped_upper[i], counts[i]) for i in range(3))
    ox, oy, oz = output_axes
    out_z, out_y, out_x = np.meshgrid(oz, oy, ox, indexing="ij")
    output_points = np.stack((out_x.ravel(), out_y.ravel(), out_z.ravel()), axis=-1)
    metadata = {"source_dimensions_xyz": dimensions, "source_bounds": [lower.tolist(), upper.tolist()],
                "output_counts_xyz": counts.tolist(), "output_bounds": [cropped_lower.tolist(),
                cropped_upper.tolist()], "source_z_is_uniform": bool(np.allclose(np.diff(zs),
                np.diff(zs).mean(), rtol=1e-4, atol=1e-8))}
    return interpolator, output_points, output_axes, lower, upper, metadata


def build(config, overwrite=False):
    entry = next(item for item in config.datasets if item["id"] == "channel")
    path = Path(entry["path"])
    output_dir = Path(config.output.cache_dir) / "channel"; output_dir.mkdir(parents=True, exist_ok=True)
    interpolator, points, axes, source_min, source_max, vtk_metadata = load_channel_vtk(
        path, int(config.sampling.max_spatial_dim),
        float(config.channel_observer.output_crop_fraction),
    )
    total_frames = int(config.channel_observer.total_frames)
    duration = float(config.channel_observer.duration)
    times = np.linspace(0.0, duration, total_frames); dt_source = float(times[1] - times[0])
    parameters = smooth_channel_observer(times / duration, source_min, source_max)
    # smooth_channel_observer differentiates normalized time; convert q to physical-time derivative.
    parameters /= duration
    rotation, displacement = integrate_killing_frame(parameters, dt_source)
    future_intervals = int(np.ceil(
        float(config.pathlines.dt_scale) * int(config.pathlines.integration_steps)
    ))
    frame_count = future_intervals + 2
    fixed_by_dataset = getattr(
        config.sampling, "fixed_time_indices_by_dataset", None
    )
    fixed_indices = (
        fixed_by_dataset.get("channel") if fixed_by_dataset is not None
        else getattr(config.sampling, "fixed_time_indices", None)
    )
    original_fixed_by_dataset = getattr(
        config.sampling, "original_fixed_time_indices_by_dataset", None
    )
    original_fixed_indices = (
        original_fixed_by_dataset.get("channel")
        if original_fixed_by_dataset is not None else None
    )
    indices = resolve_time_indices(
        total_frames, int(config.sampling.timeslices), float(config.sampling.begin_fraction),
        float(config.sampling.end_fraction), required_future_frames=frame_count - 1,
        fixed_indices=fixed_indices,
    )
    ox, oy, oz = axes; shape_zyx = (len(oz), len(oy), len(ox))
    dmin = np.array([ox[0], oy[0], oz[0]], dtype=np.float32)
    dmax = np.array([ox[-1], oy[-1], oz[-1]], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {"experiment": str(config.experiment), "dataset": "channel",
                "source": str(path.resolve()), "vtk": vtk_metadata,
                "observer_formula": "xi=R^T*x-D; v=R*s(xi)+t_vec+w_cross_x",
                "observer_parameters": parameters.tolist(),
                "selected_time_indices": indices.tolist(), "slices": []}
    for ordinal, source_index in enumerate(indices):
        output_path = output_dir / f"slice_{ordinal:02d}_index_{source_index:04d}.npz"
        if output_path.exists() and not overwrite:
            with np.load(output_path) as cached:
                metadata = json.loads(str(cached["metadata_json"]))
            manifest["slices"].append(metadata); print(f"[channel] cached {output_path.name}")
            continue
        started = time.time(); selection = slice(source_index, source_index + frame_count)
        flat_field = compose_steady_to_unsteady(
            points, interpolator, parameters[selection], rotation[selection], displacement[selection],
            bounds_min=source_min, bounds_max=source_max,
        )
        field_data = flat_field.reshape(frame_count, *shape_zyx, 3)
        field = UnsteadyVectorField3D(len(ox), len(oy), len(oz), frame_count,
                                      dmin, dmax, 0.0, dt_source * (frame_count - 1))
        field.field = np.ascontiguousarray(field_data, dtype=np.float32)
        offset = float(np.min(field.gridInterval[field.gridInterval > 0])) * float(
            config.pathlines.offset_grid_scale
        )
        grid_phase = getattr(config.sampling, "seed_grid_phase", None)
        seeds, _ = generate_seeding_grid_3d(
            field, config.sampling.seed_grid_shape,
            float(config.sampling.boundary_fraction), offset,
            grid_phase=grid_phase,
        )
        primitives, valid_mask, lengths = integrate_cross_primitives_3d(
            field, seeds, 0.0, dt_source * float(config.pathlines.dt_scale),
            int(config.pathlines.integration_steps), int(config.pathlines.sampled_steps), offset,
            method=str(config.pathlines.method), chunk_size=int(config.pathlines.chunk_size),
        )
        seeds_valid = seeds[valid_mask]
        xyz = primitives[..., :3]
        raw_features = (xyz - xyz[:, :1, :1, :]).reshape(len(xyz), -1).astype(np.float32)
        fmt_features = pathline_dft_features_3d(
            torch.from_numpy(primitives).to(device), num_freq=int(config.encoder.num_freq),
            neighbor_weight=1.0, neighbor_scale=1.0,
            neighbor_pool=str(config.encoder.neighbor_pool), mode=str(config.encoder.mode),
            include_chirality=bool(config.encoder.include_chirality),
        ).astype(np.float32)
        ivd_volume, ivd_at_seeds, _ = compute_ivd_reference_3d(field, 0.0, seeds_valid)
        threshold = float(np.percentile(ivd_volume, float(config.reference.percentile)))
        reference = ivd_at_seeds >= threshold
        metadata = {"dataset": "channel", "ordinal": ordinal,
                    "source_start_index": int(source_index), "source_time": float(times[source_index]),
                    "source_time_step": dt_source, "frame_count": frame_count,
                    "seed_grid_phase": (
                        None if grid_phase is None
                        else [float(value) for value in grid_phase]
                    ),
                    "loaded_shape_TZYXC": list(field_data.shape),
                    "valid_primitives": int(len(seeds_valid)), "total_primitives": int(len(seeds)),
                    "ivd_threshold": threshold, "ivd_positive_count": int(reference.sum()),
                    "ivd_positive_fraction": float(reference.mean()),
                    "elapsed_seconds": time.time() - started}
        if original_fixed_indices is not None:
            metadata["original_source_start_index"] = int(
                original_fixed_indices[ordinal]
            )
            metadata["source_staging_manifest"] = str(
                config.sampling.source_staging_manifest
            )
            metadata["source_staging_manifest_sha256"] = str(
                config.sampling.source_staging_manifest_sha256
            )
        np.savez_compressed(output_path, raw_features=raw_features, fmt_features=fmt_features,
                            seeds=seeds_valid.astype(np.float32), reference=reference,
                            valid_mask=valid_mask, line_lengths=lengths,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        manifest["slices"].append(metadata)
        print(f"[channel] {ordinal + 1}/{len(indices)} index={source_index}: "
              f"valid={len(seeds_valid)}/{len(seeds)}, positives={reference.sum()}, "
              f"elapsed={metadata['elapsed_seconds']:.1f}s")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_Task2Universality_1.1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); build(EasyConfig(args.config), args.overwrite)
