"""Build resumable 10-timeslice raw/FMT/IVD caches for Task2 universality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from DeepUtils.utils import EasyConfig
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_3D_pipeline import (
    compute_ivd_reference_3d, generate_seeding_grid_3d, integrate_cross_primitives_3d,
)
from FMT_Utils.NetCDF_window_3D import (
    inspect_netcdf_3d, load_netcdf_window_3d, resolve_time_indices,
)


def resolve_primitive_offset(grid_interval, scale, mode="min"):
    spacing = np.asarray(grid_interval, dtype=np.float64)
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    if len(spacing) != 3:
        raise ValueError(f"expected three positive spatial intervals, got {grid_interval}")
    if mode == "min":
        base = spacing.min()
    elif mode == "geometric_mean":
        base = float(np.prod(spacing) ** (1.0 / len(spacing)))
    elif mode == "max":
        base = spacing.max()
    else:
        raise ValueError("pathlines.offset_mode must be min, geometric_mean, or max")
    offset = float(base) * float(scale)
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError("primitive offset must be finite and positive")
    return offset


def _dataset(config, dataset_id):
    entries = []
    for item in config.datasets:
        if isinstance(item, dict):
            converted = EasyConfig(); converted.update(item); entries.append(converted)
        else:
            entries.append(item)
    matches = [item for item in entries if str(item.id) == str(dataset_id)]
    if len(matches) != 1:
        raise ValueError(f"dataset id {dataset_id!r} matched {len(matches)} entries")
    item = matches[0]
    if not bool(item.enabled):
        raise ValueError(f"dataset {dataset_id} is disabled: {item.reason}")
    path = Path(item.path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".nc":
        raise ValueError("universality cache currently requires unsteady NetCDF input")
    return item, path


def _raw_local_features(primitives):
    xyz = np.asarray(primitives[..., :3], dtype=np.float32)
    return (xyz - xyz[:, :1, :1, :]).reshape(len(xyz), -1)


def build_dataset(config, dataset_id, overwrite=False):
    item, path = _dataset(config, dataset_id)
    info = inspect_netcdf_3d(path)
    future_intervals = int(np.ceil(
        float(config.pathlines.dt_scale) * int(config.pathlines.integration_steps)
    ))
    frame_count = future_intervals + 2
    fixed_by_dataset = getattr(
        config.sampling, "fixed_time_indices_by_dataset", None
    )
    fixed_indices = (
        fixed_by_dataset.get(str(item.id)) if fixed_by_dataset is not None
        else getattr(config.sampling, "fixed_time_indices", None)
    )
    indices = resolve_time_indices(
        info["shape"]["t"], int(config.sampling.timeslices),
        float(config.sampling.begin_fraction), float(config.sampling.end_fraction),
        required_future_frames=frame_count - 1,
        fixed_indices=fixed_indices,
    )
    output_dir = Path(config.output.cache_dir) / str(item.id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": str(config.experiment), "dataset": str(item.id),
                "source": info, "selected_time_indices": indices.tolist(), "slices": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ordinal, source_index in enumerate(indices):
        output_path = output_dir / f"slice_{ordinal:02d}_index_{source_index:04d}.npz"
        if output_path.exists() and not overwrite:
            with np.load(output_path) as cached:
                metadata = json.loads(str(cached["metadata_json"]))
            manifest["slices"].append(metadata)
            print(
                f"[{item.id}] {ordinal + 1}/{len(indices)} cached: "
                f"{output_path.name}"
            )
            continue
        started = time.time()
        field, load_metadata = load_netcdf_window_3d(
            path, int(source_index), frame_count, int(config.sampling.max_spatial_dim)
        )
        offset = resolve_primitive_offset(
            field.gridInterval, float(config.pathlines.offset_grid_scale),
            str(getattr(config.pathlines, "offset_mode", "min")),
        )
        seeds, _ = generate_seeding_grid_3d(
            field, config.sampling.seed_grid_shape,
            float(config.sampling.boundary_fraction), offset,
        )
        dt = float(field.timeInterval) * float(config.pathlines.dt_scale)
        primitives, valid_mask, lengths = integrate_cross_primitives_3d(
            field, seeds, 0.0, dt, int(config.pathlines.integration_steps),
            int(config.pathlines.sampled_steps), offset,
            method=str(config.pathlines.method), chunk_size=int(config.pathlines.chunk_size),
        )
        seeds_valid = seeds[valid_mask]
        if len(seeds_valid) < 100:
            raise RuntimeError(f"{item.id} slice {source_index}: only {len(seeds_valid)} valid primitives")
        raw_features = _raw_local_features(primitives)
        fmt_features = pathline_dft_features_3d(
            torch.from_numpy(primitives).to(device),
            num_freq=int(config.encoder.num_freq), neighbor_weight=1.0,
            neighbor_scale=1.0, neighbor_pool=str(config.encoder.neighbor_pool),
            mode=str(config.encoder.mode), include_chirality=bool(config.encoder.include_chirality),
        ).astype(np.float32)
        # Match mainExp: weighting is applied after StandardScaler during training.
        ivd_volume, ivd_at_seeds, _ = compute_ivd_reference_3d(field, 0.0, seeds_valid)
        threshold = float(np.percentile(ivd_volume[np.isfinite(ivd_volume)],
                                        float(config.reference.percentile)))
        reference = ivd_at_seeds >= threshold
        metadata = {
            "dataset": str(item.id), "ordinal": ordinal,
            **load_metadata, "valid_primitives": int(len(seeds_valid)),
            "total_primitives": int(len(seeds)), "ivd_threshold": threshold,
            "ivd_positive_count": int(reference.sum()),
            "ivd_positive_fraction": float(reference.mean()),
            "primitive_offset": offset,
            "primitive_offset_mode": str(getattr(config.pathlines, "offset_mode", "min")),
            "elapsed_seconds": time.time() - started,
        }
        np.savez_compressed(
            output_path, raw_features=raw_features, fmt_features=fmt_features,
            seeds=seeds_valid.astype(np.float32), reference=reference,
            valid_mask=valid_mask, line_lengths=lengths,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        manifest["slices"].append(metadata)
        print(f"[{item.id}] {ordinal + 1}/{len(indices)} index={source_index}: "
              f"valid={len(seeds_valid)}/{len(seeds)}, positives={reference.sum()}, "
              f"elapsed={metadata['elapsed_seconds']:.1f}s")
        del field, primitives, raw_features, fmt_features, ivd_volume
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_Task2Universality_1.1.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(EasyConfig(args.config), args.dataset, args.overwrite)
