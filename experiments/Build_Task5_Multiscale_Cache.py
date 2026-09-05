"""Build exact variable-scale pathline caches for 3D Task5.

The scale assignment is deterministic, spatially shuffled, and balanced.
Labels remain the frozen whole-field IVD-p95 definition used by Task1--Task3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from Build_Channel_Killing_Cache import load_channel_vtk
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_3D_pipeline import (
    compute_ivd_reference_3d,
    generate_seeding_grid_3d,
)
from FMT_Utils.KillingObserver3D import (
    compose_steady_to_unsteady,
    integrate_killing_frame,
    smooth_channel_observer,
)
from FMT_Utils.MultiscalePathline_3D import (
    balanced_scale_assignment,
    integrate_multiscale_primitives_3d,
    maximum_horizon_in_source_frames,
    maximum_offset_grid_scale,
    parse_scale_table,
    positive_grid_intervals,
)
from FMT_Utils.NetCDF_window_3D import (
    inspect_netcdf_3d,
    load_netcdf_window_3d,
)
from FLowUtils.VectorField3d import UnsteadyVectorField3D


def _resolve_dataset(spec, dataset_id):
    matches = [item for item in spec["datasets"] if str(item["id"]) == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset_id!r} matched {len(matches)} entries")
    item = matches[0]
    candidates = [Path(value) for value in item.get("paths", [item.get("path")]) if value]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"none of the configured paths exists for {dataset_id}: {candidates}"
        )
    return item, path


def _assignment_seed(base_seed, dataset, phase, source_index, scale_set):
    identity = f"{base_seed}|{dataset}|{phase}|{source_index}|{scale_set}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")


def _config_digest(spec):
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _scale_set_for_ordinal(phase_spec, ordinal):
    matches = [
        name for name, ordinals in phase_spec["ordinal_scale_sets"].items()
        if int(ordinal) in {int(value) for value in ordinals}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"ordinal {ordinal} must belong to exactly one scale set, got {matches}"
        )
    return matches[0]


def _device(spec):
    requested = str(spec["cache_generation"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _raw_local_features(primitives):
    xyz = np.asarray(primitives[..., :3], dtype=np.float32)
    return (xyz - xyz[:, :1, :1, :]).reshape(len(xyz), -1)


def _encode_slice(
    spec,
    dataset,
    phase,
    ordinal,
    source_index,
    field,
    load_metadata,
    scale_set_name,
    scales,
    output_path,
    device,
):
    started = time.time()
    spacing = positive_grid_intervals(field)
    offset_mode = str(spec["pathlines"].get("offset_mode", "min"))
    offset_base = {
        "min": float(spacing.min()),
        "geometric_mean": float(np.prod(spacing) ** (1.0 / 3.0)),
        "max": float(spacing.max()),
    }.get(offset_mode)
    if offset_base is None:
        raise ValueError("pathlines.offset_mode must be min, geometric_mean, or max")
    maximum_offset = offset_base * maximum_offset_grid_scale(scales)
    seeds, _ = generate_seeding_grid_3d(
        field,
        spec["sampling"]["seed_grid_shape"],
        float(spec["sampling"]["boundary_fraction"]),
        maximum_offset,
    )
    assignment_seed = _assignment_seed(
        spec["seed"], dataset, phase, source_index, scale_set_name
    )
    assignment = balanced_scale_assignment(len(seeds), len(scales), assignment_seed)
    result = integrate_multiscale_primitives_3d(
        field,
        seeds,
        0.0,
        scales,
        assignment,
        int(spec["pathlines"]["sampled_steps"]),
        offset_mode=offset_mode,
        method=str(spec["pathlines"]["method"]),
        chunk_size=int(spec["pathlines"]["chunk_size"]),
    )
    valid_mask = result["valid_mask"]
    seeds_valid = seeds[valid_mask]
    primitives = result["primitives"]
    minimum_valid = int(spec["cache_generation"].get("minimum_valid_per_scale", 20))
    valid_counts = np.bincount(result["scale_id"], minlength=len(scales))
    if len(seeds_valid) < 100 or np.any(valid_counts < minimum_valid):
        raise RuntimeError(
            f"{dataset} {phase} slice {source_index}: valid={len(seeds_valid)}, "
            f"per-scale={valid_counts.tolist()}"
        )

    raw_features = _raw_local_features(primitives)
    fmt_features = pathline_dft_features_3d(
        torch.from_numpy(primitives).to(device),
        num_freq=int(spec["encoder"]["num_freq"]),
        neighbor_weight=1.0,
        neighbor_scale=1.0,
        neighbor_pool=str(spec["encoder"]["neighbor_pool"]),
        mode=str(spec["encoder"]["mode"]),
        include_chirality=bool(spec["encoder"]["include_chirality"]),
    ).astype(np.float32)
    ivd_volume, ivd_at_seeds, _ = compute_ivd_reference_3d(field, 0.0, seeds_valid)
    finite_ivd = ivd_volume[np.isfinite(ivd_volume)]
    threshold = float(np.percentile(finite_ivd, float(spec["reference"]["percentile"])))
    reference = np.asarray(ivd_at_seeds >= threshold, dtype=bool)

    assigned_counts = np.bincount(assignment, minlength=len(scales))
    metadata = {
        "experiment": str(spec["experiment"]),
        "config_sha256": _config_digest(spec),
        "task": "Task5",
        "dataset": dataset,
        "phase": phase,
        "ordinal": int(ordinal),
        "source_start_index": int(source_index),
        **load_metadata,
        "scale_set": scale_set_name,
        "scale_assignment_seed": int(assignment_seed),
        "scale_table": [scale.__dict__ for scale in scales],
        "assigned_count_by_scale": assigned_counts.tolist(),
        "valid_count_by_scale": valid_counts.tolist(),
        "valid_primitives": int(len(seeds_valid)),
        "total_primitives": int(len(seeds)),
        "sampled_steps": int(spec["pathlines"]["sampled_steps"]),
        "ivd_definition": "whole_field_ivd_percentile",
        "ivd_percentile": float(spec["reference"]["percentile"]),
        "ivd_threshold": threshold,
        "ivd_positive_count": int(reference.sum()),
        "ivd_positive_fraction": float(reference.mean()),
        "elapsed_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        raw_features=raw_features,
        fmt_features=fmt_features,
        seeds=seeds_valid.astype(np.float32),
        reference=reference,
        valid_mask=valid_mask,
        line_lengths=result["line_lengths"],
        scale_id=result["scale_id"],
        offset_grid_scale=result["offset_grid_scale"],
        dt_scale=result["dt_scale"],
        integration_steps=result["integration_steps"],
        primitive_offset=result["primitive_offset"],
        physical_dt=result["physical_dt"],
        integration_horizon=result["integration_horizon"],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"[{dataset}] {phase} {ordinal + 1}: index={source_index}, "
        f"scale_set={scale_set_name}, valid={len(seeds_valid)}/{len(seeds)}, "
        f"positive={reference.mean():.3%}, elapsed={metadata['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return metadata


def _netcdf_fields(path, indices, frame_count, max_spatial_dim):
    info = inspect_netcdf_3d(path)
    total = int(info["shape"]["t"])
    for source_index in indices:
        if int(source_index) < 0 or int(source_index) + frame_count > total:
            raise ValueError(
                f"source window [{source_index},{int(source_index) + frame_count}) "
                f"exceeds T={total} for {path}"
            )
        field, metadata = load_netcdf_window_3d(
            path, int(source_index), frame_count, int(max_spatial_dim)
        )
        yield field, metadata


def _channel_fields(path, spec, indices, frame_count):
    channel = spec["channel_observer"]
    interpolator, points, axes, source_min, source_max, vtk_metadata = load_channel_vtk(
        path,
        int(spec["sampling"]["max_spatial_dim"]),
        float(channel["output_crop_fraction"]),
    )
    total_frames = int(channel["total_frames"])
    duration = float(channel["duration"])
    times = np.linspace(0.0, duration, total_frames)
    source_dt = float(times[1] - times[0])
    parameters = smooth_channel_observer(times / duration, source_min, source_max)
    parameters /= duration
    rotation, displacement = integrate_killing_frame(parameters, source_dt)
    ox, oy, oz = axes
    shape_zyx = (len(oz), len(oy), len(ox))
    dmin = np.asarray([ox[0], oy[0], oz[0]], dtype=np.float32)
    dmax = np.asarray([ox[-1], oy[-1], oz[-1]], dtype=np.float32)
    for source_index in indices:
        source_index = int(source_index)
        if source_index < 0 or source_index + frame_count > total_frames:
            raise ValueError("channel observer window exceeds generated frame range")
        selection = slice(source_index, source_index + frame_count)
        flat = compose_steady_to_unsteady(
            points,
            interpolator,
            parameters[selection],
            rotation[selection],
            displacement[selection],
            bounds_min=source_min,
            bounds_max=source_max,
        )
        field_data = flat.reshape(frame_count, *shape_zyx, 3)
        field = UnsteadyVectorField3D(
            len(ox), len(oy), len(oz), frame_count,
            dmin, dmax, 0.0, source_dt * (frame_count - 1),
        )
        field.field = np.ascontiguousarray(field_data, dtype=np.float32)
        metadata = {
            "source_path": str(path.resolve()),
            "source_start_index": source_index,
            "source_time": float(times[source_index]),
            "source_time_step": source_dt,
            "frame_count": frame_count,
            "loaded_shape_TZYXC": list(field_data.shape),
            "generated_unsteady": "killing_observer_pushforward",
            "vtk": vtk_metadata,
        }
        yield field, metadata


def build(config_path, phase, dataset, overwrite=False):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if phase not in spec["phases"]:
        raise ValueError(f"unknown phase {phase!r}")
    item, path = _resolve_dataset(spec, dataset)
    phase_spec = spec["phases"][phase]
    indices = [int(value) for value in phase_spec["time_indices_by_dataset"][dataset]]
    covered = sorted(
        int(value)
        for ordinals in phase_spec["ordinal_scale_sets"].values()
        for value in ordinals
    )
    if covered != list(range(len(indices))):
        raise ValueError(f"{phase} ordinal scale sets do not cover {len(indices)} slices")

    all_scale_names = set(phase_spec["ordinal_scale_sets"])
    parsed = {
        name: parse_scale_table(
            spec["scale_sets"][name], int(spec["pathlines"]["sampled_steps"])
        )
        for name in all_scale_names
    }
    maximum_horizon = max(maximum_horizon_in_source_frames(value) for value in parsed.values())
    frame_count = int(np.ceil(maximum_horizon)) + 2
    output_dir = Path(spec["cache_roots"][phase]) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    config_digest = _config_digest(spec)
    snapshot = output_dir / "config_snapshot.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if _config_digest(previous) != config_digest:
            raise RuntimeError(
                f"configuration changed for existing cache {output_dir}; "
                "use a new experiment version or --overwrite after review"
            )
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    device = _device(spec)
    if str(item.get("kind", "netcdf")) == "channel_vtk":
        fields = _channel_fields(path, spec, indices, frame_count)
    else:
        fields = _netcdf_fields(
            path, indices, frame_count, int(spec["sampling"]["max_spatial_dim"])
        )

    manifest = {
        "experiment": spec["experiment"],
        "task": "Task5",
        "phase": phase,
        "dataset": dataset,
        "source": str(path.resolve()),
        "selected_time_indices": indices,
        "maximum_horizon_in_source_frames": maximum_horizon,
        "slices": [],
    }
    for ordinal, (source_index, field_and_metadata) in enumerate(zip(indices, fields)):
        field, load_metadata = field_and_metadata
        output_path = output_dir / f"slice_{ordinal:02d}_index_{source_index:04d}.npz"
        if output_path.exists() and not overwrite:
            with np.load(output_path) as cached:
                metadata = json.loads(str(cached["metadata_json"]))
            if (
                metadata["experiment"] != spec["experiment"]
                or metadata["phase"] != phase
                or metadata.get("config_sha256") != config_digest
            ):
                raise RuntimeError(f"stale incompatible cache: {output_path}")
            print(f"[{dataset}] {phase} cached: {output_path.name}", flush=True)
        else:
            scale_set = _scale_set_for_ordinal(phase_spec, ordinal)
            metadata = _encode_slice(
                spec,
                dataset,
                phase,
                ordinal,
                source_index,
                field,
                load_metadata,
                scale_set,
                parsed[scale_set],
                output_path,
                device,
            )
        manifest["slices"].append(metadata)
        del field
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mainExp_Task5_3D_1.1.yaml")
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.config, args.phase, args.dataset, args.overwrite)
