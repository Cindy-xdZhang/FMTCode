"""Build local-IVD Task3 labels for every frozen 3D universality cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from Build_Channel_Killing_Cache import load_channel_vtk
from DeepUtils.utils import EasyConfig
from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FMT_Utils.FMT_3D_pipeline import compute_ivd_reference_3d
from FMT_Utils.KillingObserver3D import (
    compose_steady_to_unsteady, integrate_killing_frame, smooth_channel_observer,
)
from FMT_Utils.LocalIVDLabel_3D import sample_local_ivd_labels_3d
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d


def _config_identity(spec):
    return json.loads(json.dumps(spec, sort_keys=True))


def _validate_snapshot(spec, output_root):
    snapshot = output_root / "config_snapshot.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if _config_identity(previous) != _config_identity(spec):
            raise RuntimeError(
                f"configuration changed for {output_root}; use a new experiment version"
            )
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _channel_context(source_config):
    entry = next(item for item in source_config.datasets if item["id"] == "channel")
    path = Path(entry["path"])
    interpolator, points, axes, source_min, source_max, vtk_metadata = load_channel_vtk(
        path, int(source_config.sampling.max_spatial_dim),
        float(source_config.channel_observer.output_crop_fraction),
    )
    total_frames = int(source_config.channel_observer.total_frames)
    duration = float(source_config.channel_observer.duration)
    times = np.linspace(0.0, duration, total_frames)
    dt = float(times[1] - times[0])
    parameters = smooth_channel_observer(times / duration, source_min, source_max)
    parameters /= duration
    rotation, displacement = integrate_killing_frame(parameters, dt)
    return {
        "interpolator": interpolator, "points": points, "axes": axes,
        "source_min": source_min, "source_max": source_max,
        "parameters": parameters, "rotation": rotation,
        "displacement": displacement, "time_step": dt,
        "vtk_metadata": vtk_metadata, "source_path": str(path.resolve()),
    }


def _channel_field(context, source_index, frame_count=2):
    begin = int(source_index); end = begin + int(frame_count)
    flat = compose_steady_to_unsteady(
        context["points"], context["interpolator"],
        context["parameters"][begin:end], context["rotation"][begin:end],
        context["displacement"][begin:end], bounds_min=context["source_min"],
        bounds_max=context["source_max"],
    )
    ox, oy, oz = context["axes"]
    shape = (len(oz), len(oy), len(ox))
    values = flat.reshape(frame_count, *shape, 3)
    dmin = np.asarray([ox[0], oy[0], oz[0]], dtype=np.float32)
    dmax = np.asarray([ox[-1], oy[-1], oz[-1]], dtype=np.float32)
    dt = float(context["time_step"])
    field = UnsteadyVectorField3D(
        len(ox), len(oy), len(oz), frame_count, dmin, dmax, 0.0,
        dt * (frame_count - 1),
    )
    field.field = np.ascontiguousarray(values, dtype=np.float32)
    return field


def _source_field(dataset, metadata, source_config, channel_context):
    source_index = int(metadata["source_start_index"])
    if dataset == "channel":
        return _channel_field(channel_context, source_index, 2)
    path = metadata.get("source_path")
    if not path:
        entry = next(item for item in source_config.datasets if item["id"] == dataset)
        path = entry["path"]
    field, _ = load_netcdf_window_3d(
        path, source_index, 2, int(source_config.sampling.max_spatial_dim)
    )
    return field


def build(config_path="config/Verify_Task3Universality_1.1.yaml", overwrite=False):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source_config = EasyConfig(spec["source_config"])
    cache_root = Path(spec["source_cache_root"])
    output_root = Path(spec["output_dir"]) / "labels"
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_snapshot(spec, output_root)
    channel_context = _channel_context(source_config) if "channel" in spec["datasets"] else None
    manifest = {"experiment": spec["experiment"], "config": spec, "datasets": {}}
    for dataset in spec["datasets"]:
        source_paths = sorted((cache_root / dataset).glob("slice_*.npz"))
        expected_slices = int(source_config.sampling.timeslices)
        if len(source_paths) != expected_slices:
            raise RuntimeError(
                f"{dataset}: expected {expected_slices} source slices, "
                f"found {len(source_paths)}"
            )
        target_dir = output_root / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest["datasets"][dataset] = []
        for ordinal, source_path in enumerate(source_paths):
            target = target_dir / source_path.name
            if target.exists() and not overwrite:
                with np.load(target) as data:
                    metadata = json.loads(str(data["metadata_json"]))
                if Path(metadata["source_cache"]).name != source_path.name:
                    raise RuntimeError(f"stale label cache: {target}")
                manifest["datasets"][dataset].append(metadata)
                continue
            started = time.perf_counter()
            with np.load(source_path) as data:
                seeds = np.asarray(data["seeds"], dtype=np.float32)
                source_metadata = json.loads(str(data["metadata_json"]))
            field = _source_field(dataset, source_metadata, source_config, channel_context)
            ivd_volume, _, axes = compute_ivd_reference_3d(field, 0.0, seeds)
            labels, ivd, thresholds = sample_local_ivd_labels_3d(
                ivd_volume, axes, seeds,
                int(spec["label"]["neighborhood_size"]),
                str(spec["label"]["mode"]), float(spec["label"]["value"]),
            )
            metadata = {
                "dataset": dataset, "ordinal": ordinal,
                "source_cache": str(source_path.resolve()),
                "source_start_index": int(source_metadata["source_start_index"]),
                "label_mode": spec["label"]["mode"],
                "label_value": float(spec["label"]["value"]),
                "neighborhood_size": int(spec["label"]["neighborhood_size"]),
                "boundary_mode": "nearest",
                "sample_count": int(len(labels)),
                "positive_count": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "ivd_volume_shape_zyx": list(ivd_volume.shape),
                "elapsed_seconds": time.perf_counter() - started,
            }
            np.savez_compressed(
                target, labels=labels.astype(bool), ivd_at_seeds=ivd,
                threshold_at_seeds=thresholds,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            manifest["datasets"][dataset].append(metadata)
            print(
                f"{dataset} {ordinal + 1}/{expected_slices}: n={len(labels)}, "
                f"positive={labels.mean():.3%}, elapsed={metadata['elapsed_seconds']:.1f}s",
                flush=True,
            )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_Task3Universality_1.1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.config, args.overwrite)
