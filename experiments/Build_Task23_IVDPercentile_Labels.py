"""Recompute trusted whole-field IVD labels for the Task2/Task3 sweep.

The existing source caches contain the exact valid primitive seeds and their
published p95 reference.  This script reloads the corresponding velocity
frame, recomputes IVD once, samples it at those same seeds, and refuses to
write any sweep label unless the recomputed p95 mask is bitwise identical to
the published source reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from Build_Task5_Multiscale_Cache import _channel_fields
from FMT_Utils.FMT_3D_pipeline import compute_ivd_reference_3d
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d


def percentile_tag(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p")
    return f"p{text}"


def all_percentiles(spec: dict) -> list[float]:
    values = [float(value) for value in spec["requested_percentiles"]]
    values.append(float(spec["audit_percentile"]))
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate IVD percentiles: {values}")
    if values != sorted(values):
        raise ValueError(f"IVD percentiles must be increasing: {values}")
    if any(not 0.0 < value < 100.0 for value in values):
        raise ValueError(f"invalid IVD percentile list: {values}")
    return values


def percentile_labels(ivd_volume: np.ndarray, ivd_at_seeds: np.ndarray,
                      percentiles: list[float]):
    """Return thresholds and nested seed masks for increasing percentiles."""
    finite = np.asarray(ivd_volume)[np.isfinite(ivd_volume)]
    if not len(finite):
        raise RuntimeError("IVD volume has no finite values")
    if list(percentiles) != sorted(percentiles):
        raise ValueError("percentiles must be increasing")
    thresholds = {
        value: float(np.percentile(finite, value)) for value in percentiles
    }
    masks = {
        value: np.asarray(ivd_at_seeds >= thresholds[value], dtype=bool)
        for value in percentiles
    }
    for lower, upper in zip(percentiles[:-1], percentiles[1:]):
        if thresholds[upper] < thresholds[lower]:
            raise RuntimeError("percentile thresholds are not monotonic")
        if np.any(masks[upper] & ~masks[lower]):
            raise RuntimeError(f"p{upper} labels are not a subset of p{lower}")
    return thresholds, masks


def _config_digest(spec: dict) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_records(source_dir: Path, expected_slices: int):
    paths = sorted(source_dir.glob("slice_*.npz"))
    if len(paths) != int(expected_slices):
        raise RuntimeError(
            f"{source_dir}: expected {expected_slices} slices, found {len(paths)}"
        )
    records = []
    for ordinal, path in enumerate(paths):
        with np.load(path) as cached:
            records.append({
                "ordinal": ordinal,
                "path": path,
                "seeds": np.asarray(cached["seeds"], dtype=np.float32),
                "published_p95": np.asarray(cached["reference"], dtype=bool),
                "metadata": json.loads(str(cached["metadata_json"])),
            })
    return records


def _netcdf_fields(records: list[dict], max_spatial_dim: int):
    for record in records:
        metadata = record["metadata"]
        path = Path(metadata["source_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        yield load_netcdf_window_3d(
            path,
            int(metadata["source_start_index"]),
            int(metadata["frame_count"]),
            int(max_spatial_dim),
        )


def _fields(spec: dict, dataset: str, records: list[dict]):
    if dataset != "channel":
        return _netcdf_fields(records, int(spec["max_spatial_dim"]))
    channel = spec["channel"]
    path = Path(channel["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    frame_counts = {int(record["metadata"]["frame_count"]) for record in records}
    if len(frame_counts) != 1:
        raise ValueError(f"channel frame counts differ: {sorted(frame_counts)}")
    helper_spec = {
        "sampling": {"max_spatial_dim": int(spec["max_spatial_dim"])},
        "channel_observer": {
            "total_frames": int(channel["total_frames"]),
            "duration": float(channel["duration"]),
            "output_crop_fraction": float(channel["output_crop_fraction"]),
        },
    }
    indices = [
        int(record["metadata"]["source_start_index"]) for record in records
    ]
    return _channel_fields(
        path, helper_spec, indices, frame_counts.pop()
    )


def _target_path(output_root: Path, percentile: float, group: str,
                 dataset: str, source_name: str) -> Path:
    return (
        output_root / "labels" / percentile_tag(percentile) / group / "labels"
        / dataset / source_name
    )


def _write_label(path: Path, labels: np.ndarray, ivd_at_seeds: np.ndarray,
                 threshold: float, metadata: dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        with np.load(path) as cached:
            old_labels = np.asarray(cached["labels"], dtype=bool)
            old_metadata = json.loads(str(cached["metadata_json"]))
        if not np.array_equal(old_labels, labels) or old_metadata != metadata:
            raise RuntimeError(f"stale incompatible label cache: {path}")
        return
    np.savez_compressed(
        path,
        labels=np.asarray(labels, dtype=bool),
        ivd_at_seeds=np.asarray(ivd_at_seeds, dtype=np.float32),
        ivd_threshold=np.float64(threshold),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _process_slice(spec: dict, output_root: Path, group: str, dataset: str,
                   record: dict, field, percentiles: list[float],
                   overwrite: bool) -> dict:
    ivd_volume, ivd_at_seeds, _ = compute_ivd_reference_3d(
        field, 0.0, record["seeds"]
    )
    thresholds, masks = percentile_labels(
        ivd_volume, ivd_at_seeds, percentiles
    )

    audit_p = float(spec["audit_percentile"])
    published = record["published_p95"]
    recomputed = masks[audit_p]
    mismatch_count = int(np.count_nonzero(published != recomputed))
    old_threshold = float(record["metadata"]["ivd_threshold"])
    threshold_error = abs(thresholds[audit_p] - old_threshold)
    threshold_ok = bool(np.isclose(
        thresholds[audit_p], old_threshold, rtol=1e-6, atol=1e-8
    ))
    if mismatch_count or not threshold_ok:
        raise RuntimeError(
            f"p95 audit failed for {record['path']}: mismatches={mismatch_count}, "
            f"new_threshold={thresholds[audit_p]:.17g}, "
            f"published_threshold={old_threshold:.17g}"
        )

    rows = {}
    for value in percentiles:
        labels = masks[value]
        metadata = {
            "experiment": spec["experiment"],
            "dataset": dataset,
            "source_group": group,
            "ordinal": int(record["ordinal"]),
            "source_cache": str(record["path"].resolve()),
            "source_start_index": int(record["metadata"]["source_start_index"]),
            "label_mode": "standard_global_ivd_percentile",
            "label_value": float(value),
            "ivd_threshold": thresholds[value],
            "sample_count": int(len(labels)),
            "positive_count": int(labels.sum()),
            "positive_fraction": float(labels.mean()),
            "recomputed_from_velocity_field": True,
            "p95_reference_bitwise_equal": True,
            "config_sha256": _config_digest(spec),
        }
        target = _target_path(
            output_root, value, group, dataset, record["path"].name
        )
        _write_label(
            target, labels, ivd_at_seeds, thresholds[value], metadata, overwrite
        )
        rows[percentile_tag(value)] = {
            "threshold": thresholds[value],
            "positive_count": int(labels.sum()),
            "positive_fraction": float(labels.mean()),
            "path": str(target),
        }
    return {
        "dataset": dataset,
        "ordinal": int(record["ordinal"]),
        "source_cache": str(record["path"]),
        "sample_count": int(len(published)),
        "p95_mismatch_count": mismatch_count,
        "p95_threshold_absolute_error": threshold_error,
        "percentiles": rows,
    }


def build_group(config_path: str, group: str, overwrite: bool = False) -> Path:
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if group not in spec["label_groups"]:
        raise ValueError(f"unknown label group {group!r}")
    percentiles = all_percentiles(spec)
    group_spec = spec["label_groups"][group]
    source_root = Path(group_spec["source_cache_root"])
    output_root = Path(spec["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot = output_root / "label_config_snapshot.yaml"
    if snapshot.exists():
        old = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if old != spec:
            raise RuntimeError(f"configuration changed for {output_root}")
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    manifest = {
        "experiment": spec["experiment"],
        "group": group,
        "source_cache_root": str(source_root),
        "percentiles": percentiles,
        "p95_audit_required": True,
        "slices": [],
    }
    for dataset in group_spec["datasets"]:
        records = _source_records(
            source_root / dataset, int(group_spec["expected_slices"])
        )
        generated_fields = _fields(spec, dataset, records)
        count = 0
        for record, (field, _) in zip(records, generated_fields):
            row = _process_slice(
                spec, output_root, group, dataset, record, field,
                percentiles, overwrite,
            )
            manifest["slices"].append(row)
            count += 1
            positives = ", ".join(
                f"{tag}={values['positive_fraction']:.2%}"
                for tag, values in row["percentiles"].items()
            )
            print(
                f"[{group}/{dataset}] {count}/{len(records)} p95 audit exact; "
                f"{positives}", flush=True,
            )
            del field
        if count != len(records):
            raise RuntimeError(
                f"field generator ended early for {group}/{dataset}: "
                f"{count} != {len(records)}"
            )
    manifest["slice_count"] = len(manifest["slices"])
    manifest["p95_total_mismatch_count"] = int(sum(
        row["p95_mismatch_count"] for row in manifest["slices"]
    ))
    manifest_path = output_root / f"label_manifest_{group}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}", flush=True)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Ablation_Task23IVDPercentile_1.1.yaml"
    )
    parser.add_argument("--group", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_group(args.config, args.group, args.overwrite)
