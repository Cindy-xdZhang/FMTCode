"""Fail-closed integrity audit for Task5 variable-scale caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def _digest(spec):
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def audit(config_path):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    expected_hash = _digest(spec)
    datasets = [str(item["id"]) for item in spec["datasets"]]
    report = {
        "experiment": spec["experiment"],
        "config_sha256": expected_hash,
        "fixed_primitive_shape": [7, int(spec["pathlines"]["sampled_steps"]), 3],
        "phases": {},
    }
    for phase, phase_spec in spec["phases"].items():
        expected_slices = len(next(iter(phase_spec["time_indices_by_dataset"].values())))
        phase_rows = []
        for dataset in datasets:
            paths = sorted((Path(spec["cache_roots"][phase]) / dataset).glob("slice_*.npz"))
            if len(paths) != expected_slices:
                raise RuntimeError(
                    f"{phase}/{dataset}: expected {expected_slices} slices, found {len(paths)}"
                )
            for path in paths:
                with np.load(path) as data:
                    raw = np.asarray(data["raw_features"])
                    fmt = np.asarray(data["fmt_features"])
                    seeds = np.asarray(data["seeds"])
                    reference = np.asarray(data["reference"])
                    scale_id = np.asarray(data["scale_id"])
                    metadata = json.loads(str(data["metadata_json"]))
                count = len(reference)
                if raw.shape != (count, 7 * int(spec["pathlines"]["sampled_steps"]) * 3):
                    raise ValueError(f"bad Raw shape in {path}: {raw.shape}")
                if fmt.shape[0] != count or seeds.shape != (count, 3) or scale_id.shape != (count,):
                    raise ValueError(f"sample arrays disagree in {path}")
                if not np.isfinite(raw).all() or not np.isfinite(fmt).all() or not np.isfinite(seeds).all():
                    raise ValueError(f"non-finite cache values in {path}")
                if metadata.get("config_sha256") != expected_hash:
                    raise ValueError(f"config fingerprint mismatch in {path}")
                expected_scale_count = len(metadata["scale_table"])
                counts = np.bincount(scale_id, minlength=expected_scale_count)
                if len(counts) != expected_scale_count or np.any(counts < 20):
                    raise ValueError(f"insufficient scale coverage in {path}: {counts}")
                phase_rows.append({
                    "dataset": dataset,
                    "file": path.name,
                    "sample_count": count,
                    "positive_fraction": float(reference.mean()),
                    "scale_count": expected_scale_count,
                    "minimum_samples_per_scale": int(counts.min()),
                    "maximum_samples_per_scale": int(counts.max()),
                })
        report["phases"][phase] = {
            "file_count": len(phase_rows),
            "sample_count": int(sum(row["sample_count"] for row in phase_rows)),
            "minimum_samples_per_scale": min(
                row["minimum_samples_per_scale"] for row in phase_rows
            ),
            "maximum_samples_per_scale": max(
                row["maximum_samples_per_scale"] for row in phase_rows
            ),
            "positive_fraction_range": [
                min(row["positive_fraction"] for row in phase_rows),
                max(row["positive_fraction"] for row in phase_rows),
            ],
        }
    output = Path("outputs/mainExp_Task5_3D_1.1/cache_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mainExp_Task5_3D_1.1.yaml")
    args = parser.parse_args()
    audit(args.config)
