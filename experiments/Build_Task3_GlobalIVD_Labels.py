"""Materialize Task3 labels from the frozen Task1/Task2 global-IVD reference.

The source universality caches already contain ``reference`` computed as
``IVD >= percentile_95(IVD_volume)``.  Reusing that boolean array avoids a
second implementation of the Task1/Task2 label and guarantees bit-for-bit
agreement across the three tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _identity(spec):
    return json.loads(json.dumps(spec, sort_keys=True))


def build(config_path: str, overwrite: bool = False) -> Path:
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source_root = Path(spec["source_cache_root"])
    output_root = Path(spec["output_dir"]) / "labels"
    output_root.mkdir(parents=True, exist_ok=True)

    snapshot = output_root / "config_snapshot.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if _identity(previous) != _identity(spec):
            raise RuntimeError(
                f"configuration changed for {output_root}; use a new version"
            )
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    percentile = float(spec["label"]["percentile"])
    if percentile != 95.0:
        raise ValueError(
            "this experiment must match the frozen Task1/Task2 IVD p95 label"
        )
    expected_slices = int(spec["expected_slices"])
    manifest = {"experiment": spec["experiment"], "config": spec, "datasets": {}}

    for dataset in spec["datasets"]:
        source_paths = sorted((source_root / dataset).glob("slice_*.npz"))
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
            with np.load(source_path) as source:
                if "reference" not in source:
                    raise KeyError(f"source cache has no global-IVD reference: {source_path}")
                labels = np.asarray(source["reference"], dtype=bool)
                source_metadata = json.loads(str(source["metadata_json"]))
                sample_count = len(np.asarray(source["seeds"]))
            if len(labels) != sample_count:
                raise ValueError(f"reference/seed mismatch in {source_path}")
            threshold = float(source_metadata["ivd_threshold"])
            metadata = {
                "dataset": dataset,
                "ordinal": ordinal,
                "source_cache": str(source_path.resolve()),
                "source_start_index": int(source_metadata["source_start_index"]),
                "label_mode": "standard_global_ivd_percentile",
                "label_value": percentile,
                "ivd_threshold": threshold,
                "sample_count": sample_count,
                "positive_count": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "copied_exactly_from_source_reference": True,
            }
            if target.exists() and not overwrite:
                with np.load(target) as existing:
                    existing_labels = np.asarray(existing["labels"], dtype=bool)
                    existing_metadata = json.loads(str(existing["metadata_json"]))
                if not np.array_equal(existing_labels, labels):
                    raise RuntimeError(f"stale label cache: {target}")
                metadata = existing_metadata
            else:
                np.savez_compressed(
                    target,
                    labels=labels,
                    threshold_at_seeds=np.full(sample_count, threshold, dtype=np.float32),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            manifest["datasets"][dataset].append(metadata)
            print(
                f"{dataset} {ordinal + 1}/{expected_slices}: n={sample_count}, "
                f"positive={labels.mean():.3%}, threshold={threshold:.6g}",
                flush=True,
            )

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.config, args.overwrite)
