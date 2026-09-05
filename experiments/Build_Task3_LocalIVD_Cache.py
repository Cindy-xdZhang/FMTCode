"""Build resumable Task3 local-IVD labels aligned with cached pathline primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from FMT_Utils.FMT_3D_pipeline import compute_ivd_reference_3d
from FMT_Utils.LocalIVDLabel_3D import sample_local_ivd_labels_3d
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d


def build(config_path="config/Verify_Task3LabelsLiteral_1.1.yaml", overwrite=False):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    sampling = yaml.safe_load(Path(spec["sampling_spec"]).read_text(encoding="utf-8"))
    source_root = Path(sampling["output_dir"]) / "cache" / spec["sampling_variant"]
    output_root = Path(spec["output_dir"]); output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": spec["experiment"], "config": spec, "datasets": {}}
    for dataset in spec["datasets"]:
        paths = sorted((source_root / dataset).glob("slice_*.npz"))
        if len(paths) != 10:
            raise RuntimeError(f"expected 10 source slices for {dataset}, found {len(paths)}")
        target_dir = output_root / dataset; target_dir.mkdir(parents=True, exist_ok=True)
        manifest["datasets"][dataset] = []
        for ordinal, source_path in enumerate(paths):
            target = target_dir / source_path.name
            if target.exists() and not overwrite:
                with np.load(target) as data:
                    metadata = json.loads(str(data["metadata_json"]))
                manifest["datasets"][dataset].append(metadata)
                continue
            started = time.perf_counter()
            with np.load(source_path) as data:
                seeds = np.asarray(data["seeds"], dtype=np.float32)
                source_metadata = json.loads(str(data["metadata_json"]))
            field, _ = load_netcdf_window_3d(
                source_metadata["source_path"], source_metadata["source_start_index"], 2,
                int(spec["max_spatial_dim"]),
            )
            ivd_volume, _, axes = compute_ivd_reference_3d(field, 0.0, seeds)
            labels, ivd, thresholds = sample_local_ivd_labels_3d(
                ivd_volume, axes, seeds, int(spec["label"]["neighborhood_size"]),
                str(spec["label"]["mode"]), float(spec["label"]["value"]),
            )
            metadata = {
                "dataset": dataset, "ordinal": ordinal,
                "source_cache": str(source_path.resolve()),
                "source_start_index": source_metadata["source_start_index"],
                "label_mode": spec["label"]["mode"],
                "label_value": spec["label"]["value"],
                "neighborhood_size": spec["label"]["neighborhood_size"],
                "sample_count": int(len(labels)),
                "positive_count": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            np.savez_compressed(
                target, labels=labels.astype(bool), ivd_at_seeds=ivd,
                threshold_at_seeds=thresholds,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            manifest["datasets"][dataset].append(metadata)
            print(f"{dataset} {ordinal + 1}/10: positive={labels.mean():.3%}", flush=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_Task3LabelsLiteral_1.1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); build(args.config, args.overwrite)
