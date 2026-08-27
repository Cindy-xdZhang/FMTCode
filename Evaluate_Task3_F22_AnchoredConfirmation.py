"""Evaluate one frozen anchored-FMT recipe on preregistered F22 slices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.Task12Data_3D import feature_matrix, load_cache_records
from Search_Task3_FMTResidual_3D import (
    _load_spec,
    _portable_basename,
    _result_path,
    _read_csv,
    _write_csv,
)
from Search_Task5_CylinderHyperparams import (
    _evaluate_baseline,
    _evaluate_residual,
)
from Verify_Task3_FMTClassifier import _stack_split


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(search, candidate, seed, source):
    rows = _read_csv(
        _result_path(search, candidate, "f22raptor", int(seed), source)
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"missing or duplicate {source} checkpoint for seed {seed}"
        )
    checkpoint = Path(rows[0]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _confirmation_records(spec, candidate, device):
    records = load_cache_records(
        Path(spec["source_cache_root"]) / "f22raptor",
        expected_count=int(spec["expected_slices"]),
    )
    result, source_indices = [], []
    for record in records:
        label_path = (
            Path(spec["label_cache_root"]) / "f22raptor" / record["path"].name
        )
        with np.load(label_path) as data:
            labels = np.asarray(data["labels"], dtype=np.float32)
            metadata = json.loads(str(data["metadata_json"]))
        if _portable_basename(metadata["source_cache"]) != record["path"].name:
            raise RuntimeError(f"label/source mismatch: {label_path}")
        if not np.array_equal(labels.astype(bool), record["reference"]):
            raise RuntimeError(f"labels differ from cached IVD reference: {label_path}")
        steps = record["raw"].shape[1] // (7 * 3)
        raw = record["raw"].reshape(-1, 7, steps, 3)
        fmt = feature_matrix(record, candidate["fmt_feature"], device)
        if not (len(raw) == len(fmt) == len(labels)):
            raise RuntimeError(f"feature/label length mismatch: {record['path']}")
        result.append((raw, fmt, labels, int(record["ordinal"]), metadata))
        source_indices.append(int(metadata["source_start_index"]))
    return result, source_indices


def run(config_path: str | Path):
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search_path = Path(spec["search_config"])
    selection_path = Path(spec["selection"])
    schedule_path = Path(spec["confirmation_schedule"])
    search = _load_spec(search_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if bool(selection.get("confirmation_opened", False)):
        raise RuntimeError("selection unexpectedly opened confirmation data")
    if selection["search_config_sha256"] != _sha256(search_path):
        raise RuntimeError("search config changed after candidate selection")
    if selection["confirmation_schedule_sha256"] != _sha256(schedule_path):
        raise RuntimeError("confirmation schedule changed after candidate selection")
    candidate = dict(selection["selected_candidate"])
    schedule = yaml.safe_load(schedule_path.read_text(encoding="utf-8"))
    frozen_indices = [
        int(value) for value in
        schedule["sampling"]["fixed_time_indices_by_dataset"]["f22raptor"]
    ]
    output = Path(spec["output_root"])
    marker = output / "audit.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["selection_sha256"] != _sha256(selection_path):
            raise RuntimeError("confirmation belongs to another frozen selection")
        print(marker.read_text(encoding="utf-8"))
        return marker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, observed_indices = _confirmation_records(spec, candidate, device)
    if observed_indices != frozen_indices:
        raise RuntimeError(
            f"confirmation indices changed: {observed_indices} != {frozen_indices}"
        )
    split = _stack_split(records, list(range(int(spec["expected_slices"]))))
    baseline_dir = Path(
        search["groups"]["f22raptor"]["raw_checkpoint_dir"]
    )
    rows = []
    for seed_value in search["screen_seeds"]:
        seed = int(seed_value)
        for method in ("raw", "raw_wide"):
            metrics = _evaluate_baseline(
                baseline_dir / f"f22raptor_{method}_seed{seed}.pt",
                split,
                search["training"]["batch_size"],
                device,
            )
            rows.append({
                "dataset": "f22raptor", "seed": seed, "method": method,
                "candidate_id": candidate["id"], **metrics,
            })
        for source, method in (
            ("raw_pca", "raw_pca_residual"),
            ("fmt", "fmt_residual"),
        ):
            metrics = _evaluate_residual(
                _checkpoint(search, candidate, seed, source),
                split,
                search["training"]["batch_size"],
                device,
            )
            rows.append({
                "dataset": "f22raptor", "seed": seed, "method": method,
                "candidate_id": candidate["id"], **metrics,
            })
        print(f"confirmation seed={seed} complete", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_run.csv", rows)
    methods = ("raw", "raw_wide", "raw_pca_residual", "fmt_residual")
    means = {
        method: {
            metric: float(np.mean([
                float(row[metric]) for row in rows if row["method"] == method
            ]))
            for metric in ("f1", "average_precision")
        }
        for method in methods
    }
    paired = []
    for seed_value in search["screen_seeds"]:
        seed = int(seed_value)
        by_method = {
            row["method"]: row for row in rows if int(row["seed"]) == seed
        }
        paired.append({
            "seed": seed,
            "f1_gain_vs_raw_pca": (
                float(by_method["fmt_residual"]["f1"])
                - float(by_method["raw_pca_residual"]["f1"])
            ),
            "average_precision_gain_vs_raw_pca": (
                float(by_method["fmt_residual"]["average_precision"])
                - float(by_method["raw_pca_residual"]["average_precision"])
            ),
        })
    gains = {
        "f1_vs_raw_pca": (
            means["fmt_residual"]["f1"] - means["raw_pca_residual"]["f1"]
        ),
        "average_precision_vs_raw_pca": (
            means["fmt_residual"]["average_precision"]
            - means["raw_pca_residual"]["average_precision"]
        ),
    }
    audit = {
        "experiment": spec["experiment"],
        "evaluation_only": True,
        "selection_or_fitting_on_confirmation": False,
        "selection_sha256": _sha256(selection_path),
        "confirmation_schedule_sha256": _sha256(schedule_path),
        "confirmation_indices": observed_indices,
        "selected_candidate": candidate,
        "means": means,
        "gains": gains,
        "paired_seed_gains": paired,
        "all_seed_metrics_positive_vs_raw_pca": bool(min(
            value
            for row in paired
            for value in (
                row["f1_gain_vs_raw_pca"],
                row["average_precision_gain_vs_raw_pca"],
            )
        ) > 0.0),
    }
    marker.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(marker.read_text(encoding="utf-8"))
    return marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
