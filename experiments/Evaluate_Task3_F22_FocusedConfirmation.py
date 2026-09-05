"""Evaluate the frozen F22 Task3 recipe on pre-registered fresh timeslices.

This program is evaluation-only: it never trains a model, selects a candidate,
changes alpha, or changes a decision threshold on confirmation data.
"""

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
    _group_for_dataset,
    _load_spec,
    _portable_basename,
    _read_csv,
    _stack_split,
    _write_csv,
)
from Search_Task3_FMTResidual_Stage2_3D import (
    _result_path,
    _selected_candidate,
)
from Search_Task5_CylinderHyperparams import (
    _evaluate_baseline,
    _evaluate_residual,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(search: dict, candidate: dict, dataset: str,
                seed: int, source: str) -> Path:
    rows = _read_csv(_result_path(search, candidate, dataset, seed, source))
    if len(rows) != 1:
        raise RuntimeError(
            f"selected {source} checkpoint row missing or duplicated for seed {seed}"
        )
    checkpoint = Path(rows[0]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _load_confirmation(spec: dict, candidate: dict,
                       device: torch.device) -> tuple[list[tuple], list[int]]:
    dataset = "f22raptor"
    records = load_cache_records(
        Path(spec["source_cache_root"]) / dataset,
        expected_count=int(spec["expected_slices"]),
    )
    result = []
    source_indices = []
    for record in records:
        label_path = Path(spec["label_cache_root"]) / dataset / record["path"].name
        with np.load(label_path) as label_file:
            labels = np.asarray(label_file["labels"], dtype=np.float32)
            metadata = json.loads(str(label_file["metadata_json"]))
        if _portable_basename(metadata["source_cache"]) != record["path"].name:
            raise ValueError(f"confirmation label/source mismatch: {label_path}")
        if not np.array_equal(labels.astype(bool), record["reference"]):
            raise RuntimeError(f"confirmation labels differ from source: {label_path}")
        sampled_steps = record["raw"].shape[1] // (7 * 3)
        raw = record["raw"].reshape(-1, 7, sampled_steps, 3)
        fmt = feature_matrix(record, candidate["fmt_feature"], device)
        if not (len(raw) == len(fmt) == len(labels)):
            raise ValueError(f"feature/label mismatch: {record['path']}")
        result.append((raw, fmt, labels, int(record["ordinal"]), metadata))
        source_indices.append(int(metadata["source_start_index"]))
    return result, source_indices


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search_path = Path(spec["search_config"])
    selection_path = Path(spec["stage2_selection"])
    schedule_path = Path(spec["confirmation_schedule"])
    search = _load_spec(search_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_hash = _sha256(selection_path)
    if selection.get("confirmation_opened", False):
        raise RuntimeError("selection file says confirmation was already opened")
    group_name, group = _group_for_dataset(search, "f22raptor")
    candidate = _selected_candidate(
        search, group_name, selection["primary_by_group"][group_name]
    )
    schedule = yaml.safe_load(schedule_path.read_text(encoding="utf-8"))
    frozen_indices = [
        int(value) for value in
        schedule["sampling"]["fixed_time_indices_by_dataset"]["f22raptor"]
    ]
    if len(frozen_indices) != int(spec["expected_slices"]):
        raise ValueError("frozen confirmation schedule length mismatch")

    output = Path(spec["output_root"])
    marker = output / "audit.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["stage2_selection_sha256"] != selection_hash:
            raise RuntimeError("confirmation output belongs to another selection")
        print(marker.read_text(encoding="utf-8"))
        return marker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, observed_indices = _load_confirmation(spec, candidate, device)
    if observed_indices != frozen_indices:
        raise RuntimeError(
            f"confirmation indices changed: {observed_indices} != {frozen_indices}"
        )
    split = _stack_split(records, list(range(int(spec["expected_slices"]))))
    rows = []
    for seed_value in search["stage2_screen_seeds"]:
        seed = int(seed_value)
        for method in ("raw", "raw_wide"):
            metrics = _evaluate_baseline(
                Path(group["raw_checkpoint_dir"])
                / f"f22raptor_{method}_seed{seed}.pt",
                split, search["training"]["batch_size"], device,
            )
            rows.append({
                "dataset": "f22raptor", "seed": seed, "method": method,
                "candidate_id": candidate["id"], **metrics,
            })
        for source, method in (
            ("raw_pca", "raw_pca_residual"), ("fmt", "fmt_residual")
        ):
            metrics = _evaluate_residual(
                _checkpoint(search, candidate, "f22raptor", seed, source),
                split, search["training"]["batch_size"], device,
            )
            rows.append({
                "dataset": "f22raptor", "seed": seed, "method": method,
                "candidate_id": candidate["id"], **metrics,
            })
        print(f"fresh confirmation seed={seed} complete", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_run.csv", rows)
    means = {}
    for method in ("raw", "raw_wide", "raw_pca_residual", "fmt_residual"):
        selected = [row for row in rows if row["method"] == method]
        means[method] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in ("f1", "average_precision")
        }
    strong_raw = {
        metric: max(means["raw"][metric], means["raw_wide"][metric])
        for metric in ("f1", "average_precision")
    }
    paired = []
    for seed_value in search["stage2_screen_seeds"]:
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
    audit = {
        "experiment": spec["experiment"],
        "evaluation_only": True,
        "selection_or_fitting_on_confirmation": False,
        "stage2_selection_sha256": selection_hash,
        "confirmation_schedule_sha256": _sha256(schedule_path),
        "confirmation_indices": observed_indices,
        "selected_candidate": candidate,
        "means": means,
        "strong_raw": strong_raw,
        "gains": {
            "f1_vs_raw_pca": (
                means["fmt_residual"]["f1"] - means["raw_pca_residual"]["f1"]
            ),
            "average_precision_vs_raw_pca": (
                means["fmt_residual"]["average_precision"]
                - means["raw_pca_residual"]["average_precision"]
            ),
            "f1_vs_strong_raw": means["fmt_residual"]["f1"] - strong_raw["f1"],
            "average_precision_vs_strong_raw": (
                means["fmt_residual"]["average_precision"]
                - strong_raw["average_precision"]
            ),
        },
        "paired_seed_gains": paired,
    }
    marker.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(marker.read_text(encoding="utf-8"))
    return marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
