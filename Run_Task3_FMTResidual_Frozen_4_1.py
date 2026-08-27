"""Frozen multi-seed confirmation for family-selected supervised 3D Task3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.Task12Data_3D import feature_matrix, load_cache_records
from Search_Task3_FMTResidual_3D import (
    _candidate_spec,
    _group_for_dataset,
    _load_records,
    _load_search_splits,
    _load_spec as _load_search_spec,
    _read_csv,
    _write_csv,
)
from Search_Task3_FMTResidual_Stage2_3D import (
    _result_path as _stage2_result_path,
    _stage2_candidates,
)
from Search_Task5_CylinderHyperparams import (
    _evaluate_baseline,
    _evaluate_residual,
)
from Verify_Task3_FMTClassifier import (
    _append_csv,
    _normalize_train_only,
    _portable_basename,
    _stack_split,
)
from Verify_Task3_FMTResidual import _train_one


def _load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "experiment", "search_config", "stage2_selection", "output_root",
        "confirmation_roots", "final_training_seeds", "frozen_recipe_manifest",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing final Task3 config keys: {missing}")
    return spec


def _frozen_state(spec: dict) -> tuple[dict, dict, str]:
    search = _load_search_spec(spec["search_config"])
    selection_path = Path(spec["stage2_selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    manifest = json.loads(
        Path(spec["frozen_recipe_manifest"]).read_text(encoding="utf-8")
    )
    if manifest["selections"]["task3"]["sha256"] != selection_hash:
        raise RuntimeError("Task3 selection changed after confirmation cache access")
    return search, selection, selection_hash


def _confirmation_group(spec: dict, dataset: str) -> dict:
    matches = [
        group for group in spec["confirmation_roots"].values()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"confirmation dataset {dataset} matched {len(matches)} roots")
    return matches[0]


def _selected_candidate(search: dict, selection: dict,
                        group_name: str) -> dict:
    candidate_id = str(selection["primary_by_group"][group_name]["candidate_id"])
    candidates = {
        row["id"]: row for row in _stage2_candidates(search, group_name)
    }
    if candidate_id not in candidates:
        raise RuntimeError(f"frozen Task3 candidate not found: {candidate_id}")
    return candidates[candidate_id]


def _load_confirmation(spec: dict, search: dict, dataset: str,
                       candidate: dict, device) -> list[tuple]:
    group = _confirmation_group(spec, dataset)
    source_records = load_cache_records(
        Path(group["source_root"]) / dataset,
        expected_count=int(spec.get("confirmation_count", 4)),
    )
    result = []
    for record in source_records:
        label_path = Path(group["label_root"]) / dataset / record["path"].name
        with np.load(label_path) as labels_file:
            labels = np.asarray(labels_file["labels"], dtype=np.float32)
            metadata = json.loads(str(labels_file["metadata_json"]))
        if _portable_basename(metadata["source_cache"]) != record["path"].name:
            raise ValueError(f"confirmation label/source mismatch: {label_path}")
        expected_percentile = spec.get("expected_ivd_percentile")
        if expected_percentile is not None:
            actual_percentile = metadata.get(
                "label_value", metadata.get("ivd_percentile")
            )
            if actual_percentile is None or not np.isclose(
                float(actual_percentile), float(expected_percentile)
            ):
                raise RuntimeError(
                    f"confirmation label percentile mismatch in {label_path}: "
                    f"expected {expected_percentile}, found {actual_percentile}"
                )
        require_reference_match = bool(
            spec.get("require_confirmation_reference_match", True)
        )
        if require_reference_match and not np.array_equal(
            labels.astype(bool), record["reference"]
        ):
            raise RuntimeError(f"confirmation global-IVD labels differ: {label_path}")
        sampled_steps = record["raw"].shape[1] // (7 * 3)
        raw = record["raw"].reshape(-1, 7, sampled_steps, 3)
        fmt = feature_matrix(record, candidate["fmt_feature"], device)
        result.append((raw, fmt, labels, int(record["ordinal"]), metadata))
    return result


def _final_result_path(spec: dict, candidate: dict, dataset: str,
                       seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "trained_selected" / candidate["id"]
        / dataset / f"seed{int(seed)}" / source / "per_run.csv"
    )


def _checkpoint_from_row(path: Path) -> Path:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"checkpoint row missing or duplicated: {path}")
    checkpoint = Path(rows[0]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _ensure_residual_checkpoint(spec: dict, search: dict, group: dict,
                                candidate: dict, dataset: str, seed: int,
                                source: str, splits, stats, device, fmt_dim) -> Path:
    if seed in {int(value) for value in search["stage2_screen_seeds"]}:
        stage2 = _stage2_result_path(search, candidate, dataset, seed, source)
        if stage2.exists():
            return _checkpoint_from_row(stage2)
    result_path = _final_result_path(spec, candidate, dataset, seed, source)
    if result_path.exists():
        return _checkpoint_from_row(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    run_spec = _candidate_spec(
        search, group, candidate, dataset, seed, source,
        result_path.parent, fmt_dim,
    )
    (result_path.parent / "config_snapshot.yaml").write_text(
        yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8"
    )
    row = _train_one(
        run_spec, dataset, seed, splits, stats, device, result_path.parent
    )
    row.update({
        "candidate_id": candidate["id"],
        "fmt_feature": candidate["fmt_feature"],
        "fmt_dim": fmt_dim,
        "final_selected_training": True,
    })
    _append_csv(result_path, row)
    return Path(row["checkpoint"])


def run_dataset(config_path: str, dataset: str) -> Path:
    spec = _load_spec(config_path)
    search, selection, selection_hash = _frozen_state(spec)
    if dataset not in search["datasets"]:
        raise ValueError(f"unknown Task3 final dataset {dataset!r}")
    group_name, group = _group_for_dataset(search, dataset)
    candidate = _selected_candidate(search, selection, group_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Newer searches may append an explicitly exposed spatial population to
    # development validation.  The helper is backward compatible with 4.1,
    # where no robust_validation block exists.
    train, validation = _load_search_splits(
        search, dataset, candidate, device
    )
    train, validation, _, stats = _normalize_train_only(train, validation)
    splits = (train, validation, None)
    fmt_dim = int(train[1].shape[1])
    confirmation_records = _load_confirmation(
        spec, search, dataset, candidate, device
    )
    confirmation = _stack_split(
        confirmation_records, list(range(int(spec.get("confirmation_count", 4))))
    )
    target = Path(spec["output_root"]) / "shards" / f"{dataset}.csv"
    rows = _read_csv(target)
    if rows and {row["stage2_selection_sha256"] for row in rows} != {selection_hash}:
        raise RuntimeError(f"stale Task3 final shard for {dataset}")
    completed = {
        (row["method"], int(row["seed"])) for row in rows
    }
    baseline_dir = Path(group["raw_checkpoint_dir"])
    for seed_value in spec["final_training_seeds"]:
        seed = int(seed_value)
        for variant in ("raw", "raw_wide"):
            if (variant, seed) in completed:
                continue
            metrics = _evaluate_baseline(
                baseline_dir / f"{dataset}_{variant}_seed{seed}.pt",
                confirmation, search["training"]["batch_size"], device,
            )
            rows.append({
                "experiment": spec["experiment"],
                "stage2_selection_sha256": selection_hash,
                "dataset": dataset, "group": group_name, "seed": seed,
                "method": variant, "candidate_id": candidate["id"], **metrics,
            })
            _write_csv(target, rows)
            completed.add((variant, seed))
        for source, method in (
            ("raw_pca", "raw_pca_residual"), ("fmt", "fmt_residual")
        ):
            if (method, seed) in completed:
                continue
            checkpoint = _ensure_residual_checkpoint(
                spec, search, group, candidate, dataset, seed, source,
                splits, stats, device, fmt_dim,
            )
            metrics = _evaluate_residual(
                checkpoint, confirmation,
                search["training"]["batch_size"], device,
            )
            rows.append({
                "experiment": spec["experiment"],
                "stage2_selection_sha256": selection_hash,
                "dataset": dataset, "group": group_name, "seed": seed,
                "method": method, "candidate_id": candidate["id"], **metrics,
            })
            _write_csv(target, rows)
            completed.add((method, seed))
        print(f"Task3 final {dataset} seed={seed} complete", flush=True)
    return target


def summarize(config_path: str) -> Path:
    spec = _load_spec(config_path)
    search, _, selection_hash = _frozen_state(spec)
    rows = []
    expected = 4 * len(spec["final_training_seeds"])
    for dataset in search["datasets"]:
        values = _read_csv(
            Path(spec["output_root"]) / "shards" / f"{dataset}.csv"
        )
        if len(values) != expected:
            raise RuntimeError(
                f"Task3 final shard {dataset} has {len(values)} rows, expected {expected}"
            )
        if {row["stage2_selection_sha256"] for row in values} != {selection_hash}:
            raise RuntimeError(f"Task3 selection hash mismatch for {dataset}")
        rows.extend(values)
    output = Path(spec["output_root"])
    _write_csv(output / "per_run.csv", rows)
    summary = {}
    for dataset in search["datasets"]:
        selected = [row for row in rows if row["dataset"] == dataset]
        summary[dataset] = {}
        for method in ("raw", "raw_wide", "raw_pca_residual", "fmt_residual"):
            values = [row for row in selected if row["method"] == method]
            summary[dataset][method] = {
                metric: float(np.mean([float(row[metric]) for row in values]))
                for metric in ("f1", "average_precision")
            }
        strong = {
            metric: max(summary[dataset][method][metric]
                        for method in ("raw", "raw_wide"))
            for metric in ("f1", "average_precision")
        }
        summary[dataset]["strong_raw"] = strong
        summary[dataset]["gains"] = {
            f"{metric}_vs_raw_pca": (
                summary[dataset]["fmt_residual"][metric]
                - summary[dataset]["raw_pca_residual"][metric]
            ) for metric in ("f1", "average_precision")
        }
        summary[dataset]["gains"].update({
            f"{metric}_vs_strong_raw": (
                summary[dataset]["fmt_residual"][metric] - strong[metric]
            ) for metric in ("f1", "average_precision")
        })
    macro_f1 = float(np.mean([
        row["gains"]["f1_vs_raw_pca"] for row in summary.values()
    ]))
    macro_ap = float(np.mean([
        row["gains"]["average_precision_vs_raw_pca"]
        for row in summary.values()
    ]))
    result = {
        "experiment": spec["experiment"],
        "stage2_selection_sha256": selection_hash,
        "comparison": "FMT residual minus same-width same-structure Raw-PCA residual",
        "datasets": summary,
        "dataset_macro_f1_gain_vs_raw_pca": macro_f1,
        "dataset_macro_ap_gain_vs_raw_pca": macro_ap,
        "target_gain": float(spec["target_dataset_macro_f1_gain"]),
        "target_reached": macro_f1 >= float(spec["target_dataset_macro_f1_gain"]),
    }
    target = output / "summary.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("dataset", "summary"), required=True)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    if args.mode == "summary":
        summarize(args.config)
    elif args.dataset is None:
        parser.error("dataset mode requires --dataset")
    else:
        run_dataset(args.config, args.dataset)


if __name__ == "__main__":
    main()
