"""Score the frozen Task2-4.1 recipes under several whole-field IVD labels.

The VAE and KMeans are label-free.  Every dataset/arm/seed is trained once;
only the calibration mapping and confirmation labels change by percentile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans

from Build_Task23_IVDPercentile_Labels import all_percentiles, percentile_tag
from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from Run_Task2_3D_Main import _prepare_inputs
from Run_Task2_FMTVAE_Frozen_4_1 import (
    _frozen_state,
    _group_for_dataset,
    _selected_recipe,
)
from Verify_HighReVAE import _train


ARMS = {"raw": "Raw+VAE", "fmt": "FMT+VAE"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _population(main_spec: dict, dataset: str) -> str:
    matches = [
        name for name, group in main_spec["confirmation_roots"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset} matched {len(matches)} populations")
    return matches[0]


def _label_root(master: dict, split: str, population: str,
                percentile: float) -> Path:
    template = master["label_roots"][split][population]
    return Path(str(template).format(tag=percentile_tag(percentile)))


def _labels(master: dict, split: str, population: str, percentile: float,
            dataset: str, records: list[dict]) -> np.ndarray:
    root = _label_root(master, split, population, percentile) / dataset
    values = []
    for record in records:
        path = root / record["path"].name
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as cached:
            labels = np.asarray(cached["labels"], dtype=bool)
            metadata = json.loads(str(cached["metadata_json"]))
        source_name = Path(str(metadata["source_cache"]).replace("\\", "/")).name
        actual = metadata.get("label_value", metadata.get("ivd_percentile"))
        if source_name != record["path"].name:
            raise ValueError(f"label/source mismatch in {path}")
        if actual is None or not np.isclose(float(actual), float(percentile)):
            raise ValueError(f"wrong percentile metadata in {path}: {actual}")
        if len(labels) != len(record["reference"]):
            raise ValueError(f"label length mismatch in {path}")
        values.append(labels)
    return np.concatenate(values)


def _prediction_path(output: Path, dataset: str, arm: str, seed: int) -> Path:
    return output / "predictions" / f"{dataset}_{arm}_seed{seed}.npz"


def _load_prediction(path: Path, selection_hash: str,
                     calibration_count: int, confirmation_count: int):
    if not path.exists():
        return None
    with np.load(path) as cached:
        calibration = np.asarray(cached["calibration_labels"], dtype=np.int64)
        confirmation = np.asarray(cached["confirmation_labels"], dtype=np.int64)
        losses = json.loads(str(cached["losses_json"]))
        stored_hash = str(cached["selection_sha256"])
    if stored_hash != selection_hash:
        raise RuntimeError(f"stale Task2 selection in {path}")
    if len(calibration) != calibration_count or len(confirmation) != confirmation_count:
        raise RuntimeError(f"stale Task2 prediction sizes in {path}")
    return calibration, confirmation, losses


def _save_prediction(path: Path, selection_hash: str, calibration,
                     confirmation, losses: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        calibration_labels=np.asarray(calibration, dtype=np.int8),
        confirmation_labels=np.asarray(confirmation, dtype=np.int8),
        losses_json=np.asarray(json.dumps(losses, sort_keys=True)),
        selection_sha256=np.asarray(selection_hash),
    )


def run_dataset(config_path: str, dataset: str, resume: bool = False) -> Path:
    master = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    main_path = Path(master["task2"]["main_config"])
    main_spec = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    search, selection, selection_hash = _frozen_state(main_spec)
    if dataset not in search["datasets"]:
        raise ValueError(f"unknown Task2 dataset {dataset!r}")
    group_name, group, feature, architecture = _selected_recipe(
        search, selection, dataset
    )
    population = _population(main_spec, dataset)
    development = load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(search.get("expected_slices", 10)),
    )
    confirmation_group = main_spec["confirmation_roots"][population]
    confirmation = load_cache_records(
        Path(confirmation_group["root"]) / dataset,
        expected_count=int(main_spec["splits"]["confirmation_count"]),
    )
    by_ordinal = {int(row["ordinal"]): row for row in development}
    train_records = [
        by_ordinal[int(value)] for value in main_spec["splits"]["train"]
    ]
    calibration_records = [
        by_ordinal[int(value)]
        for value in main_spec["splits"]["cluster_calibration"]
    ]
    evaluate_records = calibration_records + confirmation
    calibration_count = sum(len(row["reference"]) for row in calibration_records)
    confirmation_count = sum(len(row["reference"]) for row in confirmation)
    references = {
        value: (
            _labels(master, "development", population, value, dataset,
                    calibration_records),
            _labels(master, "confirmation", population, value, dataset,
                    confirmation),
        ) for value in all_percentiles(master)
    }

    output = Path(master["task2"]["output_dir"])
    target = output / "shards" / f"{dataset}.csv"
    rows = _read_csv(target) if resume else []
    if rows and {row["stage2_selection_sha256"] for row in rows} != {selection_hash}:
        raise RuntimeError(f"stale Task2 percentile shard for {dataset}")
    source = EasyConfig(group["source_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arm in ARMS:
        train_x, evaluate_x = _prepare_inputs(
            train_records, evaluate_records, arm, feature["name"], device
        )
        for seed_value in main_spec["final_training_seeds"]:
            seed = int(seed_value)
            prediction_path = _prediction_path(output, dataset, arm, seed)
            cached = _load_prediction(
                prediction_path, selection_hash, calibration_count,
                confirmation_count,
            ) if resume else None
            if cached is None:
                train_mu, evaluate_mu, losses = _train(
                    train_x, evaluate_x, architecture, source, seed, device
                )
                kmeans = KMeans(
                    n_clusters=2, random_state=int(search["kmeans_seed"]),
                    n_init=int(search["kmeans_n_init"]),
                ).fit(train_mu)
                predicted = kmeans.predict(evaluate_mu)
                calibration_labels = predicted[:calibration_count]
                confirmation_labels = predicted[calibration_count:]
                _save_prediction(
                    prediction_path, selection_hash, calibration_labels,
                    confirmation_labels, losses,
                )
            else:
                calibration_labels, confirmation_labels, losses = cached
            rows = [
                row for row in rows
                if not (
                    row["dataset"] == dataset and row["arm"] == arm
                    and int(row["training_seed"]) == seed
                )
            ]
            for value, (calibration_reference, confirmation_reference) in references.items():
                cluster = calibrate_vortex_cluster(
                    calibration_reference, calibration_labels
                )
                metrics = binary_cluster_metrics(
                    confirmation_reference, confirmation_labels, cluster
                )
                rows.append({
                    "experiment": master["experiment"],
                    "stage2_selection_sha256": selection_hash,
                    "dataset": dataset, "group": group_name,
                    "population": population, "arm": arm,
                    "method": ARMS[arm], "feature_id": feature["id"],
                    "fmt_feature": feature["name"],
                    "architecture": architecture["id"],
                    "training_seed": seed, "ivd_percentile": float(value),
                    "percentile_tag": percentile_tag(value),
                    "cluster_as_vortex": int(cluster), **metrics, **losses,
                })
            _write_csv(target, rows)
            print(f"Task2 percentile {dataset}/{arm}/seed={seed} complete", flush=True)
    return target


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def summarize(config_path: str) -> Path:
    master = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    main_spec = yaml.safe_load(
        Path(master["task2"]["main_config"]).read_text(encoding="utf-8")
    )
    search, _, selection_hash = _frozen_state(main_spec)
    output = Path(master["task2"]["output_dir"])
    all_rows = []
    expected = 2 * len(main_spec["final_training_seeds"]) * len(
        all_percentiles(master)
    )
    for dataset in search["datasets"]:
        rows = _read_csv(output / "shards" / f"{dataset}.csv")
        if len(rows) != expected:
            raise RuntimeError(f"{dataset}: expected {expected} rows, found {len(rows)}")
        if {row["stage2_selection_sha256"] for row in rows} != {selection_hash}:
            raise RuntimeError(f"Task2 selection hash mismatch for {dataset}")
        all_rows.extend(rows)
    _write_csv(output / "per_run.csv", all_rows)

    dataset_rows = []
    for value in all_percentiles(master):
        for dataset in search["datasets"]:
            selected = [
                row for row in all_rows if row["dataset"] == dataset
                and float(row["ivd_percentile"]) == value
            ]
            raw = [row for row in selected if row["arm"] == "raw"]
            fmt = [row for row in selected if row["arm"] == "fmt"]
            raw_by_seed = {int(row["training_seed"]): float(row["f1"]) for row in raw}
            fmt_by_seed = {int(row["training_seed"]): float(row["f1"]) for row in fmt}
            if set(raw_by_seed) != set(fmt_by_seed):
                raise RuntimeError(f"unpaired Task2 seeds for {dataset}/{value}")
            gains = np.asarray([
                fmt_by_seed[seed] - raw_by_seed[seed]
                for seed in sorted(raw_by_seed)
            ])
            group_name, _ = _group_for_dataset(search, dataset)
            dataset_rows.append({
                "ivd_percentile": value, "percentile_tag": percentile_tag(value),
                "dataset": dataset, "family": group_name,
                "raw_f1_mean": _mean(raw, "f1"),
                "fmt_f1_mean": _mean(fmt, "f1"),
                "paired_f1_gain_mean": float(gains.mean()),
                "paired_f1_gain_std": float(gains.std(ddof=1)),
                "positive_seed_count": int((gains > 0).sum()),
                "reference_positive_fraction": _mean(raw, "positive_fraction"),
            })
    _write_csv(output / "paper_table.csv", dataset_rows)

    summaries = []
    for value in all_percentiles(master):
        selected = [row for row in dataset_rows if row["ivd_percentile"] == value]
        family_gains = [
            float(np.mean([
                row["paired_f1_gain_mean"] for row in selected
                if row["family"] == family
            ])) for family in search["groups"]
        ]
        summaries.append({
            "ivd_percentile": value, "percentile_tag": percentile_tag(value),
            "dataset_macro_raw_f1": float(np.mean([row["raw_f1_mean"] for row in selected])),
            "dataset_macro_fmt_f1": float(np.mean([row["fmt_f1_mean"] for row in selected])),
            "dataset_macro_f1_gain": float(np.mean([row["paired_f1_gain_mean"] for row in selected])),
            "family_macro_f1_gain": float(np.mean(family_gains)),
            "positive_dataset_count": int(sum(row["paired_f1_gain_mean"] > 0 for row in selected)),
            "positive_family_count": int(sum(value_ > 0 for value_ in family_gains)),
            "mean_reference_positive_fraction": float(np.mean([row["reference_positive_fraction"] for row in selected])),
        })
    _write_csv(output / "percentile_summary.csv", summaries)

    published = _read_csv(Path(master["task2"]["published_p95_runs"]))
    old = {
        (row["dataset"], row["arm"], int(row["training_seed"])):
        float(row["confirmation_f1"]) for row in published
    }
    new = {
        (row["dataset"], row["arm"], int(row["training_seed"])): float(row["f1"])
        for row in all_rows
        if float(row["ivd_percentile"]) == float(master["audit_percentile"])
    }
    if set(old) != set(new):
        raise RuntimeError("published/new Task2-4.1 p95 coverage differs")
    differences = np.asarray([new[key] - old[key] for key in sorted(old)])
    audit = {
        "run_count": len(differences),
        "mean_signed_f1_difference": float(differences.mean()),
        "max_absolute_f1_difference": float(np.abs(differences).max()),
    }
    requested = {float(value) for value in master["requested_percentiles"]}
    requested_rows = [row for row in summaries if row["ivd_percentile"] in requested]
    payload = {
        "experiment": master["experiment"], "task": "Task2",
        "training_reused_across_percentiles": True,
        "percentile_summary": summaries,
        "largest_requested_dataset_macro_gain": max(
            requested_rows, key=lambda row: row["dataset_macro_f1_gain"]
        ),
        "largest_requested_family_macro_gain": max(
            requested_rows, key=lambda row: row["family_macro_f1_gain"]
        ),
        "p95_reproduction_audit": audit,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return output / "percentile_summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("dataset", "summary"), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "summary":
        summarize(args.config)
    elif args.dataset is None:
        parser.error("dataset mode requires --dataset")
    else:
        run_dataset(args.config, args.dataset, args.resume)


if __name__ == "__main__":
    main()
