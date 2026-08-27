"""Frozen multi-seed confirmation for the family-selected 3D Task2 recipe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from Run_Task2_3D_Main import _prepare_inputs
from Search_Task2_FMTVAE_3D import _group_for_dataset, _write_csv
from Verify_HighReVAE import _train


def _load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "experiment", "search_config", "stage2_selection", "output_root",
        "confirmation_roots", "final_training_seeds", "splits",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing final Task2 config keys: {missing}")
    return spec


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _frozen_state(spec: dict) -> tuple[dict, dict, str]:
    search = yaml.safe_load(
        Path(spec["search_config"]).read_text(encoding="utf-8")
    )
    selection_path = Path(spec["stage2_selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    manifest_path = Path(spec["frozen_recipe_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered = manifest["selections"]["task2"]["sha256"]
    if registered != selection_hash:
        raise RuntimeError("Task2 selection changed after confirmation cache access")
    return search, selection, selection_hash


def _confirmation_root(spec: dict, dataset: str) -> Path:
    for group in spec["confirmation_roots"].values():
        if dataset in group["datasets"]:
            return Path(group["root"]) / dataset
    raise ValueError(f"no confirmation root for {dataset}")


def _selected_recipe(search: dict, selection: dict, dataset: str):
    group_name, group = _group_for_dataset(search, dataset)
    chosen = selection["primary_by_group"][group_name]
    features = {row["id"]: row for row in search["features"]}
    architectures = {row["id"]: row for row in search["stage2_architectures"]}
    return group_name, group, features[chosen["feature_id"]], architectures[
        chosen["architecture"]
    ]


def _score(train_mu, calibration_mu, confirmation_mu,
           calibration_reference, confirmation_reference, search):
    model = KMeans(
        n_clusters=2, random_state=int(search["kmeans_seed"]),
        n_init=int(search["kmeans_n_init"]),
    ).fit(train_mu)
    calibration_labels = model.predict(calibration_mu)
    vortex_cluster = calibrate_vortex_cluster(
        calibration_reference, calibration_labels
    )
    calibration = binary_cluster_metrics(
        calibration_reference, calibration_labels, vortex_cluster
    )
    confirmation = binary_cluster_metrics(
        confirmation_reference, model.predict(confirmation_mu), vortex_cluster
    )
    return vortex_cluster, calibration, confirmation


def run_dataset(config_path: str, dataset: str) -> Path:
    spec = _load_spec(config_path)
    search, selection, selection_hash = _frozen_state(spec)
    if dataset not in search["datasets"]:
        raise ValueError(f"unknown Task2 final dataset {dataset!r}")
    group_name, group, feature, architecture = _selected_recipe(
        search, selection, dataset
    )
    development = load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(search.get("expected_slices", 10)),
        ordinals=sorted(
            set(spec["splits"]["train"])
            | set(spec["splits"]["cluster_calibration"])
        ),
    )
    confirmation = load_cache_records(
        _confirmation_root(spec, dataset),
        expected_count=int(spec["splits"]["confirmation_count"]),
    )
    by_ordinal = {int(row["ordinal"]): row for row in development}
    train_records = [by_ordinal[int(value)] for value in spec["splits"]["train"]]
    calibration_records = [
        by_ordinal[int(value)] for value in spec["splits"]["cluster_calibration"]
    ]
    evaluation_records = calibration_records + confirmation
    calibration_reference = stack_reference(calibration_records)
    confirmation_reference = stack_reference(confirmation)
    calibration_count = len(calibration_reference)
    source = EasyConfig(group["source_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = Path(spec["output_root"]) / "shards" / f"{dataset}.csv"
    rows = _read_csv(target)
    if rows and {row["stage2_selection_sha256"] for row in rows} != {selection_hash}:
        raise RuntimeError(f"stale Task2 final shard for {dataset}")
    completed = {
        (row["arm"], int(row["training_seed"])) for row in rows
        if row["stage2_selection_sha256"] == selection_hash
    }
    for arm in ("raw", "fmt"):
        train_x, evaluation_x = _prepare_inputs(
            train_records, evaluation_records, arm, feature["name"], device
        )
        for seed_value in spec["final_training_seeds"]:
            seed = int(seed_value)
            if (arm, seed) in completed:
                continue
            train_mu, evaluation_mu, losses = _train(
                train_x, evaluation_x, architecture, source, seed, device
            )
            calibration_mu = evaluation_mu[:calibration_count]
            confirmation_mu = evaluation_mu[calibration_count:]
            vortex_cluster, calibration_metrics, confirmation_metrics = _score(
                train_mu, calibration_mu, confirmation_mu,
                calibration_reference, confirmation_reference, search,
            )
            row = {
                "experiment": spec["experiment"],
                "stage2_selection_sha256": selection_hash,
                "dataset": dataset, "group": group_name, "arm": arm,
                "feature_id": feature["id"], "fmt_feature": feature["name"],
                "architecture": architecture["id"], "training_seed": seed,
                "cluster_as_vortex": vortex_cluster,
                **{f"calibration_{key}": value
                   for key, value in calibration_metrics.items()},
                **{f"confirmation_{key}": value
                   for key, value in confirmation_metrics.items()},
                **losses,
            }
            rows.append(row)
            _write_csv(target, rows)
            completed.add((arm, seed))
            print(
                f"Task2 final {dataset}/{arm}/seed={seed}: "
                f"F1={confirmation_metrics['f1']:.5f}", flush=True,
            )
    return target


def summarize(config_path: str) -> Path:
    spec = _load_spec(config_path)
    search, _, selection_hash = _frozen_state(spec)
    rows = []
    expected = 2 * len(spec["final_training_seeds"])
    for dataset in search["datasets"]:
        values = _read_csv(
            Path(spec["output_root"]) / "shards" / f"{dataset}.csv"
        )
        if len(values) != expected:
            raise RuntimeError(
                f"Task2 final shard {dataset} has {len(values)} rows, expected {expected}"
            )
        if {row["stage2_selection_sha256"] for row in values} != {selection_hash}:
            raise RuntimeError(f"Task2 selection hash mismatch for {dataset}")
        rows.extend(values)
    output = Path(spec["output_root"])
    _write_csv(output / "per_run.csv", rows)
    datasets = {}
    paired_gains = []
    for dataset in search["datasets"]:
        selected = [row for row in rows if row["dataset"] == dataset]
        by_arm = {}
        for arm in ("raw", "fmt"):
            arm_rows = [row for row in selected if row["arm"] == arm]
            by_arm[arm] = float(np.mean([
                float(row["confirmation_f1"]) for row in arm_rows
            ]))
        gains = []
        for seed in spec["final_training_seeds"]:
            raw = next(float(row["confirmation_f1"]) for row in selected
                       if row["arm"] == "raw"
                       and int(row["training_seed"]) == int(seed))
            fmt = next(float(row["confirmation_f1"]) for row in selected
                       if row["arm"] == "fmt"
                       and int(row["training_seed"]) == int(seed))
            gains.append(fmt - raw)
        paired_gains.extend(gains)
        datasets[dataset] = {
            "raw_f1": by_arm["raw"], "fmt_f1": by_arm["fmt"],
            "fmt_minus_raw_f1": by_arm["fmt"] - by_arm["raw"],
            "seed_gains": gains,
        }
    macro_gain = float(np.mean([
        row["fmt_minus_raw_f1"] for row in datasets.values()
    ]))
    summary = {
        "experiment": spec["experiment"],
        "stage2_selection_sha256": selection_hash,
        "comparison": "same VAE hyperparameters and paired seed: FMT+VAE minus Raw+VAE",
        "datasets": datasets,
        "dataset_macro_f1_gain": macro_gain,
        "all_paired_seed_dataset_gains_mean": float(np.mean(paired_gains)),
        "target_gain": float(spec["target_dataset_macro_f1_gain"]),
        "target_reached": macro_gain >= float(spec["target_dataset_macro_f1_gain"]),
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
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
