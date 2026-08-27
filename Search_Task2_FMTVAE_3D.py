"""Development-only family search for the 3D Task2 FMT+VAE gain.

The Raw and FMT arms always share one VAE architecture and one training seed.
Only development train/validation ordinals are opened.  Confirmation and
outer-development files are inaccessible to candidate training and selection.
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

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from Run_Task2_3D_Main import _prepare_inputs
from Verify_HighReVAE import _train


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "groups", "datasets", "features",
        "architectures", "screen_seeds", "splits", "task1_development_f1",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    dataset_ids = list(spec["datasets"])
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("datasets must be unique")
    memberships = [
        dataset
        for group in spec["groups"].values()
        for dataset in group["datasets"]
    ]
    if sorted(memberships) != sorted(dataset_ids):
        raise ValueError("groups must partition datasets exactly once")
    for collection, label in (
        (spec["features"], "feature"), (spec["architectures"], "architecture")
    ):
        ids = [str(row["id"]) for row in collection]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {label} ids")
    train = {int(value) for value in spec["splits"]["selection_train"]}
    validation = {
        int(value) for value in spec["splits"]["selection_validation"]
    }
    outer = {int(value) for value in spec["splits"].get("outer", [])}
    if train & validation or (train | validation) & outer:
        raise ValueError("Task2 train, validation, and outer ordinals must be disjoint")
    if set(spec["task1_development_f1"]) != set(dataset_ids):
        raise ValueError("Task1 development F1 must cover every dataset")
    for group in spec["groups"].values():
        root = str(group["development_cache"]).lower()
        if "confirmation" in root or "test" in root:
            raise ValueError(f"development search cannot read held-out root: {root}")
    return spec


def _group_for_dataset(spec: dict, dataset: str) -> tuple[str, dict]:
    matches = [
        (name, group) for name, group in spec["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} matched {len(matches)} groups")
    return matches[0]


def _load_development(spec: dict, dataset: str) -> tuple[dict[int, dict], EasyConfig]:
    _, group = _group_for_dataset(spec, dataset)
    required = sorted(
        {int(value) for value in spec["splits"]["selection_train"]}
        | {int(value) for value in spec["splits"]["selection_validation"]}
    )
    records = load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(spec.get("expected_slices", 10)),
        ordinals=required,
    )
    return {int(record["ordinal"]): record for record in records}, EasyConfig(
        group["source_config"]
    )


def _split_records(records: dict[int, dict], ordinals) -> list[dict]:
    return [records[int(ordinal)] for ordinal in ordinals]


def _latent_metrics(train_mu, validation_mu, reference, spec) -> dict:
    model = KMeans(
        n_clusters=2,
        random_state=int(spec["kmeans_seed"]),
        n_init=int(spec["kmeans_n_init"]),
    ).fit(train_mu)
    labels = model.predict(validation_mu)
    vortex_cluster = calibrate_vortex_cluster(reference, labels)
    return binary_cluster_metrics(reference, labels, vortex_cluster)


def _result_path(spec: dict, arm: str, dataset: str, architecture: dict,
                 feature: dict | None = None) -> Path:
    root = Path(spec["output_root"]) / "stage1" / arm / dataset
    if arm == "raw":
        return root / f"{architecture['id']}.csv"
    if feature is None:
        raise ValueError("FMT result path requires a feature")
    return root / str(feature["id"]) / f"{architecture['id']}.csv"


def _run_arm(config_path: str, dataset: str, architecture_index: int,
             feature_index: int | None = None) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    architecture = dict(spec["architectures"][int(architecture_index)])
    feature = None if feature_index is None else dict(
        spec["features"][int(feature_index)]
    )
    arm = "raw" if feature is None else "fmt"
    path = _result_path(spec, arm, dataset, architecture, feature)
    rows = _read_csv(path)
    completed = {int(row["training_seed"]) for row in rows}
    records, source = _load_development(spec, dataset)
    train_records = _split_records(records, spec["splits"]["selection_train"])
    validation_records = _split_records(
        records, spec["splits"]["selection_validation"]
    )
    reference = stack_reference(validation_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_name = "fmt_all" if feature is None else str(feature["name"])
    train_x, validation_x = _prepare_inputs(
        train_records, validation_records, arm, feature_name, device
    )
    if architecture.get("pca_init", False):
        latent = int(architecture["latent_dim"])
        if latent > train_x.shape[1]:
            raise ValueError(
                f"{architecture['id']} latent={latent} exceeds {arm} input "
                f"dimension={train_x.shape[1]}"
            )
    for seed in spec["screen_seeds"]:
        seed = int(seed)
        if seed in completed:
            continue
        train_mu, validation_mu, losses = _train(
            train_x, validation_x, architecture, source, seed, device
        )
        metrics = _latent_metrics(
            train_mu, validation_mu, reference, spec
        )
        group_name, _ = _group_for_dataset(spec, dataset)
        row = {
            "experiment": spec["experiment"],
            "stage": "stage1",
            "group": group_name,
            "dataset": dataset,
            "arm": arm,
            "feature_id": "" if feature is None else feature["id"],
            "fmt_feature": "" if feature is None else feature["name"],
            "architecture": architecture["id"],
            "training_seed": seed,
            "input_dim": int(train_x.shape[1]),
            **metrics,
            **losses,
        }
        rows.append(row)
        _write_csv(path, rows)
        completed.add(seed)
        print(
            f"{dataset}/{arm}/{feature_name}/{architecture['id']}/seed={seed}: "
            f"F1={metrics['f1']:.5f}", flush=True,
        )
    return path


def _decode_job(spec: dict, arm: str, job_index: int) -> tuple:
    datasets = list(spec["datasets"])
    architectures = list(spec["architectures"])
    index = int(job_index)
    if arm == "raw":
        count = len(datasets) * len(architectures)
        if not 0 <= index < count:
            raise IndexError(f"raw job index {index} outside [0,{count})")
        dataset_index, architecture_index = divmod(index, len(architectures))
        return datasets[dataset_index], architecture_index, None
    features = list(spec["features"])
    per_dataset = len(features) * len(architectures)
    count = len(datasets) * per_dataset
    if not 0 <= index < count:
        raise IndexError(f"FMT job index {index} outside [0,{count})")
    dataset_index, remainder = divmod(index, per_dataset)
    feature_index, architecture_index = divmod(remainder, len(architectures))
    return datasets[dataset_index], architecture_index, feature_index


def run_job(config_path: str, arm: str, job_index: int) -> Path:
    spec = _load_spec(config_path)
    dataset, architecture_index, feature_index = _decode_job(
        spec, arm, job_index
    )
    return _run_arm(
        config_path, dataset, architecture_index, feature_index
    )


def _paired_candidate(spec: dict, group_name: str, feature: dict,
                      architecture: dict) -> dict:
    datasets = spec["groups"][group_name]["datasets"]
    seeds = [int(value) for value in spec["screen_seeds"]]
    per_dataset = {}
    seed_gains = {seed: [] for seed in seeds}
    for dataset in datasets:
        fmt_rows = _read_csv(_result_path(
            spec, "fmt", dataset, architecture, feature
        ))
        raw_rows = _read_csv(_result_path(
            spec, "raw", dataset, architecture
        ))
        fmt = {int(row["training_seed"]): float(row["f1"]) for row in fmt_rows}
        raw = {int(row["training_seed"]): float(row["f1"]) for row in raw_rows}
        if set(fmt) != set(seeds) or set(raw) != set(seeds):
            raise RuntimeError(
                f"incomplete Task2 candidate {dataset}/{feature['id']}/"
                f"{architecture['id']}"
            )
        fmt_values = np.asarray([fmt[seed] for seed in seeds])
        raw_values = np.asarray([raw[seed] for seed in seeds])
        for seed in seeds:
            seed_gains[seed].append(fmt[seed] - raw[seed])
        per_dataset[dataset] = {
            "raw_f1": float(raw_values.mean()),
            "fmt_f1": float(fmt_values.mean()),
            "fmt_minus_raw_f1": float((fmt_values - raw_values).mean()),
            "fmt_minus_task1_f1": float(
                fmt_values.mean() - float(spec["task1_development_f1"][dataset])
            ),
        }
    raw_macro = float(np.mean([row["raw_f1"] for row in per_dataset.values()]))
    fmt_macro = float(np.mean([row["fmt_f1"] for row in per_dataset.values()]))
    mean_gain = fmt_macro - raw_macro
    seed_macro_gains = {
        str(seed): float(np.mean(values)) for seed, values in seed_gains.items()
    }
    task1_guard = min(
        row["fmt_minus_task1_f1"] for row in per_dataset.values()
    ) >= -float(spec.get("selection", {}).get("allowed_fmt_below_task1", 0.02))
    return {
        "group": group_name,
        "feature_id": feature["id"],
        "fmt_feature": feature["name"],
        "architecture": architecture["id"],
        "dataset_count": len(datasets),
        "raw_f1_macro": raw_macro,
        "fmt_f1_macro": fmt_macro,
        "fmt_minus_raw_f1_macro": mean_gain,
        "worst_seed_f1_gain": min(seed_macro_gains.values()),
        "all_seed_gains_positive": min(seed_macro_gains.values()) > 0.0,
        "absolute_fmt_guard_passed": task1_guard,
        "seed_gains_json": json.dumps(seed_macro_gains, sort_keys=True),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
    }


def select(config_path: str) -> Path:
    spec = _load_spec(config_path)
    rows = []
    selected = {}
    top_k = int(spec.get("selection", {}).get("top_k", 3))
    for group_name in spec["groups"]:
        candidates = [
            _paired_candidate(spec, group_name, feature, architecture)
            for feature in spec["features"]
            for architecture in spec["architectures"]
        ]
        ranked = sorted(
            candidates,
            key=lambda row: (
                bool(row["absolute_fmt_guard_passed"]),
                float(row["fmt_minus_raw_f1_macro"]),
                float(row["worst_seed_f1_gain"]),
                float(row["fmt_f1_macro"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["rank_within_group"] = rank
            rows.append(row)
        unique_features = []
        seen_features = set()
        for row in ranked:
            if row["feature_id"] in seen_features:
                continue
            unique_features.append(row)
            seen_features.add(row["feature_id"])
            if len(unique_features) == top_k:
                break
        if len(unique_features) != top_k:
            raise RuntimeError(
                f"group {group_name} produced only {len(unique_features)} "
                f"unique feature recipes for top_k={top_k}"
            )
        selected[group_name] = unique_features
    output = Path(spec["output_root"])
    _write_csv(output / "stage1_leaderboard.csv", rows)
    primary = {group: values[0] for group, values in selected.items()}
    dataset_rows = []
    for group, row in primary.items():
        details = json.loads(row["datasets_json"])
        dataset_rows.extend({"group": group, "dataset": dataset, **metrics}
                            for dataset, metrics in details.items())
    development_gain = float(np.mean([
        row["fmt_minus_raw_f1"] for row in dataset_rows
    ]))
    payload = {
        "experiment": spec["experiment"],
        "selection_rule": (
            "family-specific: pass FMT absolute Task1 guard, then maximize paired "
            "same-VAE development F1 gain; tie-break by worst seed and FMT F1"
        ),
        "opened_ordinals": sorted(
            set(spec["splits"]["selection_train"])
            | set(spec["splits"]["selection_validation"])
        ),
        "outer_ordinals_opened": False,
        "confirmation_opened": False,
        "top_k_by_group": selected,
        "primary_by_group": primary,
        "development_dataset_macro_f1_gain": development_gain,
        "development_target_gain": float(
            spec.get("selection", {}).get("target_dataset_macro_f1_gain", 0.15)
        ),
        "development_target_reached": development_gain >= float(
            spec.get("selection", {}).get("target_dataset_macro_f1_gain", 0.15)
        ),
        "dataset_details": dataset_rows,
    }
    target = output / "stage1_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("raw", "fmt", "select"), required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--architecture-index", type=int)
    parser.add_argument("--feature-index", type=int)
    args = parser.parse_args()
    if args.mode == "select":
        select(args.config)
        return
    if args.job_index is not None:
        run_job(args.config, args.mode, args.job_index)
        return
    if args.dataset is None or args.architecture_index is None:
        parser.error("arm mode requires --job-index or dataset/architecture-index")
    _run_arm(
        args.config, args.dataset, args.architecture_index,
        args.feature_index if args.mode == "fmt" else None,
    )


if __name__ == "__main__":
    main()
