"""Reproduce the Task2 paper experiment under several whole-field IVD labels.

The VAE and KMeans are label-free.  Each dataset/method/seed is therefore
trained exactly once; its frozen cluster predictions are scored against every
requested percentile.  Only the development calibration slices are used to
map the anonymous cluster ID to "vortex" for each label definition.
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
from Run_Task2_3D_Main import _architecture, _cache_dir, _prepare_inputs
from Verify_HighReVAE import _train


METHODS = {"raw": "Raw+VAE", "fmt": "FMT+VAE"}


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


def _label_group(main_spec: dict, split: str, dataset: str) -> str:
    suffix = "new2" if dataset in main_spec.get("cache_overrides", {}) else "old8"
    return f"{split}_{suffix}"


def _label_vector(master: dict, main_spec: dict, percentile: float, split: str,
                  dataset: str, records: list[dict]) -> np.ndarray:
    group = _label_group(main_spec, split, dataset)
    root = (
        Path(master["output_dir"]) / "labels" / percentile_tag(percentile)
        / group / "labels" / dataset
    )
    labels = []
    for record in records:
        path = root / record["path"].name
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as cached:
            values = np.asarray(cached["labels"], dtype=bool)
            metadata = json.loads(str(cached["metadata_json"]))
        if len(values) != len(record["reference"]):
            raise ValueError(f"label length mismatch in {path}")
        source_name = Path(str(metadata["source_cache"]).replace("\\", "/")).name
        if source_name != record["path"].name:
            raise ValueError(f"label/source mismatch in {path}")
        if float(metadata["label_value"]) != float(percentile):
            raise ValueError(f"wrong percentile metadata in {path}")
        labels.append(values)
    return np.concatenate(labels)


def _prediction_path(output: Path, dataset: str, method: str, seed: int) -> Path:
    return output / "predictions" / f"{dataset}_{method}_seed{seed}.npz"


def _load_predictions(path: Path, calibration_count: int, test_count: int):
    if not path.exists():
        return None
    with np.load(path) as cached:
        calibration = np.asarray(cached["calibration_labels"], dtype=np.int64)
        test = np.asarray(cached["test_labels"], dtype=np.int64)
        losses = json.loads(str(cached["losses_json"]))
    if len(calibration) != calibration_count or len(test) != test_count:
        raise RuntimeError(f"stale Task2 predictions: {path}")
    return calibration, test, losses


def _save_predictions(path: Path, calibration: np.ndarray, test: np.ndarray,
                      losses: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        calibration_labels=np.asarray(calibration, dtype=np.int8),
        test_labels=np.asarray(test, dtype=np.int8),
        losses_json=np.asarray(json.dumps(losses, sort_keys=True)),
    )


def run_group(config_path: str, group: str, resume: bool = False) -> Path:
    master = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    main_path = Path(master["task2"]["main_config"])
    main_spec = yaml.safe_load(main_path.read_text(encoding="utf-8"))
    if group not in main_spec["groups"]:
        raise ValueError(f"unknown Task2 group {group!r}")
    output = Path(master["task2"]["output_dir"]) / "groups" / group
    output.mkdir(parents=True, exist_ok=True)
    snapshot = {"sweep": master, "frozen_task2_main": main_spec}
    snapshot_path = output / "config_snapshot.yaml"
    if snapshot_path.exists():
        previous = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        if previous != snapshot:
            raise RuntimeError(f"configuration changed for {output}")
    snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")

    datasets = main_spec["groups"][group]["datasets"]
    development = {
        dataset: load_cache_records(
            _cache_dir(main_spec, "development", dataset), 10
        ) for dataset in datasets
    }
    confirmation_count = int(main_spec["splits"]["confirmation_count"])
    confirmation = {
        dataset: load_cache_records(
            _cache_dir(main_spec, "confirmation", dataset), confirmation_count
        ) for dataset in datasets
    }
    source = EasyConfig(str(main_spec["source_config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    percentiles = all_percentiles(master)
    result_path = output / "per_run.csv"
    rows = _read_csv(result_path) if resume else []
    completed = {
        (row["dataset"], row["method"], int(row["training_seed"]),
         float(row["ivd_percentile"])) for row in rows
    }

    train_ids = main_spec["splits"]["final_train"]
    calibration_ids = main_spec["splits"]["cluster_calibration"]
    fmt_feature = main_spec["groups"][group]["fmt_feature"]
    architecture_id = main_spec["groups"][group]["fixed_architecture"]
    architecture = _architecture(main_spec, architecture_id)
    for dataset in datasets:
        train_records = [development[dataset][index] for index in train_ids]
        calibration_records = [
            development[dataset][index] for index in calibration_ids
        ]
        test_records = confirmation[dataset]
        evaluate_records = [*calibration_records, *test_records]
        calibration_count = sum(len(record["reference"]) for record in calibration_records)
        test_count = sum(len(record["reference"]) for record in test_records)
        references = {
            value: (
                _label_vector(
                    master, main_spec, value, "development", dataset,
                    calibration_records,
                ),
                _label_vector(
                    master, main_spec, value, "confirmation", dataset,
                    test_records,
                ),
            ) for value in percentiles
        }
        for method in METHODS:
            train_x, evaluate_x = _prepare_inputs(
                train_records, evaluate_records, method, fmt_feature, device
            )
            for seed_value in main_spec["final_training_seeds"]:
                seed = int(seed_value)
                keys = {
                    (dataset, METHODS[method], seed, float(value))
                    for value in percentiles
                }
                if resume and keys <= completed:
                    print(f"cached {group}/{dataset}/{method}/seed={seed}", flush=True)
                    continue
                prediction_path = _prediction_path(output, dataset, method, seed)
                cached = _load_predictions(
                    prediction_path, calibration_count, test_count
                ) if resume else None
                if cached is None:
                    train_mu, evaluate_mu, losses = _train(
                        train_x, evaluate_x, architecture, source, seed, device
                    )
                    kmeans = KMeans(
                        n_clusters=2,
                        random_state=int(main_spec["kmeans_seed"]),
                        n_init=int(main_spec["kmeans_n_init"]),
                    ).fit(train_mu)
                    labels = kmeans.predict(evaluate_mu)
                    calibration_labels = labels[:calibration_count]
                    test_labels = labels[calibration_count:]
                    _save_predictions(
                        prediction_path, calibration_labels, test_labels, losses
                    )
                else:
                    calibration_labels, test_labels, losses = cached
                rows = [
                    row for row in rows
                    if not (
                        row["dataset"] == dataset
                        and row["method"] == METHODS[method]
                        and int(row["training_seed"]) == seed
                    )
                ]
                for value in percentiles:
                    calibration_reference, test_reference = references[value]
                    vortex_cluster = calibrate_vortex_cluster(
                        calibration_reference, calibration_labels
                    )
                    score = binary_cluster_metrics(
                        test_reference, test_labels, vortex_cluster
                    )
                    rows.append({
                        "experiment": master["experiment"],
                        "dataset": dataset,
                        "group": group,
                        "method": METHODS[method],
                        "fmt_feature": fmt_feature,
                        "architecture": architecture_id,
                        "training_seed": seed,
                        "ivd_percentile": float(value),
                        "percentile_tag": percentile_tag(value),
                        "cluster_as_vortex": int(vortex_cluster),
                        **score,
                        **losses,
                    })
                    completed.add((dataset, METHODS[method], seed, float(value)))
                _write_csv(result_path, rows)
                p_values = ", ".join(
                    f"{percentile_tag(value)}="
                    f"{next(float(row['f1']) for row in rows if row['dataset'] == dataset and row['method'] == METHODS[method] and int(row['training_seed']) == seed and float(row['ivd_percentile']) == value):.4f}"
                    for value in percentiles
                )
                print(
                    f"DONE {group}/{dataset}/{METHODS[method]}/seed={seed}: {p_values}",
                    flush=True,
                )
    return result_path


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def summarize(config_path: str) -> Path:
    master = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    main_spec = yaml.safe_load(
        Path(master["task2"]["main_config"]).read_text(encoding="utf-8")
    )
    output = Path(master["task2"]["output_dir"])
    all_rows = []
    for group, group_spec in main_spec["groups"].items():
        rows = _read_csv(output / "groups" / group / "per_run.csv")
        expected = (
            len(group_spec["datasets"]) * 2
            * len(main_spec["final_training_seeds"]) * len(all_percentiles(master))
        )
        if len(rows) != expected:
            raise RuntimeError(f"{group}: expected {expected} rows, found {len(rows)}")
        all_rows.extend(rows)
    _write_csv(output / "per_run.csv", all_rows)

    dataset_rows = []
    for value in all_percentiles(master):
        for group, group_spec in main_spec["groups"].items():
            for dataset in group_spec["datasets"]:
                selected = [
                    row for row in all_rows
                    if row["dataset"] == dataset
                    and float(row["ivd_percentile"]) == value
                ]
                raw = [row for row in selected if row["method"] == "Raw+VAE"]
                fmt = [row for row in selected if row["method"] == "FMT+VAE"]
                raw_by_seed = {
                    int(row["training_seed"]): float(row["f1"]) for row in raw
                }
                fmt_by_seed = {
                    int(row["training_seed"]): float(row["f1"]) for row in fmt
                }
                if set(raw_by_seed) != set(fmt_by_seed):
                    raise RuntimeError(f"unpaired Task2 seeds for {dataset}/{value}")
                gains = np.asarray([
                    fmt_by_seed[seed] - raw_by_seed[seed]
                    for seed in sorted(raw_by_seed)
                ])
                dataset_rows.append({
                    "ivd_percentile": value,
                    "percentile_tag": percentile_tag(value),
                    "dataset": dataset,
                    "family": group,
                    "raw_f1_mean": _mean(raw, "f1"),
                    "raw_f1_std": float(np.std(list(raw_by_seed.values()), ddof=1)),
                    "fmt_f1_mean": _mean(fmt, "f1"),
                    "fmt_f1_std": float(np.std(list(fmt_by_seed.values()), ddof=1)),
                    "paired_f1_gain_mean": float(gains.mean()),
                    "paired_f1_gain_std": float(gains.std(ddof=1)),
                    "positive_seed_count": int((gains > 0).sum()),
                    "raw_positive_fraction": _mean(raw, "positive_fraction"),
                    "fmt_predicted_positive_fraction": _mean(
                        fmt, "predicted_positive_fraction"
                    ),
                })
    _write_csv(output / "paper_table.csv", dataset_rows)

    summaries = []
    for value in all_percentiles(master):
        selected = [row for row in dataset_rows if row["ivd_percentile"] == value]
        family_gains = []
        for family in main_spec["groups"]:
            family_rows = [row for row in selected if row["family"] == family]
            family_gains.append(float(np.mean([
                row["paired_f1_gain_mean"] for row in family_rows
            ])))
        summaries.append({
            "ivd_percentile": value,
            "percentile_tag": percentile_tag(value),
            "dataset_macro_raw_f1": float(np.mean([
                row["raw_f1_mean"] for row in selected
            ])),
            "dataset_macro_fmt_f1": float(np.mean([
                row["fmt_f1_mean"] for row in selected
            ])),
            "dataset_macro_f1_gain": float(np.mean([
                row["paired_f1_gain_mean"] for row in selected
            ])),
            "family_macro_f1_gain": float(np.mean(family_gains)),
            "positive_dataset_count": int(sum(
                row["paired_f1_gain_mean"] > 0 for row in selected
            )),
            "positive_family_count": int(sum(value_ > 0 for value_ in family_gains)),
            "mean_reference_positive_fraction": float(np.mean([
                row["raw_positive_fraction"] for row in selected
            ])),
        })
    _write_csv(output / "percentile_summary.csv", summaries)

    audit = {"published_p95_available": False}
    published_path = Path(master["task2"]["published_p95_runs"])
    if published_path.exists():
        published = _read_csv(published_path)
        new_p95 = [
            row for row in all_rows
            if float(row["ivd_percentile"]) == float(master["audit_percentile"])
        ]
        old = {
            (row["dataset"], row["method"], int(row["training_seed"])): float(row["f1"])
            for row in published
        }
        new = {
            (row["dataset"], row["method"], int(row["training_seed"])): float(row["f1"])
            for row in new_p95
        }
        if set(old) != set(new):
            raise RuntimeError("published/new Task2 p95 run coverage differs")
        differences = np.asarray([new[key] - old[key] for key in sorted(old)])
        audit = {
            "published_p95_available": True,
            "published_path": str(published_path),
            "run_count": len(differences),
            "mean_signed_f1_difference": float(differences.mean()),
            "max_absolute_f1_difference": float(np.abs(differences).max()),
        }
    (output / "p95_reproduction_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    requested = {
        float(value) for value in master["requested_percentiles"]
    }
    requested_rows = [
        row for row in summaries if row["ivd_percentile"] in requested
    ]
    largest_dataset = max(
        requested_rows, key=lambda row: row["dataset_macro_f1_gain"]
    )
    largest_family = max(
        requested_rows, key=lambda row: row["family_macro_f1_gain"]
    )
    lines = [
        "# Task2 IVD percentile sensitivity",
        "",
        "每个 dataset/method/seed 的 VAE 与 KMeans 只训练一次；不同百分位只改变 "
        "development 匿名簇映射和 confirmation 评分标签。",
        "",
        "| IVD label | Raw+VAE F1 | FMT+VAE F1 | dataset-macro gain | family-macro gain | positive datasets | positive families |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['percentile_tag']} | {row['dataset_macro_raw_f1']:.4f} | "
            f"{row['dataset_macro_fmt_f1']:.4f} | "
            f"{row['dataset_macro_f1_gain']:+.4f} | "
            f"{row['family_macro_f1_gain']:+.4f} | "
            f"{row['positive_dataset_count']}/10 | "
            f"{row['positive_family_count']}/7 |"
        )
    lines.extend([
        "",
        f"请求范围内最大 dataset-macro F1 增益：{largest_dataset['percentile_tag']} "
        f"({largest_dataset['dataset_macro_f1_gain']:+.4f})。",
        f"请求范围内最大 family-macro F1 增益：{largest_family['percentile_tag']} "
        f"({largest_family['family_macro_f1_gain']:+.4f})。",
        "",
        "该最大值是敏感性分析结果，不单独证明该百分位具有最佳物理定义。",
    ])
    (output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_payload = {
        "experiment": master["experiment"],
        "task": "Task2",
        "training_reused_across_percentiles": True,
        "percentile_summary": summaries,
        "largest_requested_dataset_macro_gain": largest_dataset,
        "largest_requested_family_macro_gain": largest_family,
        "p95_reproduction_audit": audit,
    }
    (output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2), flush=True)
    return output / "percentile_summary.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Ablation_Task23IVDPercentile_1.1.yaml"
    )
    parser.add_argument("--group")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.config)
    elif args.group:
        run_group(args.config, args.group, args.resume)
    else:
        raise SystemExit("provide --group NAME or --summarize")
