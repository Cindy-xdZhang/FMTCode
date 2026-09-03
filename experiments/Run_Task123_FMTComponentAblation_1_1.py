"""Run fixed-downstream FMT component ablations for 3D Task1--Task3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.RobustnessFeatures_3D import (
    fmt_kin4_ablation_mask,
    masked_fmt_kin4_feature_matrix,
)
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
    fit_kmeans_transform,
)
from experiments.Verify_HighReVAE import _train
from experiments import Search_Task3_FMTResidual_3D as task3_search


def _read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _spec(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"experiment", "output_root", "canonical_variants", "task1", "task2", "task3"}
    if required - set(value):
        raise ValueError(f"component-ablation config misses {sorted(required - set(value))}")
    return value


def _task1_source_for_dataset(source: dict, dataset: str) -> dict:
    matches = [row for row in source["sources"].values() if dataset in row["datasets"]]
    if len(matches) != 1:
        raise ValueError(f"Task1 dataset {dataset} matched {len(matches)} sources")
    return matches[0]


def _by_ordinal(records):
    result = {int(row["ordinal"]): row for row in records}
    if len(result) != len(records):
        raise RuntimeError("duplicate cache ordinal")
    return result


def _masked_stack(records, variant, device):
    return np.concatenate([
        masked_fmt_kin4_feature_matrix(record, variant, device)
        for record in records
    ])


def run_task1(config_path: str | Path, job_index: int) -> Path:
    spec = _spec(config_path)
    task = spec["task1"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets):
        raise IndexError("Task1 component-ablation array index out of range")
    dataset = datasets[index]
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    required = sorted(set(task["final_train"]) | set(task["cluster_calibration"]))
    item = _task1_source_for_dataset(source, dataset)
    development = _by_ordinal(load_cache_records(
        Path(item["development_cache"]) / dataset,
        expected_count=int(source["development_count"]), ordinals=required,
    ))
    confirmation = load_cache_records(
        Path(item["confirmation_cache"]) / dataset,
        expected_count=int(source["confirmation_count"]),
    )
    train = [development[int(value)] for value in task["final_train"]]
    calibration = [development[int(value)] for value in task["cluster_calibration"]]
    calibration_reference = stack_reference(calibration)
    confirmation_reference = stack_reference(confirmation)
    full_train = _masked_stack(train, "full", device)
    full_calibration = _masked_stack(calibration, "full", device)
    full_confirmation = _masked_stack(confirmation, "full", device)
    for variant in spec["canonical_variants"]:
        name = variant["id"]
        mask = fmt_kin4_ablation_mask(name)[None, :]
        train_x = np.ascontiguousarray(full_train * mask)
        calibration_x = np.ascontiguousarray(full_calibration * mask)
        confirmation_x = np.ascontiguousarray(full_confirmation * mask)
        for seed_value in task["kmeans_seeds"]:
            seed = int(seed_value)
            fitted = fit_kmeans_transform(
                train_x, int(task["pca_dim"]), seed, int(task["kmeans_n_init"])
            )
            cluster = calibrate_vortex_cluster(
                calibration_reference, fitted.predict(calibration_x)
            )
            score = binary_cluster_metrics(
                confirmation_reference, fitted.predict(confirmation_x), cluster
            )
            rows.append({
                "experiment": spec["experiment"], "task": "Task1",
                "dataset": dataset, "variant": name,
                "removed_component": variant["removes"],
                "input_dim": int(train_x.shape[1]),
                "pca_dim": int(task["pca_dim"]), "seed": seed,
                "cluster_as_vortex": int(cluster), **score,
            })
    target = Path(spec["output_root"]) / "task1" / "shards" / f"{dataset}.csv"
    _write_csv(target, rows)
    return target


def merge_task1(config_path: str | Path) -> Path:
    spec = _spec(config_path)
    source = yaml.safe_load(Path(
        spec["task1"]["source_config"]
    ).read_text(encoding="utf-8"))
    rows = []
    for dataset in source["datasets"]:
        rows.extend(_read_csv(
            Path(spec["output_root"]) / "task1" / "shards" / f"{dataset}.csv"
        ))
    expected = (
        len(source["datasets"]) * len(spec["canonical_variants"])
        * len(spec["task1"]["kmeans_seeds"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"incomplete Task1 ablation: {len(rows)} != {expected}")
    target = Path(spec["output_root"]) / "task1" / "per_run.csv"
    _write_csv(target, rows)
    return target


def _task2_group(source: dict, dataset: str) -> dict:
    matches = [row for row in source["groups"].values() if dataset in row["datasets"]]
    if len(matches) != 1:
        raise ValueError(f"Task2 dataset {dataset} matched {len(matches)} groups")
    return matches[0]


def _architecture(source: dict, identifier: str) -> dict:
    matches = [dict(row) for row in source["architectures"] if row["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"architecture {identifier} is ambiguous")
    return matches[0]


def run_task2(config_path: str | Path, job_index: int) -> Path:
    spec = _spec(config_path)
    task = spec["task2"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    datasets, variants = list(source["datasets"]), list(spec["canonical_variants"])
    index = int(job_index)
    if not 0 <= index < len(datasets) * len(variants):
        raise IndexError("Task2 component-ablation array index out of range")
    dataset_index, variant_index = divmod(index, len(variants))
    dataset, variant = datasets[dataset_index], variants[variant_index]
    group = _task2_group(source, dataset)
    required = sorted({
        int(value) for key in ("train", "cluster_calibration", "confirmation")
        for value in task[key]
    })
    records = _by_ordinal(load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(source["expected_slices"]), ordinals=required,
    ))
    train = [records[int(value)] for value in task["train"]]
    calibration = [records[int(value)] for value in task["cluster_calibration"]]
    confirmation = [records[int(value)] for value in task["confirmation"]]
    evaluate = [*calibration, *confirmation]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = _masked_stack(train, variant["id"], device)
    evaluate_x = _masked_stack(evaluate, variant["id"], device)
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x).astype(np.float32)
    evaluate_x = scaler.transform(evaluate_x).astype(np.float32)
    architecture = _architecture(source, task["architecture"])
    source_config = EasyConfig(group["source_config"])
    calibration_count = sum(len(row["reference"]) for row in calibration)
    calibration_reference = stack_reference(calibration)
    confirmation_reference = stack_reference(confirmation)
    target = (
        Path(spec["output_root"]) / "task2" / "shards"
        / f"{dataset}_{variant['id']}.csv"
    )
    rows = _read_csv(target)
    completed = {int(row["seed"]) for row in rows}
    for seed_value in task["training_seeds"]:
        seed = int(seed_value)
        if seed in completed:
            continue
        train_mu, evaluate_mu, losses = _train(
            train_x, evaluate_x, architecture, source_config, seed, device
        )
        kmeans = KMeans(
            n_clusters=2, random_state=int(task["kmeans_seed"]),
            n_init=int(task["kmeans_n_init"]),
        ).fit(train_mu)
        cluster = calibrate_vortex_cluster(
            calibration_reference, kmeans.predict(evaluate_mu[:calibration_count])
        )
        score = binary_cluster_metrics(
            confirmation_reference,
            kmeans.predict(evaluate_mu[calibration_count:]), cluster,
        )
        rows.append({
            "experiment": spec["experiment"], "task": "Task2",
            "dataset": dataset, "variant": variant["id"],
            "removed_component": variant["removes"],
            "architecture": task["architecture"], "input_dim": int(train_x.shape[1]),
            "seed": seed, "cluster_as_vortex": int(cluster), **score, **losses,
        })
        _write_csv(target, rows)
        completed.add(seed)
        print(f"{dataset}/{variant['id']}/seed{seed}: F1={score['f1']:.5f}", flush=True)
    return target


def run_task3(config_path: str | Path, job_index: int) -> Path:
    spec = _spec(config_path)
    task = spec["task3"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    datasets, variants = list(source["datasets"]), list(task["variants"])
    index = int(job_index)
    if not 0 <= index < len(datasets) * len(variants):
        raise IndexError("Task3 component-ablation array index out of range")
    dataset_index, variant_index = divmod(index, len(variants))
    dataset, variant = datasets[dataset_index], variants[variant_index]
    candidate = next(
        dict(row) for row in source["candidates"]
        if row["id"] == variant["source_candidate"]
    )
    _, group = task3_search._group_for_dataset(source, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    required = sorted(set(task["train"]) | set(task["validation"]) | set(task["confirmation"]))
    records = task3_search._load_records(
        source, dataset, candidate, device, ordinals=required
    )
    train = task3_search._stack_split(records, task["train"])
    validation = task3_search._stack_split(records, task["validation"])
    confirmation = task3_search._stack_split(records, task["confirmation"])
    # Residual-head training seeds may differ from the frozen Raw backbone
    # seeds (position-paired ``backbone_seeds``); 1.1 used identical lists.
    seeds = [int(value) for value in task["seeds"]]
    backbone_seeds = [int(value) for value in task.get("backbone_seeds", seeds)]
    if len(backbone_seeds) != len(seeds):
        raise ValueError("task3.backbone_seeds must pair one-to-one with task3.seeds")
    raw_stats = task3_search._frozen_raw_normalization(
        group, dataset, backbone_seeds[0]
    )
    train, validation, confirmation, stats = task3_search._normalize_train_only(
        train, validation, confirmation, raw_stats=raw_stats
    )
    fmt_dim = int(train[1].shape[1])
    target = (
        Path(spec["output_root"]) / "task3" / "shards"
        / f"{dataset}_{variant['id']}.csv"
    )
    rows = _read_csv(target)
    completed = {(row["source"], int(row["seed"])) for row in rows}
    for seed, backbone_seed in zip(seeds, backbone_seeds):
        for auxiliary_source in task["paired_sources"]:
            if (auxiliary_source, seed) in completed:
                continue
            output_dir = (
                Path(spec["output_root"]) / "task3" / "temporary_training"
                / dataset / variant["id"] / f"seed{seed}" / auxiliary_source
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            run_spec = task3_search._candidate_spec(
                source, group, candidate, dataset, seed, auxiliary_source,
                output_dir, fmt_dim,
            )
            run_spec["experiment"] = spec["experiment"]
            run_spec["split"] = {
                "train_ordinals": list(task["train"]),
                "validation_ordinals": list(task["validation"]),
                "test_ordinals": list(task["confirmation"]),
            }
            run_spec["evaluation"] = {"test_enabled": True}
            run_spec["raw_backbone_seed"] = backbone_seed
            row = task3_search._train_one(
                run_spec, dataset, seed, (train, validation, confirmation),
                stats, device, output_dir,
            )
            row.update({
                "experiment": spec["experiment"], "task": "Task3",
                "component_variant": variant["id"],
                "interpretation": variant["interpretation"],
                "source": auxiliary_source,
                "fmt_feature": candidate["fmt_feature"], "fmt_dim": fmt_dim,
                "raw_backbone_seed": backbone_seed,
            })
            rows.append(row)
            _write_csv(target, rows)
            completed.add((auxiliary_source, seed))
            print(
                f"{dataset}/{variant['id']}/{auxiliary_source}/seed{seed}: "
                f"F1={float(row['test_f1']):.5f}", flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return target


def summarize(config_path: str | Path) -> Path:
    spec = _spec(config_path)
    output = Path(spec["output_root"])
    source2 = yaml.safe_load(Path(spec["task2"]["source_config"]).read_text(encoding="utf-8"))
    source3 = yaml.safe_load(Path(spec["task3"]["source_config"]).read_text(encoding="utf-8"))
    task1 = _read_csv(output / "task1" / "per_run.csv")
    task2 = []
    for dataset in source2["datasets"]:
        for variant in spec["canonical_variants"]:
            task2.extend(_read_csv(output / "task2" / "shards" / f"{dataset}_{variant['id']}.csv"))
    task3 = []
    for dataset in source3["datasets"]:
        for variant in spec["task3"]["variants"]:
            task3.extend(_read_csv(output / "task3" / "shards" / f"{dataset}_{variant['id']}.csv"))
    expected = {
        "Task1": len(source2["datasets"]) * len(spec["canonical_variants"]) * len(spec["task1"]["kmeans_seeds"]),
        "Task2": len(source2["datasets"]) * len(spec["canonical_variants"]) * len(spec["task2"]["training_seeds"]),
        "Task3": len(source3["datasets"]) * len(spec["task3"]["variants"]) * len(spec["task3"]["seeds"]) * 2,
    }
    observed = {"Task1": len(task1), "Task2": len(task2), "Task3": len(task3)}
    if observed != expected:
        raise RuntimeError(f"incomplete component ablation: {observed} != {expected}")
    rows = []
    for task_name, values, variant_key, source_key, metric_key in (
        ("Task1", task1, "variant", None, "f1"),
        ("Task2", task2, "variant", None, "f1"),
        ("Task3", task3, "component_variant", "source", "test_f1"),
    ):
        keys = sorted({
            (row[variant_key], "" if source_key is None else row[source_key])
            for row in values
        })
        for variant, source_name in keys:
            selected = [
                row for row in values
                if row[variant_key] == variant
                and (source_key is None or row[source_key] == source_name)
            ]
            dataset_means = []
            for dataset in sorted({row["dataset"] for row in selected}):
                dataset_means.append(np.mean([
                    float(row[metric_key]) for row in selected if row["dataset"] == dataset
                ]))
            aggregate = {
                "task": task_name, "variant": variant,
                "source": source_name, "dataset_macro_f1": float(np.mean(dataset_means)),
                "dataset_count": len(dataset_means), "run_count": len(selected),
            }
            if task_name == "Task3":
                dataset_ap = []
                for dataset in sorted({row["dataset"] for row in selected}):
                    dataset_ap.append(np.mean([
                        float(row["test_average_precision"])
                        for row in selected if row["dataset"] == dataset
                    ]))
                aggregate["dataset_macro_average_precision"] = float(
                    np.mean(dataset_ap)
                )
            rows.append(aggregate)
    _write_csv(output / "component_ablation_table.csv", rows)
    full = {
        task_name: next(row["dataset_macro_f1"] for row in rows if row["task"] == task_name and row["variant"] == variant and row["source"] == source_name)
        for task_name, variant, source_name in (
            ("Task1", "full", ""), ("Task2", "full", ""), ("Task3", "dft", "fmt")
        )
    }
    for row in rows:
        row["delta_from_full"] = float(row["dataset_macro_f1"] - full[row["task"]])
        if row["task"] == "Task3":
            full_ap = next(
                value["dataset_macro_average_precision"] for value in rows
                if value["task"] == "Task3" and value["variant"] == "dft"
                and value["source"] == "fmt"
            )
            row["average_precision_delta_from_full"] = float(
                row["dataset_macro_average_precision"] - full_ap
            )
    _write_csv(output / "component_ablation_table.csv", rows)
    summary = {
        "schema": 1, "experiment": spec["experiment"],
        "record_counts": observed, "full_reference": full, "rows": rows,
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return target


def cleanup(config_path: str | Path) -> Path:
    spec = _spec(config_path)
    summary = Path(spec["output_root"]) / "summary.json"
    if not summary.exists():
        raise FileNotFoundError("refusing cleanup before complete summary")
    audit_path = Path(spec["output_root"]) / "independent_audit.json"
    if not audit_path.exists() or json.loads(
        audit_path.read_text(encoding="utf-8")
    ).get("status") != "PASS":
        raise RuntimeError("refusing cleanup before independent audit PASS")
    root = (Path(spec["output_root"]) / "task3" / "temporary_training").resolve()
    expected_parent = (Path(spec["output_root"]) / "task3").resolve()
    root.relative_to(expected_parent)
    count = len(list(root.rglob("*.pt"))) if root.exists() else 0
    if root.exists():
        shutil.rmtree(root)
    target = Path(spec["output_root"]) / "cleanup.json"
    target.write_text(json.dumps({
        "status": "PASS", "deleted_checkpoint_count": count,
        "temporary_training_exists": root.exists(),
        "independent_audit_sha256": hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Ablation_Task123_FMTComponents_1.1.yaml")
    parser.add_argument(
        "--mode", required=True,
        choices=("task1", "task1-merge", "task2", "task3", "summarize", "cleanup"),
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "task1":
        if args.job_index is None:
            parser.error("task1 requires --job-index")
        run_task1(args.config, args.job_index)
    elif args.mode == "task1-merge":
        merge_task1(args.config)
    elif args.mode == "task2":
        if args.job_index is None:
            parser.error("task2 requires --job-index")
        run_task2(args.config, args.job_index)
    elif args.mode == "task3":
        if args.job_index is None:
            parser.error("task3 requires --job-index")
        run_task3(args.config, args.job_index)
    elif args.mode == "summarize":
        summarize(args.config)
    else:
        cleanup(args.config)


if __name__ == "__main__":
    main()
