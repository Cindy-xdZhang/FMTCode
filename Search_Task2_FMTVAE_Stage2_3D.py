"""Stage-2 network search and frozen outer check for 3D Task2.

Stage 2 is gated by the immutable stage-1 selection JSON.  It expands only
three label-free FMT recipes per physical family.  Development ordinals 8--9
remain unopened until ``--mode outer`` is called after stage-2 selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from Run_Task2_3D_Main import _prepare_inputs
from Search_Task2_FMTVAE_3D import (
    _group_for_dataset,
    _latent_metrics,
    _load_spec,
    _read_csv,
    _result_path as _stage1_result_path,
    _split_records,
    _write_csv,
)
from Verify_HighReVAE import _train
from DeepUtils.utils import EasyConfig


def _selection_path(spec: dict, stage: int) -> Path:
    if stage == 1:
        return Path(spec["selection"]["stage1_selection_file"])
    return Path(spec["output_root"]) / "stage2_selection.json"


def _read_selection(spec: dict, stage: int) -> tuple[dict, str]:
    path = _selection_path(spec, stage)
    if not path.exists():
        raise FileNotFoundError(
            f"stage {stage} selection is required before this operation: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if bool(payload.get("confirmation_opened", False)):
        raise RuntimeError("selection payload unexpectedly used confirmation data")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_lookup(spec: dict) -> dict[str, dict]:
    return {str(row["id"]): dict(row) for row in spec["features"]}


def _selected_features(spec: dict) -> dict[str, list[dict]]:
    payload, _ = _read_selection(spec, 1)
    lookup = _feature_lookup(spec)
    count = int(spec["selection"].get("stage2_top_k", 3))
    selected = {}
    for group in spec["groups"]:
        rows = payload["top_k_by_group"][group]
        feature_ids = []
        for row in rows:
            feature_id = str(row["feature_id"])
            if feature_id not in feature_ids:
                feature_ids.append(feature_id)
            if len(feature_ids) == count:
                break
        if len(feature_ids) != count:
            raise RuntimeError(
                f"stage 1 retained {len(feature_ids)} unique features for {group}, "
                f"expected {count}"
            )
        selected[group] = [lookup[feature_id] for feature_id in feature_ids]
    return selected


def _load_development(spec: dict, dataset: str, include_outer=False):
    _, group = _group_for_dataset(spec, dataset)
    ordinals = (
        set(spec["splits"]["selection_train"])
        | set(spec["splits"]["selection_validation"])
    )
    if include_outer:
        ordinals |= set(spec["splits"]["outer"])
    records = load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(spec.get("expected_slices", 10)),
        ordinals=sorted(ordinals),
    )
    return {int(record["ordinal"]): record for record in records}, EasyConfig(
        group["source_config"]
    )


def _stage2_result_path(spec: dict, arm: str, dataset: str,
                        architecture: dict, feature: dict | None = None) -> Path:
    root = Path(spec["output_root"]) / "stage2" / arm / dataset
    if arm == "raw":
        return root / f"{architecture['id']}.csv"
    if feature is None:
        raise ValueError("FMT path requires a selected feature")
    return root / str(feature["id"]) / f"{architecture['id']}.csv"


def _reuse_stage1(spec: dict, arm: str, dataset: str, architecture: dict,
                  feature: dict | None, rows: list[dict]) -> list[dict]:
    stage1_arch = {row["id"]: row for row in spec["architectures"]}
    if architecture["id"] not in stage1_arch:
        return rows
    path = _stage1_result_path(
        spec, arm, dataset, stage1_arch[architecture["id"]], feature
    )
    if not path.exists():
        return rows
    existing = {int(row["training_seed"]) for row in rows}
    allowed = {int(seed) for seed in spec["stage2_screen_seeds"]}
    for source in _read_csv(path):
        seed = int(source["training_seed"])
        if seed not in allowed or seed in existing:
            continue
        copied = dict(source)
        copied.update({
            "experiment": spec["experiment"],
            "stage": "stage2",
            "reused_from_stage1": True,
        })
        rows.append(copied)
        existing.add(seed)
    return rows


def _run_arm(config_path: str, dataset: str, architecture_index: int,
             feature_rank: int | None = None) -> Path:
    spec = _load_spec(config_path)
    architectures = spec["stage2_architectures"]
    architecture = dict(architectures[int(architecture_index)])
    group_name, _ = _group_for_dataset(spec, dataset)
    feature = None
    if feature_rank is not None:
        feature = _selected_features(spec)[group_name][int(feature_rank)]
    arm = "raw" if feature is None else "fmt"
    path = _stage2_result_path(spec, arm, dataset, architecture, feature)
    rows = _read_csv(path)
    rows = _reuse_stage1(spec, arm, dataset, architecture, feature, rows)
    _write_csv(path, rows) if rows else None
    completed = {int(row["training_seed"]) for row in rows}
    records, source = _load_development(spec, dataset, include_outer=False)
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
    if architecture.get("pca_init", False) and int(
        architecture["latent_dim"]
    ) > train_x.shape[1]:
        raise ValueError(
            f"{architecture['id']} latent dimension exceeds {arm} input width"
        )
    for seed_value in spec["stage2_screen_seeds"]:
        seed = int(seed_value)
        if seed in completed:
            continue
        train_mu, validation_mu, losses = _train(
            train_x, validation_x, architecture, source, seed, device
        )
        metrics = _latent_metrics(train_mu, validation_mu, reference, spec)
        row = {
            "experiment": spec["experiment"],
            "stage": "stage2",
            "group": group_name,
            "dataset": dataset,
            "arm": arm,
            "feature_id": "" if feature is None else feature["id"],
            "fmt_feature": "" if feature is None else feature["name"],
            "architecture": architecture["id"],
            "training_seed": seed,
            "input_dim": int(train_x.shape[1]),
            "reused_from_stage1": False,
            **metrics,
            **losses,
        }
        rows.append(row)
        _write_csv(path, rows)
        completed.add(seed)
        print(
            f"stage2 {dataset}/{arm}/{feature_name}/{architecture['id']}/"
            f"seed={seed}: F1={metrics['f1']:.5f}", flush=True,
        )
    return path


def _decode_job(spec: dict, arm: str, job_index: int) -> tuple:
    datasets = list(spec["datasets"])
    architectures = list(spec["stage2_architectures"])
    index = int(job_index)
    if arm == "raw":
        count = len(datasets) * len(architectures)
        if not 0 <= index < count:
            raise IndexError(f"stage2 raw index {index} outside [0,{count})")
        dataset_index, architecture_index = divmod(index, len(architectures))
        return datasets[dataset_index], architecture_index, None
    top_k = int(spec["selection"].get("stage2_top_k", 3))
    per_dataset = top_k * len(architectures)
    count = len(datasets) * per_dataset
    if not 0 <= index < count:
        raise IndexError(f"stage2 FMT index {index} outside [0,{count})")
    dataset_index, remainder = divmod(index, per_dataset)
    feature_rank, architecture_index = divmod(remainder, len(architectures))
    return datasets[dataset_index], architecture_index, feature_rank


def run_job(config_path: str, arm: str, job_index: int) -> Path:
    spec = _load_spec(config_path)
    dataset, architecture_index, feature_rank = _decode_job(spec, arm, job_index)
    return _run_arm(config_path, dataset, architecture_index, feature_rank)


def _candidate_summary(spec: dict, group_name: str, feature: dict,
                       architecture: dict) -> dict:
    datasets = spec["groups"][group_name]["datasets"]
    seeds = [int(value) for value in spec["stage2_screen_seeds"]]
    per_dataset = {}
    seed_gains = {seed: [] for seed in seeds}
    for dataset in datasets:
        fmt_rows = _read_csv(_stage2_result_path(
            spec, "fmt", dataset, architecture, feature
        ))
        raw_rows = _read_csv(_stage2_result_path(
            spec, "raw", dataset, architecture
        ))
        fmt = {int(row["training_seed"]): float(row["f1"]) for row in fmt_rows}
        raw = {int(row["training_seed"]): float(row["f1"]) for row in raw_rows}
        if set(fmt) != set(seeds) or set(raw) != set(seeds):
            raise RuntimeError(
                f"incomplete stage2 Task2 {dataset}/{feature['id']}/"
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
    seed_macro = {
        str(seed): float(np.mean(values)) for seed, values in seed_gains.items()
    }
    guard = min(
        row["fmt_minus_task1_f1"] for row in per_dataset.values()
    ) >= -float(spec["selection"].get("allowed_fmt_below_task1", 0.02))
    return {
        "group": group_name,
        "feature_id": feature["id"],
        "fmt_feature": feature["name"],
        "architecture": architecture["id"],
        "raw_f1_macro": raw_macro,
        "fmt_f1_macro": fmt_macro,
        "fmt_minus_raw_f1_macro": fmt_macro - raw_macro,
        "worst_seed_f1_gain": min(seed_macro.values()),
        "all_seed_gains_positive": min(seed_macro.values()) > 0.0,
        "absolute_fmt_guard_passed": guard,
        "seed_gains_json": json.dumps(seed_macro, sort_keys=True),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
    }


def select(config_path: str) -> Path:
    spec = _load_spec(config_path)
    stage1, stage1_hash = _read_selection(spec, 1)
    selected_features = _selected_features(spec)
    leaderboard = []
    primary = {}
    for group_name in spec["groups"]:
        rows = [
            _candidate_summary(spec, group_name, feature, architecture)
            for feature in selected_features[group_name]
            for architecture in spec["stage2_architectures"]
        ]
        ranked = sorted(
            rows,
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
            leaderboard.append(row)
        primary[group_name] = ranked[0]
    output = Path(spec["output_root"])
    _write_csv(output / "stage2_leaderboard.csv", leaderboard)
    dataset_rows = []
    for group, row in primary.items():
        for dataset, metrics in json.loads(row["datasets_json"]).items():
            dataset_rows.append({"group": group, "dataset": dataset, **metrics})
    gain = float(np.mean([
        row["fmt_minus_raw_f1"] for row in dataset_rows
    ]))
    target_gain = float(spec["selection"].get(
        "target_dataset_macro_f1_gain", 0.15
    ))
    payload = {
        "experiment": spec["experiment"],
        "stage": 2,
        "stage1_selection_sha256": stage1_hash,
        "stage1_experiment": stage1["experiment"],
        "selection_rule": (
            "family-specific maximum paired same-VAE development F1 gain "
            "subject to the absolute Task1 FMT guard"
        ),
        "opened_ordinals": sorted(
            set(spec["splits"]["selection_train"])
            | set(spec["splits"]["selection_validation"])
        ),
        "outer_ordinals_opened": False,
        "confirmation_opened": False,
        "primary_by_group": primary,
        "development_dataset_macro_f1_gain": gain,
        "development_target_gain": target_gain,
        "development_target_reached": gain >= target_gain,
        "dataset_details": dataset_rows,
    }
    target = _selection_path(spec, 2)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def _outer_metrics(train_mu, validation_mu, outer_mu,
                   validation_reference, outer_reference, spec):
    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=2, random_state=int(spec["kmeans_seed"]),
        n_init=int(spec["kmeans_n_init"]),
    ).fit(train_mu)
    validation_labels = model.predict(validation_mu)
    vortex_cluster = calibrate_vortex_cluster(
        validation_reference, validation_labels
    )
    validation = binary_cluster_metrics(
        validation_reference, validation_labels, vortex_cluster
    )
    outer = binary_cluster_metrics(
        outer_reference, model.predict(outer_mu), vortex_cluster
    )
    return validation, outer, vortex_cluster


def evaluate_outer_dataset(config_path: str, dataset: str) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown outer dataset {dataset!r}")
    selection, selection_hash = _read_selection(spec, 2)
    output = Path(spec["output_root"]) / "outer_development"
    shard = output / "shards" / f"{dataset}.csv"
    existing = _read_csv(shard)
    expected = 2 * len(spec["stage2_screen_seeds"])
    if existing:
        hashes = {row["stage2_selection_sha256"] for row in existing}
        if hashes != {selection_hash} or len(existing) != expected:
            raise RuntimeError(f"stale or incomplete outer shard: {shard}")
        print(f"cached outer shard: {shard}")
        return shard
    feature_lookup = _feature_lookup(spec)
    architecture_lookup = {
        row["id"]: row for row in spec["stage2_architectures"]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    group_name, _ = _group_for_dataset(spec, dataset)
    chosen = selection["primary_by_group"][group_name]
    feature = feature_lookup[chosen["feature_id"]]
    architecture = architecture_lookup[chosen["architecture"]]
    records, source = _load_development(spec, dataset, include_outer=True)
    train_records = _split_records(records, spec["splits"]["selection_train"])
    validation_records = _split_records(
        records, spec["splits"]["selection_validation"]
    )
    outer_records = _split_records(records, spec["splits"]["outer"])
    evaluation_records = validation_records + outer_records
    validation_reference = stack_reference(validation_records)
    outer_reference = stack_reference(outer_records)
    validation_count = len(validation_reference)
    for arm in ("raw", "fmt"):
        train_x, evaluation_x = _prepare_inputs(
            train_records, evaluation_records, arm, feature["name"], device
        )
        for seed_value in spec["stage2_screen_seeds"]:
            seed = int(seed_value)
            train_mu, evaluation_mu, losses = _train(
                train_x, evaluation_x, architecture, source, seed, device
            )
            validation_mu = evaluation_mu[:validation_count]
            outer_mu = evaluation_mu[validation_count:]
            validation, metrics, vortex_cluster = _outer_metrics(
                train_mu, validation_mu, outer_mu,
                validation_reference, outer_reference, spec,
            )
            row = {
                "stage2_selection_sha256": selection_hash,
                "dataset": dataset, "group": group_name, "arm": arm,
                "feature_id": feature["id"], "fmt_feature": feature["name"],
                "architecture": architecture["id"], "training_seed": seed,
                "cluster_as_vortex": vortex_cluster,
                **{f"validation_{key}": value for key, value in validation.items()},
                **{f"outer_{key}": value for key, value in metrics.items()},
                **losses,
            }
            rows.append(row)
            print(
                f"outer {dataset}/{arm}/seed={seed}: "
                f"F1={metrics['f1']:.5f}", flush=True,
            )
    _write_csv(shard, rows)
    return shard


def summarize_outer(config_path: str) -> Path:
    spec = _load_spec(config_path)
    _, selection_hash = _read_selection(spec, 2)
    output = Path(spec["output_root"]) / "outer_development"
    marker = output / "audit.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["stage2_selection_sha256"] != selection_hash:
            raise RuntimeError("outer summary belongs to another stage2 selection")
        print(marker.read_text(encoding="utf-8"))
        return marker
    rows = []
    expected = 2 * len(spec["stage2_screen_seeds"])
    for dataset in spec["datasets"]:
        shard = output / "shards" / f"{dataset}.csv"
        values = _read_csv(shard)
        if len(values) != expected:
            raise RuntimeError(
                f"outer shard {dataset} has {len(values)} rows, expected {expected}"
            )
        if {row["stage2_selection_sha256"] for row in values} != {selection_hash}:
            raise RuntimeError(f"outer shard selection hash mismatch: {dataset}")
        rows.extend(values)
    _write_csv(output / "per_run.csv", rows)
    dataset_summary = {}
    for dataset in spec["datasets"]:
        selected_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_summary[dataset] = {}
        for arm in ("raw", "fmt"):
            values = [row for row in selected_rows if row["arm"] == arm]
            dataset_summary[dataset][arm] = {
                "f1": float(np.mean([float(row["outer_f1"]) for row in values]))
            }
        dataset_summary[dataset]["fmt_minus_raw_f1"] = (
            dataset_summary[dataset]["fmt"]["f1"]
            - dataset_summary[dataset]["raw"]["f1"]
        )
    macro_gain = float(np.mean([
        row["fmt_minus_raw_f1"] for row in dataset_summary.values()
    ]))
    audit = {
        "experiment": spec["experiment"],
        "stage2_selection_sha256": selection_hash,
        "opened_only_after_selection": True,
        "outer_ordinals": list(spec["splits"]["outer"]),
        "confirmation_opened": False,
        "dataset_summary": dataset_summary,
        "outer_dataset_macro_f1_gain": macro_gain,
        "target_gain": float(spec["selection"].get(
            "target_dataset_macro_f1_gain", 0.15
        )),
    }
    marker.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(marker.read_text(encoding="utf-8"))
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=(
        "raw", "fmt", "select", "outer", "outer-summary"
    ),
                        required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    if args.mode == "select":
        select(args.config)
    elif args.mode == "outer":
        if args.dataset is None:
            parser.error("outer mode requires --dataset")
        evaluate_outer_dataset(args.config, args.dataset)
    elif args.mode == "outer-summary":
        summarize_outer(args.config)
    elif args.job_index is None:
        parser.error("raw/fmt mode requires --job-index")
    else:
        run_job(args.config, args.mode, args.job_index)


if __name__ == "__main__":
    main()
