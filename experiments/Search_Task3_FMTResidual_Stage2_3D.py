"""Stage-2 residual-network search and frozen outer check for 3D Task3."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from Search_Task3_FMTResidual_3D import (
    _candidate_spec,
    _frozen_raw_normalization,
    _group_for_dataset,
    _load_records,
    _load_search_splits,
    _load_spec,
    _read_csv,
    _selection_key,
    _write_csv,
)
from Search_Task5_CylinderHyperparams import (
    _evaluate_baseline,
    _evaluate_residual,
)
from Verify_Task3_FMTClassifier import (
    _append_csv,
    _normalize_train_only,
    _stack_split,
)
from FMT_Utils.PathlineClassifier_3D import (
    PathlineFMTResidualClassifier3D, residual_model_kwargs,
)
from Verify_Task3_FMTResidual import _load_raw_model, _train_one


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
    return {str(row["id"]): dict(row) for row in spec["candidates"]}


def _selected_features(spec: dict) -> dict[str, list[dict]]:
    payload, _ = _read_selection(spec, 1)
    lookup = _feature_lookup(spec)
    count = int(spec["selection"].get("stage2_top_k", 3))
    result = {}
    for group in spec["groups"]:
        ids = [str(row["candidate_id"])
               for row in payload["top_k_by_group"][group]][:count]
        if len(ids) != count or len(set(ids)) != count:
            raise RuntimeError(
                f"stage 1 did not retain {count} unique Task3 features for {group}"
            )
        result[group] = [lookup[value] for value in ids]
    return result


def _combined_candidate(feature: dict, network: dict) -> dict:
    candidate = {
        "id": f"{feature['id']}__{network['id']}",
        "feature_candidate_id": feature["id"],
        "network_id": network["id"],
        "fmt_feature": feature["fmt_feature"],
    }
    candidate.update({key: value for key, value in network.items() if key != "id"})
    return candidate


def _stage2_candidates(spec: dict, group: str) -> list[dict]:
    return [
        _combined_candidate(feature, network)
        for feature in _selected_features(spec)[group]
        for network in spec["stage2_networks"]
    ]


def _result_path(spec: dict, candidate: dict, dataset: str,
                 seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "stage2" / "candidates"
        / str(candidate["id"]) / dataset / f"seed{int(seed)}"
        / source / "per_run.csv"
    )


def _parameter_budget_status(spec: dict, group: dict, candidate: dict,
                             dataset: str, fmt_dim: int) -> dict:
    """Preflight a stage-2 architecture against the frozen Raw-wide cap.

    Feature concatenations change ``fmt_dim`` and therefore the residual
    parameter count.  A Cartesian network grid can consequently contain
    structurally inadmissible feature/network pairs even though both grid
    axes are valid in isolation.  Check every paired seed before training so
    these pairs are recorded as ineligible rather than failing mid-array.
    """
    total_counts = []
    trainable_counts = []
    for seed_value in spec["stage2_screen_seeds"]:
        seed = int(seed_value)
        checkpoint = (
            Path(group["raw_checkpoint_dir"])
            / f"{dataset}_raw_seed{seed}.pt"
        )
        raw_model, _ = _load_raw_model(
            checkpoint, int(fmt_dim), torch.device("cpu")
        )
        model = PathlineFMTResidualClassifier3D(
            raw_model,
            fmt_dim=int(fmt_dim),
            **residual_model_kwargs(candidate),
        )
        total_counts.append(sum(
            parameter.numel() for parameter in model.parameters()
        ))
        trainable_counts.append(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ))
    if len(set(total_counts)) != 1 or len(set(trainable_counts)) != 1:
        raise RuntimeError(
            f"parameter counts differ across paired seeds for "
            f"{candidate['id']}/{dataset}: total={total_counts}, "
            f"trainable={trainable_counts}"
        )
    total = int(total_counts[0])
    trainable = int(trainable_counts[0])
    limit = int(spec["raw_wide_parameter_count"])
    return {
        "eligible": bool(total < limit),
        "total_parameter_count": total,
        "trainable_residual_parameter_count": trainable,
        "raw_wide_parameter_count": limit,
        "reason": (
            "" if total < limit else
            f"residual model has {total} parameters, not below Raw-wide {limit}"
        ),
    }


def _write_ineligible_results(spec: dict, candidate: dict, dataset: str,
                              fmt_dim: int, budget: dict) -> Path:
    """Write one auditable sentinel for every skipped paired run."""
    if bool(budget["eligible"]):
        raise ValueError("eligible candidates must not be written as ineligible")
    last_path = None
    for seed_value in spec["stage2_screen_seeds"]:
        seed = int(seed_value)
        for source in ("fmt", "raw_pca"):
            result_path = _result_path(spec, candidate, dataset, seed, source)
            existing = _read_csv(result_path)
            if len(existing) > 1:
                raise RuntimeError(f"duplicate stage2 result: {result_path}")
            expected = {
                "dataset": dataset,
                "variant": "invalid_parameter_budget",
                "seed": seed,
                "status": "invalid_parameter_budget",
                "invalid_reason": str(budget["reason"]),
                "parameter_count": int(budget["total_parameter_count"]),
                "trainable_residual_parameter_count": int(
                    budget["trainable_residual_parameter_count"]
                ),
                "raw_wide_parameter_count": int(
                    budget["raw_wide_parameter_count"]
                ),
                "auxiliary_source": source,
                "candidate_id": candidate["id"],
                "network_id": candidate["network_id"],
                "feature_candidate_id": candidate["feature_candidate_id"],
                "fmt_feature": candidate["fmt_feature"],
                "fmt_dim": int(fmt_dim),
                "reused_from_stage1": False,
            }
            if existing:
                observed = existing[0]
                for key, value in expected.items():
                    if str(observed.get(key, "")) != str(value):
                        raise RuntimeError(
                            f"ineligible sentinel changed for {result_path}: {key}"
                        )
            else:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                _append_csv(result_path, expected)
            last_path = result_path
    return last_path


def _stage1_result_path(spec: dict, candidate: dict, dataset: str,
                        seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "stage1" / "candidates"
        / str(candidate["feature_candidate_id"]) / dataset / f"seed{int(seed)}"
        / source / "per_run.csv"
    )


def _can_reuse_stage1(candidate: dict) -> bool:
    return str(candidate["network_id"]) == "n00_geom_aux64_min"


def _reuse_stage1_row(spec: dict, candidate: dict, dataset: str,
                      seed: int, source: str, result_path: Path) -> bool:
    if not _can_reuse_stage1(candidate) or seed not in {
        int(value) for value in spec["screen_seeds"]
    }:
        return False
    rows = _read_csv(_stage1_result_path(
        spec, candidate, dataset, seed, source
    ))
    if len(rows) != 1:
        return False
    copied = dict(rows[0])
    copied.update({
        "experiment": spec["experiment"],
        "candidate_id": candidate["id"],
        "network_id": candidate["network_id"],
        "feature_candidate_id": candidate["feature_candidate_id"],
        "reused_from_stage1": True,
    })
    _append_csv(result_path, copied)
    return True


def run_candidate(config_path: str, dataset: str,
                  candidate_index: int) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    group_name, group = _group_for_dataset(spec, dataset)
    candidates = _stage2_candidates(spec, group_name)
    index = int(candidate_index)
    if not 0 <= index < len(candidates):
        raise IndexError(
            f"stage2 candidate index {index} outside [0,{len(candidates)})"
        )
    candidate = candidates[index]
    device_name = str(spec["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    train, validation = _load_search_splits(
        spec, dataset, candidate, device
    )
    raw_stats = _frozen_raw_normalization(
        group, dataset, int(spec["stage2_screen_seeds"][0])
    )
    train, validation, _, stats = _normalize_train_only(
        train, validation, raw_stats=raw_stats
    )
    fmt_dim = int(train[1].shape[1])
    if fmt_dim > train[0].reshape(len(train[0]), -1).shape[1]:
        raise ValueError(f"Raw-PCA cannot match {fmt_dim} FMT dimensions")
    budget = _parameter_budget_status(
        spec, group, candidate, dataset, fmt_dim
    )
    if not budget["eligible"]:
        print(
            f"INELIGIBLE {candidate['id']} {dataset}: {budget['reason']}",
            flush=True,
        )
        return _write_ineligible_results(
            spec, candidate, dataset, fmt_dim, budget
        )
    last_path = None
    for seed_value in spec["stage2_screen_seeds"]:
        seed = int(seed_value)
        for source in ("fmt", "raw_pca"):
            result_path = _result_path(spec, candidate, dataset, seed, source)
            existing = _read_csv(result_path)
            if len(existing) > 1:
                raise RuntimeError(f"duplicate stage2 result: {result_path}")
            if existing:
                last_path = result_path
                continue
            result_path.parent.mkdir(parents=True, exist_ok=True)
            if _reuse_stage1_row(
                spec, candidate, dataset, seed, source, result_path
            ):
                print(
                    f"reused stage1 {candidate['id']} {dataset} seed={seed} {source}"
                )
                last_path = result_path
                continue
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, seed, source,
                result_path.parent, fmt_dim,
            )
            (result_path.parent / "config_snapshot.yaml").write_text(
                yaml.safe_dump(run_spec, sort_keys=False),
                encoding="utf-8",
            )
            row = _train_one(
                run_spec, dataset, seed, (train, validation, None), stats,
                device, result_path.parent,
            )
            row.update({
                "candidate_id": candidate["id"],
                "network_id": candidate["network_id"],
                "feature_candidate_id": candidate["feature_candidate_id"],
                "fmt_feature": candidate["fmt_feature"],
                "fmt_dim": fmt_dim,
                "reused_from_stage1": False,
            })
            _append_csv(result_path, row)
            last_path = result_path
            print(
                f"DONE {candidate['id']} {dataset} seed={seed} {source}: "
                f"F1={row['validation_f1']:.5f} "
                f"AP={row['validation_average_precision']:.5f}", flush=True,
            )
    return last_path


def _decode_job(spec: dict, job_index: int) -> tuple[str, int]:
    top_k = int(spec["selection"].get("stage2_top_k", 3))
    candidates_per_dataset = top_k * len(spec["stage2_networks"])
    count = len(spec["datasets"]) * candidates_per_dataset
    index = int(job_index)
    if not 0 <= index < count:
        raise IndexError(f"Task3 stage2 job index {index} outside [0,{count})")
    dataset_index, candidate_index = divmod(index, candidates_per_dataset)
    return spec["datasets"][dataset_index], candidate_index


def run_job(config_path: str, job_index: int) -> Path:
    spec = _load_spec(config_path)
    dataset, candidate_index = _decode_job(spec, job_index)
    return run_candidate(config_path, dataset, candidate_index)


def _baseline_from_row(row: dict) -> dict:
    value = row["validation_selection_baseline"]
    parsed = value if isinstance(value, dict) else ast.literal_eval(str(value))
    return {key: float(parsed[key]) for key in ("f1", "average_precision")}


def _strong_baseline(spec: dict, dataset: str, seed: int,
                     candidate_row: dict) -> dict:
    value = candidate_row.get("validation_selection_baseline", "")
    if str(value).strip() not in {"", "None"}:
        return _baseline_from_row(candidate_row)
    _, group = _group_for_dataset(spec, dataset)
    baseline_path = Path(group["raw_checkpoint_dir"]).parent / "per_run.csv"
    rows = [
        row for row in _read_csv(baseline_path)
        if row["dataset"] == dataset and int(row["seed"]) == int(seed)
        and row["variant"] in {"raw", "raw_wide"}
    ]
    if {row["variant"] for row in rows} != {"raw", "raw_wide"}:
        raise RuntimeError(
            f"missing Raw/Raw-wide validation baselines for {dataset} seed={seed}"
        )
    return {
        metric: max(float(row[f"validation_{metric}"]) for row in rows)
        for metric in ("f1", "average_precision")
    }


def _candidate_summary(spec: dict, group_name: str, candidate: dict) -> dict:
    datasets = spec["groups"][group_name]["datasets"]
    seeds = [int(value) for value in spec["stage2_screen_seeds"]]
    metrics = ("f1", "average_precision")
    result_rows = {}
    ineligible = []
    for dataset in datasets:
        for seed in seeds:
            for source in ("fmt", "raw_pca"):
                values = _read_csv(_result_path(
                    spec, candidate, dataset, seed, source
                ))
                if len(values) != 1:
                    raise RuntimeError(
                        f"incomplete Task3 stage2 {candidate['id']}/{dataset}/"
                        f"seed={seed}/{source}"
                    )
                row = values[0]
                status = str(row.get("status", ""))
                if status and status != "invalid_parameter_budget":
                    raise RuntimeError(
                        f"unknown stage2 status {status!r} for "
                        f"{candidate['id']}/{dataset}/seed={seed}/{source}"
                    )
                if status == "invalid_parameter_budget":
                    ineligible.append({
                        "dataset": dataset,
                        "seed": seed,
                        "source": source,
                        "reason": str(row["invalid_reason"]),
                        "parameter_count": int(row["parameter_count"]),
                        "raw_wide_parameter_count": int(
                            row["raw_wide_parameter_count"]
                        ),
                    })
                result_rows[(dataset, seed, source)] = row
    if ineligible:
        affected = sorted({row["dataset"] for row in ineligible})
        for dataset in affected:
            observed = [
                row for row in ineligible if row["dataset"] == dataset
            ]
            expected = len(seeds) * 2
            if len(observed) != expected:
                raise RuntimeError(
                    f"partial parameter-budget sentinels for "
                    f"{candidate['id']}/{dataset}: {len(observed)}/{expected}"
                )
        reasons = sorted({row["reason"] for row in ineligible})
        return {
            "group": group_name,
            "candidate_id": candidate["id"],
            "feature_candidate_id": candidate["feature_candidate_id"],
            "network_id": candidate["network_id"],
            "fmt_feature": candidate["fmt_feature"],
            "eligible": False,
            "ineligible_datasets_json": json.dumps(affected),
            "ineligible_reasons_json": json.dumps(reasons),
        }
    per_dataset = {}
    seed_gains = {seed: [] for seed in seeds}
    for dataset in datasets:
        per_seed = {}
        for seed in seeds:
            rows = {
                source: result_rows[(dataset, seed, source)]
                for source in ("fmt", "raw_pca")
            }
            if {
                int(rows[source]["trainable_residual_parameter_count"])
                for source in ("fmt", "raw_pca")
            } != {int(rows["fmt"]["trainable_residual_parameter_count"])}:
                raise RuntimeError(
                    f"FMT/Raw-PCA trainable parameter mismatch for "
                    f"{candidate['id']}/{dataset}/seed={seed}"
                )
            fmt = {metric: float(rows["fmt"][f"validation_{metric}"])
                   for metric in metrics}
            raw_pca = {
                metric: float(rows["raw_pca"][f"validation_{metric}"])
                for metric in metrics
            }
            strong = _strong_baseline(spec, dataset, seed, rows["fmt"])
            per_seed[seed] = {"fmt": fmt, "raw_pca": raw_pca, "strong_raw": strong}
            seed_gains[seed].append(fmt["f1"] - raw_pca["f1"])
        per_dataset[dataset] = {}
        for source in ("fmt", "raw_pca", "strong_raw"):
            per_dataset[dataset][source] = {
                metric: float(np.mean([
                    per_seed[seed][source][metric] for seed in seeds
                ])) for metric in metrics
            }
        per_dataset[dataset]["gains"] = {
            f"{metric}_vs_raw_pca": (
                per_dataset[dataset]["fmt"][metric]
                - per_dataset[dataset]["raw_pca"][metric]
            ) for metric in metrics
        }
        per_dataset[dataset]["gains"].update({
            f"{metric}_vs_strong_raw": (
                per_dataset[dataset]["fmt"][metric]
                - per_dataset[dataset]["strong_raw"][metric]
            ) for metric in metrics
        })
    macro = {}
    for source in ("fmt", "raw_pca", "strong_raw"):
        macro[source] = {
            metric: float(np.mean([
                row[source][metric] for row in per_dataset.values()
            ])) for metric in metrics
        }
    gains = {
        f"{metric}_vs_raw_pca": macro["fmt"][metric] - macro["raw_pca"][metric]
        for metric in metrics
    }
    gains.update({
        f"{metric}_vs_strong_raw": macro["fmt"][metric] - macro["strong_raw"][metric]
        for metric in metrics
    })
    tolerance = float(spec["selection"].get(
        "allowed_fmt_below_strong_raw", 0.005
    ))
    guard = min(
        gains["f1_vs_strong_raw"], gains["average_precision_vs_strong_raw"]
    ) >= -tolerance
    seed_macro = {
        str(seed): float(np.mean(values)) for seed, values in seed_gains.items()
    }
    return {
        "group": group_name,
        "candidate_id": candidate["id"],
        "feature_candidate_id": candidate["feature_candidate_id"],
        "network_id": candidate["network_id"],
        "fmt_feature": candidate["fmt_feature"],
        "eligible": True,
        "ineligible_datasets_json": "[]",
        "ineligible_reasons_json": "[]",
        "fmt_f1_macro": macro["fmt"]["f1"],
        "raw_pca_f1_macro": macro["raw_pca"]["f1"],
        "strong_raw_f1_macro": macro["strong_raw"]["f1"],
        "fmt_minus_raw_pca_f1_macro": gains["f1_vs_raw_pca"],
        "fmt_minus_strong_raw_f1_macro": gains["f1_vs_strong_raw"],
        "fmt_ap_macro": macro["fmt"]["average_precision"],
        "raw_pca_ap_macro": macro["raw_pca"]["average_precision"],
        "strong_raw_ap_macro": macro["strong_raw"]["average_precision"],
        "fmt_minus_raw_pca_ap_macro": gains["average_precision_vs_raw_pca"],
        "fmt_minus_strong_raw_ap_macro": gains["average_precision_vs_strong_raw"],
        "worst_seed_f1_gain": min(seed_macro.values()),
        "strong_raw_guard_passed": guard,
        "seed_gains_json": json.dumps(seed_macro, sort_keys=True),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
    }


def select(config_path: str) -> Path:
    spec = _load_spec(config_path)
    stage1, stage1_hash = _read_selection(spec, 1)
    primary = {}
    leaderboard = []
    for group_name in spec["groups"]:
        rows = [
            _candidate_summary(spec, group_name, candidate)
            for candidate in _stage2_candidates(spec, group_name)
        ]
        eligible = [row for row in rows if bool(row["eligible"])]
        if not eligible:
            raise RuntimeError(
                f"all Task3 stage2 candidates are ineligible for {group_name}"
            )
        ranked = sorted(
            eligible,
            key=_selection_key,
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["rank_within_group"] = rank
            leaderboard.append(row)
        for row in rows:
            if not bool(row["eligible"]):
                row["rank_within_group"] = ""
                leaderboard.append(row)
        primary[group_name] = ranked[0]
    output = Path(spec["output_root"])
    _write_csv(output / "stage2_leaderboard.csv", leaderboard)
    dataset_rows = []
    for group, row in primary.items():
        for dataset, metrics in json.loads(row["datasets_json"]).items():
            dataset_rows.append({"group": group, "dataset": dataset, **metrics})
    f1_gain = float(np.mean([
        row["gains"]["f1_vs_raw_pca"] for row in dataset_rows
    ]))
    ap_gain = float(np.mean([
        row["gains"]["average_precision_vs_raw_pca"] for row in dataset_rows
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
            "family-specific: maximize same-width Raw-PCA F1 gain; tie-break "
            "by AP and worst seed. Strong Raw remains a reported diagnostic"
        ),
        "opened_ordinals": sorted(
            set(spec["screen_split"]["train_ordinals"])
            | set(spec["screen_split"]["validation_ordinals"])
        ),
        "exposed_spatial_training": spec.get("exposed_training"),
        "exposed_spatial_validation": spec.get("robust_validation"),
        "outer_ordinals_opened": False,
        "confirmation_opened": False,
        "primary_by_group": primary,
        "development_dataset_macro_f1_gain_vs_raw_pca": f1_gain,
        "development_dataset_macro_ap_gain_vs_raw_pca": ap_gain,
        "development_target_gain": target_gain,
        "development_target_reached": f1_gain >= target_gain,
        "dataset_details": dataset_rows,
    }
    target = _selection_path(spec, 2)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def _selected_candidate(spec: dict, group_name: str, row: dict) -> dict:
    candidates = {
        candidate["id"]: candidate
        for candidate in _stage2_candidates(spec, group_name)
    }
    return candidates[str(row["candidate_id"])]


def _checkpoint(spec: dict, candidate: dict, dataset: str,
                seed: int, source: str) -> Path:
    rows = _read_csv(_result_path(spec, candidate, dataset, seed, source))
    if len(rows) != 1:
        raise RuntimeError("selected checkpoint row is missing or duplicated")
    path = Path(rows[0]["checkpoint"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def evaluate_outer(config_path: str) -> Path:
    spec = _load_spec(config_path)
    selection, selection_hash = _read_selection(spec, 2)
    output = Path(spec["output_root"]) / "outer_development"
    marker = output / "audit.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["stage2_selection_sha256"] != selection_hash:
            raise RuntimeError("outer development was opened for another selection")
        print(marker.read_text(encoding="utf-8"))
        return marker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in spec["datasets"]:
        group_name, group = _group_for_dataset(spec, dataset)
        candidate = _selected_candidate(
            spec, group_name, selection["primary_by_group"][group_name]
        )
        records = _load_records(
            spec, dataset, candidate, device, ordinals=spec["outer_ordinals"]
        )
        split = _stack_split(records, spec["outer_ordinals"])
        for seed_value in spec["stage2_screen_seeds"]:
            seed = int(seed_value)
            baseline_dir = Path(group["raw_checkpoint_dir"])
            for variant in ("raw", "raw_wide"):
                metrics = _evaluate_baseline(
                    baseline_dir / f"{dataset}_{variant}_seed{seed}.pt",
                    split, spec["training"]["batch_size"], device,
                )
                rows.append({
                    "dataset": dataset, "group": group_name, "seed": seed,
                    "method": variant, "candidate_id": candidate["id"], **metrics,
                })
            for source, method in (
                ("raw_pca", "raw_pca_residual"), ("fmt", "fmt_residual")
            ):
                metrics = _evaluate_residual(
                    _checkpoint(spec, candidate, dataset, seed, source),
                    split, spec["training"]["batch_size"], device,
                )
                rows.append({
                    "dataset": dataset, "group": group_name, "seed": seed,
                    "method": method, "candidate_id": candidate["id"], **metrics,
                })
            print(f"outer Task3 {dataset} seed={seed} complete", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_run.csv", rows)
    summary = {}
    for dataset in spec["datasets"]:
        summary[dataset] = {}
        selected_rows = [row for row in rows if row["dataset"] == dataset]
        for method in ("raw", "raw_wide", "raw_pca_residual", "fmt_residual"):
            values = [row for row in selected_rows if row["method"] == method]
            summary[dataset][method] = {
                metric: float(np.mean([row[metric] for row in values]))
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
    audit = {
        "experiment": spec["experiment"],
        "stage2_selection_sha256": selection_hash,
        "opened_only_after_selection": True,
        "outer_ordinals": list(spec["outer_ordinals"]),
        "historically_exposed_in_prior_tasks": True,
        "confirmation_opened": False,
        "dataset_summary": summary,
        "outer_dataset_macro_f1_gain_vs_raw_pca": macro_f1,
        "outer_dataset_macro_ap_gain_vs_raw_pca": macro_ap,
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
    parser.add_argument("--mode", choices=("candidate", "select", "outer"),
                        required=True)
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "select":
        select(args.config)
    elif args.mode == "outer":
        evaluate_outer(args.config)
    elif args.job_index is None:
        parser.error("candidate mode requires --job-index")
    else:
        run_job(args.config, args.job_index)


if __name__ == "__main__":
    main()
