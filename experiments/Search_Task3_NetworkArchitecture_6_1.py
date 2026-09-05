"""Architecture-only Task3 search with paired Raw-PCA and FMT inputs.

The family-specific FMT feature recipes and all data populations are frozen.
Each array child changes one trainable network architecture and trains exactly
the same architecture twice per seed: once with train-only Raw-PCA features
and once with FMT features of the same width.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.PathlineClassifier_3D import (
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Search_Task3_FMTResidual_3D import (
    _candidate_spec,
    _frozen_raw_normalization,
    _group_for_dataset,
    _load_search_splits,
    _load_spec,
    _read_csv,
    _write_csv,
)
from Verify_Task3_FMTClassifier import _append_csv, _normalize_train_only
from Verify_Task3_FMTResidual import _load_raw_model, _train_one


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_architecture_spec(path: str | Path) -> dict:
    path = Path(path)
    overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "base_search_config",
        "base_search_config_sha256", "feature_selection",
        "feature_selection_sha256", "paired_seeds", "common_model",
        "common_fusion", "architectures", "selection",
    }
    missing = sorted(required.difference(overlay))
    if missing:
        raise ValueError(f"missing architecture config keys: {missing}")

    base_path = Path(overlay["base_search_config"])
    if _sha256(base_path) != str(overlay["base_search_config_sha256"]).lower():
        raise RuntimeError("base Task3 development config changed")
    base = _load_spec(base_path)
    selection_path = Path(overlay["feature_selection"])
    if _sha256(selection_path) != str(
        overlay["feature_selection_sha256"]
    ).lower():
        raise RuntimeError("frozen Task3 FMT feature selection changed")
    feature_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if bool(feature_selection.get("confirmation_opened", False)):
        raise RuntimeError("feature selection unexpectedly opened confirmation")

    architecture_ids = [str(row["id"]) for row in overlay["architectures"]]
    if not architecture_ids or len(architecture_ids) != len(set(architecture_ids)):
        raise ValueError("architecture ids must be non-empty and unique")
    paired_seeds = [int(value) for value in overlay["paired_seeds"]]
    if not paired_seeds or len(paired_seeds) != len(set(paired_seeds)):
        raise ValueError("paired seeds must be non-empty and unique")
    available_seeds = {int(value) for value in base["stage2_screen_seeds"]}
    if not set(paired_seeds).issubset(available_seeds):
        raise ValueError("paired seeds require unavailable frozen Raw checkpoints")
    if bool(overlay["selection"].get("confirmation_opened", False)):
        raise RuntimeError("architecture config must not open confirmation")

    spec = dict(base)
    spec.update({
        "experiment": str(overlay["experiment"]),
        "output_root": str(overlay["output_root"]),
        "architecture_config": str(path),
        "architecture_config_sha256": _sha256(path),
        "base_search_config": str(base_path),
        "base_search_config_sha256": _sha256(base_path),
        "feature_selection": str(selection_path),
        "feature_selection_sha256": _sha256(selection_path),
        "feature_selection_payload": feature_selection,
        "paired_seeds": paired_seeds,
        "stage2_screen_seeds": paired_seeds,
        "common_model": dict(overlay["common_model"]),
        "common_fusion": dict(overlay["common_fusion"]),
        "architectures": [dict(row) for row in overlay["architectures"]],
        "architecture_selection": dict(overlay["selection"]),
    })
    return spec


def _feature_by_group(spec: dict) -> dict[str, dict]:
    primary = spec["feature_selection_payload"]["primary_by_group"]
    if set(primary) != set(spec["groups"]):
        raise RuntimeError("frozen feature selection groups changed")
    return {
        group: {
            "feature_candidate_id": str(row["feature_candidate_id"]),
            "fmt_feature": str(row["fmt_feature"]),
        }
        for group, row in primary.items()
    }


def _architecture_candidate(spec: dict, dataset: str,
                            architecture_index: int) -> dict:
    group_name, _ = _group_for_dataset(spec, dataset)
    architecture_index = int(architecture_index)
    if not 0 <= architecture_index < len(spec["architectures"]):
        raise IndexError("architecture index outside configured grid")
    architecture = dict(spec["architectures"][architecture_index])
    candidate = {
        **spec["common_model"],
        **spec["common_fusion"],
        **architecture,
        **_feature_by_group(spec)[group_name],
        "network_id": architecture["id"],
        "architecture_id": architecture["id"],
    }
    candidate["id"] = str(architecture["id"])
    return candidate


def _result_path(spec: dict, candidate: dict, dataset: str,
                 seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "architectures" / candidate["id"]
        / dataset / f"seed{int(seed)}" / source / "per_run.csv"
    )


def _parameter_budget(spec: dict, group: dict, candidate: dict,
                      dataset: str, fmt_dim: int) -> dict:
    totals, trainables = [], []
    for seed_value in spec["paired_seeds"]:
        raw_model, _ = _load_raw_model(
            Path(group["raw_checkpoint_dir"])
            / f"{dataset}_raw_seed{int(seed_value)}.pt",
            int(fmt_dim), torch.device("cpu"),
        )
        model = PathlineFMTResidualClassifier3D(
            raw_model, fmt_dim=int(fmt_dim),
            **residual_model_kwargs(candidate),
        )
        totals.append(sum(parameter.numel() for parameter in model.parameters()))
        trainables.append(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ))
    if len(set(totals)) != 1 or len(set(trainables)) != 1:
        raise RuntimeError("paired seeds changed architecture parameter counts")
    limit = int(spec["raw_wide_parameter_count"])
    return {
        "eligible": int(totals[0]) < limit,
        "total_parameter_count": int(totals[0]),
        "trainable_residual_parameter_count": int(trainables[0]),
        "raw_wide_parameter_count": limit,
    }


def _write_ineligible(spec: dict, candidate: dict, dataset: str,
                      fmt_dim: int, budget: dict) -> Path:
    last = None
    for seed in spec["paired_seeds"]:
        for source in ("fmt", "raw_pca"):
            path = _result_path(spec, candidate, dataset, seed, source)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                _append_csv(path, {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": "invalid_parameter_budget",
                    "status": "invalid_parameter_budget",
                    "auxiliary_source": source,
                    "architecture_id": candidate["architecture_id"],
                    "fmt_feature": candidate["fmt_feature"],
                    "fmt_dim": fmt_dim,
                    **budget,
                })
            last = path
    return last


def run_candidate(config_path: str, dataset: str,
                  architecture_index: int) -> Path:
    spec = _load_architecture_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    candidate = _architecture_candidate(spec, dataset, architecture_index)
    _, group = _group_for_dataset(spec, dataset)
    device_name = str(spec["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    train, validation = _load_search_splits(spec, dataset, candidate, device)
    frozen_stats = _frozen_raw_normalization(
        group, dataset, spec["paired_seeds"][0]
    )
    train, validation, _, stats = _normalize_train_only(
        train, validation, raw_stats=frozen_stats
    )
    fmt_dim = int(train[1].shape[1])
    if fmt_dim > train[0].reshape(len(train[0]), -1).shape[1]:
        raise ValueError("Raw-PCA cannot match the frozen FMT feature width")
    budget = _parameter_budget(spec, group, candidate, dataset, fmt_dim)
    if not budget["eligible"]:
        return _write_ineligible(spec, candidate, dataset, fmt_dim, budget)

    last = None
    for seed in spec["paired_seeds"]:
        for source in ("fmt", "raw_pca"):
            result_path = _result_path(spec, candidate, dataset, seed, source)
            rows = _read_csv(result_path)
            if len(rows) > 1:
                raise RuntimeError(f"duplicate architecture result: {result_path}")
            if rows:
                expected_hashes = {
                    "architecture_config_sha256": spec[
                        "architecture_config_sha256"
                    ],
                    "base_search_config_sha256": spec[
                        "base_search_config_sha256"
                    ],
                    "feature_selection_sha256": spec[
                        "feature_selection_sha256"
                    ],
                }
                for key, expected in expected_hashes.items():
                    if str(rows[0].get(key, "")).lower() != str(expected).lower():
                        raise RuntimeError(
                            f"stale cached architecture result {result_path}: "
                            f"{key} changed"
                        )
                print(f"cached {candidate['id']} {dataset} seed={seed} {source}")
                last = result_path
                continue
            result_path.parent.mkdir(parents=True, exist_ok=True)
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, seed, source,
                result_path.parent, fmt_dim,
            )
            (result_path.parent / "config_snapshot.yaml").write_text(
                yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8"
            )
            row = _train_one(
                run_spec, dataset, seed, (train, validation, None), stats,
                device, result_path.parent,
            )
            row.update({
                "architecture_id": candidate["architecture_id"],
                "head_architecture": candidate["head_architecture"],
                "fmt_feature": candidate["fmt_feature"],
                "feature_candidate_id": candidate["feature_candidate_id"],
                "fmt_dim": fmt_dim,
                "architecture_config_sha256": spec[
                    "architecture_config_sha256"
                ],
                "base_search_config_sha256": spec[
                    "base_search_config_sha256"
                ],
                "feature_selection_sha256": spec[
                    "feature_selection_sha256"
                ],
            })
            _append_csv(result_path, row)
            last = result_path
            print(
                f"DONE {candidate['id']} {dataset} seed={seed} {source}: "
                f"F1={row['validation_f1']:.5f} "
                f"AP={row['validation_average_precision']:.5f}",
                flush=True,
            )
    return last


def _decode_job(spec: dict, job_index: int) -> tuple[str, int]:
    per_dataset = len(spec["architectures"])
    total = len(spec["datasets"]) * per_dataset
    job_index = int(job_index)
    if not 0 <= job_index < total:
        raise IndexError(f"architecture job index outside [0,{total})")
    dataset_index, architecture_index = divmod(job_index, per_dataset)
    return spec["datasets"][dataset_index], architecture_index


def run_job(config_path: str, job_index: int) -> Path:
    spec = _load_architecture_spec(config_path)
    dataset, architecture_index = _decode_job(spec, job_index)
    return run_candidate(config_path, dataset, architecture_index)


def preflight(config_path: str) -> Path:
    """Validate every real data contract and parameter budget without training."""
    spec = _load_architecture_spec(config_path)
    datasets = []
    for dataset in spec["datasets"]:
        candidate = _architecture_candidate(spec, dataset, 0)
        _, group = _group_for_dataset(spec, dataset)
        train, validation = _load_search_splits(
            spec, dataset, candidate, torch.device("cpu")
        )
        fmt_dim = int(train[1].shape[1])
        raw_dim = int(train[0].reshape(len(train[0]), -1).shape[1])
        if fmt_dim > raw_dim:
            raise RuntimeError(
                f"{dataset}: FMT width {fmt_dim} exceeds Raw width {raw_dim}"
            )
        if not (0 < int(train[2].sum()) < len(train[2])):
            raise RuntimeError(f"{dataset}: training labels are single-class")
        if not (0 < int(validation[2].sum()) < len(validation[2])):
            raise RuntimeError(f"{dataset}: validation labels are single-class")
        architectures = []
        for index in range(len(spec["architectures"])):
            architecture = _architecture_candidate(spec, dataset, index)
            budget = _parameter_budget(
                spec, group, architecture, dataset, fmt_dim
            )
            if not budget["eligible"]:
                raise RuntimeError(
                    f"{dataset}/{architecture['id']} exceeds Raw-wide capacity"
                )
            architectures.append({
                "architecture_id": architecture["id"],
                **budget,
            })
        datasets.append({
            "dataset": dataset,
            "feature_candidate_id": candidate["feature_candidate_id"],
            "fmt_feature": candidate["fmt_feature"],
            "fmt_dim": fmt_dim,
            "raw_dim": raw_dim,
            "training_samples": int(len(train[2])),
            "training_positive_count": int(train[2].sum()),
            "validation_samples": int(len(validation[2])),
            "validation_positive_count": int(validation[2].sum()),
            "architectures": architectures,
        })
    payload = {
        "experiment": spec["experiment"],
        "architecture_config_sha256": spec["architecture_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "feature_selection_sha256": spec["feature_selection_sha256"],
        "confirmation_opened": False,
        "dataset_count": len(spec["datasets"]),
        "architecture_count": len(spec["architectures"]),
        "paired_seed_count": len(spec["paired_seeds"]),
        "paired_arm_count": 2,
        "expected_training_runs": (
            len(spec["datasets"]) * len(spec["architectures"])
            * len(spec["paired_seeds"]) * 2
        ),
        "datasets": datasets,
    }
    target = Path(spec["output_root"]) / "preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(target)
    return target


def _architecture_summary(spec: dict, architecture: dict) -> dict:
    architecture_id = str(architecture["id"])
    per_dataset, parameter_counts = {}, set()
    seed_gains = {int(seed): [] for seed in spec["paired_seeds"]}
    for dataset in spec["datasets"]:
        candidate = _architecture_candidate(
            spec, dataset,
            next(index for index, row in enumerate(spec["architectures"])
                 if str(row["id"]) == architecture_id),
        )
        per_seed = {}
        for seed in spec["paired_seeds"]:
            rows = {}
            for source in ("fmt", "raw_pca"):
                values = _read_csv(_result_path(
                    spec, candidate, dataset, seed, source
                ))
                if len(values) != 1:
                    raise RuntimeError(
                        f"incomplete architecture result {architecture_id}/"
                        f"{dataset}/seed={seed}/{source}"
                    )
                if values[0].get("status") == "invalid_parameter_budget":
                    raise RuntimeError(
                        f"architecture {architecture_id} exceeds parameter cap"
                    )
                rows[source] = values[0]
            paired_counts = {
                int(rows[source]["trainable_residual_parameter_count"])
                for source in rows
            }
            if len(paired_counts) != 1:
                raise RuntimeError("FMT and Raw-PCA parameter counts differ")
            parameter_counts.add(int(rows["fmt"]["parameter_count"]))
            per_seed[int(seed)] = {
                source: {
                    "f1": float(rows[source]["validation_f1"]),
                    "average_precision": float(
                        rows[source]["validation_average_precision"]
                    ),
                } for source in rows
            }
            seed_gains[int(seed)].append(
                per_seed[int(seed)]["fmt"]["f1"]
                - per_seed[int(seed)]["raw_pca"]["f1"]
            )
        means = {
            source: {
                metric: float(np.mean([
                    per_seed[int(seed)][source][metric]
                    for seed in spec["paired_seeds"]
                ]))
                for metric in ("f1", "average_precision")
            }
            for source in ("fmt", "raw_pca")
        }
        per_dataset[dataset] = {
            **means,
            "f1_gain": means["fmt"]["f1"] - means["raw_pca"]["f1"],
            "average_precision_gain": (
                means["fmt"]["average_precision"]
                - means["raw_pca"]["average_precision"]
            ),
        }

    f1_gains = [row["f1_gain"] for row in per_dataset.values()]
    ap_gains = [row["average_precision_gain"] for row in per_dataset.values()]
    family_details = {}
    for group_name, group in spec["groups"].items():
        family_details[group_name] = {
            "f1_gain": float(np.mean([
                per_dataset[dataset]["f1_gain"] for dataset in group["datasets"]
            ])),
            "average_precision_gain": float(np.mean([
                per_dataset[dataset]["average_precision_gain"]
                for dataset in group["datasets"]
            ])),
        }
    return {
        "architecture_id": architecture_id,
        "head_architecture": str(architecture["head_architecture"]),
        "architecture_json": json.dumps(architecture, sort_keys=True),
        "dataset_macro_fmt_f1": float(np.mean([
            row["fmt"]["f1"] for row in per_dataset.values()
        ])),
        "dataset_macro_raw_pca_f1": float(np.mean([
            row["raw_pca"]["f1"] for row in per_dataset.values()
        ])),
        "dataset_macro_f1_gain_vs_raw_pca": float(np.mean(f1_gains)),
        "dataset_macro_average_precision_gain_vs_raw_pca": float(
            np.mean(ap_gains)
        ),
        "positive_dataset_count": int(np.count_nonzero(np.asarray(f1_gains) > 0)),
        "positive_family_count": int(np.count_nonzero([
            row["f1_gain"] > 0 for row in family_details.values()
        ])),
        "worst_dataset_f1_gain": float(min(f1_gains)),
        "worst_seed_f1_gain": float(min(
            np.mean(values) for values in seed_gains.values()
        )),
        "minimum_total_parameter_count": min(parameter_counts),
        "maximum_total_parameter_count": max(parameter_counts),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
        "families_json": json.dumps(family_details, sort_keys=True),
        "seed_gains_json": json.dumps({
            str(seed): float(np.mean(values))
            for seed, values in seed_gains.items()
        }, sort_keys=True),
    }


def select(config_path: str) -> Path:
    spec = _load_architecture_spec(config_path)
    rows = [
        _architecture_summary(spec, architecture)
        for architecture in spec["architectures"]
    ]
    order = tuple(spec["architecture_selection"]["tie_breakers"])
    required = (
        str(spec["architecture_selection"]["primary_metric"]), *order
    )
    for key in required:
        if any(key not in row for row in rows):
            raise KeyError(f"unknown architecture selection key {key!r}")
    ranked = sorted(
        rows,
        key=lambda row: tuple(float(row[key]) for key in required),
        reverse=True,
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    output_root = Path(spec["output_root"])
    _write_csv(output_root / "architecture_leaderboard.csv", ranked)
    winner = ranked[0]
    payload = {
        "experiment": spec["experiment"],
        "selection_rule": (
            "global architecture: maximize dataset-macro paired F1 gain of "
            "FMT over same-width train-only Raw-PCA; tie-break by paired AP, "
            "positive dataset count, worst dataset, and worst paired seed"
        ),
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "feature_selection_sha256": spec["feature_selection_sha256"],
        "opened_only_exposed_development_populations": True,
        "confirmation_opened": False,
        "paired_seeds": spec["paired_seeds"],
        "winner": winner,
        "target_dataset_macro_f1_gain": float(
            spec["architecture_selection"]["target_dataset_macro_f1_gain"]
        ),
        "target_reached": float(
            winner["dataset_macro_f1_gain_vs_raw_pca"]
        ) >= float(spec["architecture_selection"][
            "target_dataset_macro_f1_gain"
        ]),
        "leaderboard": ranked,
    }
    target = output_root / "architecture_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("candidate", "preflight", "select"), required=True
    )
    parser.add_argument("--job-index", type=int)
    arguments = parser.parse_args()
    if arguments.mode == "select":
        select(arguments.config)
    elif arguments.mode == "preflight":
        preflight(arguments.config)
    elif arguments.job_index is None:
        parser.error("candidate mode requires --job-index")
    else:
        run_job(arguments.config, arguments.job_index)


if __name__ == "__main__":
    main()
