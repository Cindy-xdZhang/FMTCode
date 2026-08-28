"""Paired loss and optimization search for supervised 3D Task3.

The experiment waits for the development-only 5.2 selector, freezes its
family-specific feature/network recipes in a preflight manifest, and then
changes the same training recipe in both the FMT and train-only Raw-PCA arms.
No confirmation population is opened by this script.
"""

from __future__ import annotations

import argparse
import json
import hashlib
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
from Search_Task3_FMTResidual_Stage2_3D import _combined_candidate
from Verify_Task3_FMTClassifier import _append_csv, _normalize_train_only
from Verify_Task3_FMTResidual import (
    _build_training_loss,
    _load_raw_model,
    _train_one,
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_text_sha256(path: str | Path) -> str:
    """Hash text with LF newlines so Git content is OS-independent."""
    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_optimization_spec(path: str | Path) -> dict:
    path = Path(path)
    overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "base_search_config",
        "base_search_config_sha256", "upstream_selection", "paired_seeds",
        "model_override", "optimization_candidates", "selection",
    }
    missing = sorted(required.difference(overlay))
    if missing:
        raise ValueError(f"missing optimization config keys: {missing}")
    base_path = Path(overlay["base_search_config"])
    base_hash = _canonical_text_sha256(base_path)
    if base_hash != str(overlay["base_search_config_sha256"]).lower():
        raise RuntimeError("base Task3 development config changed")
    base = _load_spec(base_path)
    seeds = [int(value) for value in overlay["paired_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("paired_seeds must be non-empty and unique")
    if not set(seeds).issubset({int(value) for value in base["stage2_screen_seeds"]}):
        raise ValueError("paired seeds require unavailable frozen Raw checkpoints")
    candidates = [dict(row) for row in overlay["optimization_candidates"]]
    candidate_ids = [str(row["id"]) for row in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("optimization candidate ids must be non-empty and unique")
    if bool(overlay["selection"].get("confirmation_opened", False)):
        raise RuntimeError("optimization search must not open confirmation")
    spec = dict(base)
    spec.update({
        "experiment": str(overlay["experiment"]),
        "output_root": str(overlay["output_root"]),
        "optimization_config": str(path),
        "optimization_config_sha256": _sha256(path),
        "base_search_config": str(base_path),
        "base_search_config_sha256": base_hash,
        "upstream_selection": str(overlay["upstream_selection"]),
        "upstream_selector_job_id": str(
            overlay.get("upstream_selector_job_id", "")
        ),
        "paired_seeds": seeds,
        "stage2_screen_seeds": seeds,
        "model_override": dict(overlay["model_override"]),
        "optimization_candidates": candidates,
        "optimization_selection": dict(overlay["selection"]),
    })
    return spec


def _manifest_path(spec: dict) -> Path:
    return Path(spec["output_root"]) / "preflight_manifest.json"


def _upstream_base_candidates(spec: dict, selection: dict) -> dict[str, dict]:
    if bool(selection.get("confirmation_opened", False)):
        raise RuntimeError("upstream selection unexpectedly opened confirmation")
    if int(selection.get("stage", -1)) != 2:
        raise RuntimeError("Task3 7.1 requires a completed stage-2 selection")
    primary = selection["primary_by_group"]
    if set(primary) != set(spec["groups"]):
        raise RuntimeError("upstream selection physical families changed")
    feature_lookup = {str(row["id"]): dict(row) for row in spec["candidates"]}
    network_lookup = {
        str(row["id"]): dict(row) for row in spec["stage2_networks"]
    }
    result = {}
    for group, row in primary.items():
        feature_id = str(row["feature_candidate_id"])
        network_id = str(row["network_id"])
        if feature_id not in feature_lookup or network_id not in network_lookup:
            raise RuntimeError(f"unknown upstream recipe for {group}")
        candidate = _combined_candidate(
            feature_lookup[feature_id], network_lookup[network_id]
        )
        if str(candidate["id"]) != str(row["candidate_id"]):
            raise RuntimeError(f"upstream candidate identity changed for {group}")
        candidate.update(spec["model_override"])
        candidate["upstream_candidate_id"] = str(row["candidate_id"])
        result[group] = candidate
    return result


def _load_manifest(spec: dict) -> dict:
    path = _manifest_path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"preflight manifest is required before training: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
    }
    for key, value in expected.items():
        if str(manifest.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"preflight manifest changed: {key}")
    selection_path = Path(spec["upstream_selection"])
    if _sha256(selection_path) != str(
        manifest["upstream_selection_sha256"]
    ).lower():
        raise RuntimeError("upstream 5.2 selection changed after 7.1 preflight")
    if bool(manifest.get("confirmation_opened", True)):
        raise RuntimeError("invalid preflight confirmation state")
    return manifest


def _optimization_candidate(spec: dict, manifest: dict, dataset: str,
                            candidate_index: int) -> dict:
    group_name, _ = _group_for_dataset(spec, dataset)
    index = int(candidate_index)
    if not 0 <= index < len(spec["optimization_candidates"]):
        raise IndexError("optimization candidate index outside configured grid")
    base = dict(manifest["base_candidate_by_group"][group_name])
    recipe = dict(spec["optimization_candidates"][index])
    training = dict(base.get("training", {}))
    training.update(recipe.get("training", {}))
    model = dict(recipe.get("model", {}))
    base.update(model)
    base["training"] = training
    base["id"] = str(recipe["id"])
    base["optimization_id"] = str(recipe["id"])
    base["optimization_recipe"] = recipe
    return base


def _parameter_budget(spec: dict, group: dict, candidate: dict,
                      dataset: str, fmt_dim: int) -> dict:
    totals, trainables = [], []
    for seed in spec["paired_seeds"]:
        raw_model, _ = _load_raw_model(
            Path(group["raw_checkpoint_dir"])
            / f"{dataset}_raw_seed{int(seed)}.pt",
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
        raise RuntimeError("paired seeds changed parameter counts")
    limit = int(spec["raw_wide_parameter_count"])
    return {
        "eligible": int(totals[0]) < limit,
        "total_parameter_count": int(totals[0]),
        "trainable_residual_parameter_count": int(trainables[0]),
        "raw_wide_parameter_count": limit,
    }


def static_preflight(config_path: str) -> None:
    spec = _load_optimization_spec(config_path)
    payload = {
        "experiment": spec["experiment"],
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "upstream_selection": spec["upstream_selection"],
        "upstream_selection_exists": Path(spec["upstream_selection"]).exists(),
        "dataset_count": len(spec["datasets"]),
        "physical_family_count": len(spec["groups"]),
        "optimization_candidate_count": len(spec["optimization_candidates"]),
        "paired_seed_count": len(spec["paired_seeds"]),
        "paired_arm_count": 2,
        "expected_training_runs": (
            len(spec["datasets"]) * len(spec["optimization_candidates"])
            * len(spec["paired_seeds"]) * 2
        ),
        "confirmation_opened": False,
    }
    print(json.dumps(payload, indent=2))


def preflight(config_path: str) -> Path:
    spec = _load_optimization_spec(config_path)
    selection_path = Path(spec["upstream_selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    base_candidates = _upstream_base_candidates(spec, selection)
    manifest_stub = {"base_candidate_by_group": base_candidates}
    datasets = []
    for dataset in spec["datasets"]:
        group_name, group = _group_for_dataset(spec, dataset)
        first = _optimization_candidate(spec, manifest_stub, dataset, 0)
        train, validation = _load_search_splits(
            spec, dataset, first, torch.device("cpu")
        )
        fmt_dim = int(train[1].shape[1])
        raw_dim = int(train[0].reshape(len(train[0]), -1).shape[1])
        if fmt_dim > raw_dim:
            raise RuntimeError(f"{dataset}: FMT width exceeds Raw width")
        if not (0 < int(train[2].sum()) < len(train[2])):
            raise RuntimeError(f"{dataset}: training labels are single-class")
        if not (0 < int(validation[2].sum()) < len(validation[2])):
            raise RuntimeError(f"{dataset}: validation labels are single-class")
        recipes = []
        for index in range(len(spec["optimization_candidates"])):
            candidate = _optimization_candidate(
                spec, manifest_stub, dataset, index
            )
            budget = _parameter_budget(
                spec, group, candidate, dataset, fmt_dim
            )
            if not budget["eligible"]:
                raise RuntimeError(
                    f"{dataset}/{candidate['id']} exceeds Raw-wide capacity"
                )
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, spec["paired_seeds"][0],
                "fmt", Path(spec["output_root"]) / "preflight", fmt_dim,
            )
            _build_training_loss(
                run_spec["training"], float(train[2].sum()),
                float(len(train[2]) - train[2].sum()), torch.device("cpu"),
            )
            recipes.append({"optimization_id": candidate["id"], **budget})
        datasets.append({
            "dataset": dataset,
            "physical_family": group_name,
            "fmt_feature": first["fmt_feature"],
            "upstream_candidate_id": first["upstream_candidate_id"],
            "fmt_dim": fmt_dim,
            "raw_dim": raw_dim,
            "training_samples": int(len(train[2])),
            "training_positive_count": int(train[2].sum()),
            "validation_samples": int(len(validation[2])),
            "validation_positive_count": int(validation[2].sum()),
            "recipes": recipes,
        })
    payload = {
        "experiment": spec["experiment"],
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "upstream_selection_sha256": _sha256(selection_path),
        "upstream_selector_job_id": spec["upstream_selector_job_id"],
        "base_candidate_by_group": base_candidates,
        "confirmation_opened": False,
        "dataset_count": len(spec["datasets"]),
        "optimization_candidate_count": len(spec["optimization_candidates"]),
        "paired_seed_count": len(spec["paired_seeds"]),
        "paired_arm_count": 2,
        "expected_training_runs": (
            len(spec["datasets"]) * len(spec["optimization_candidates"])
            * len(spec["paired_seeds"]) * 2
        ),
        "datasets": datasets,
    }
    target = _manifest_path(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target)
    return target


def _result_path(spec: dict, candidate: dict, dataset: str,
                 seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "candidates" / str(candidate["id"])
        / dataset / f"seed{int(seed)}" / source / "per_run.csv"
    )


def run_candidate(config_path: str, dataset: str,
                  candidate_index: int) -> Path:
    spec = _load_optimization_spec(config_path)
    manifest = _load_manifest(spec)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    candidate = _optimization_candidate(
        spec, manifest, dataset, candidate_index
    )
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
    budget = _parameter_budget(spec, group, candidate, dataset, fmt_dim)
    if not budget["eligible"]:
        raise RuntimeError(f"{dataset}/{candidate['id']} exceeds parameter cap")
    last = None
    manifest_hash = _sha256(_manifest_path(spec))
    for seed in spec["paired_seeds"]:
        for source in ("fmt", "raw_pca"):
            path = _result_path(spec, candidate, dataset, seed, source)
            rows = _read_csv(path)
            if len(rows) > 1:
                raise RuntimeError(f"duplicate optimization result: {path}")
            if rows:
                expected = {
                    "optimization_config_sha256": spec[
                        "optimization_config_sha256"
                    ],
                    "preflight_manifest_sha256": manifest_hash,
                    "upstream_selection_sha256": manifest[
                        "upstream_selection_sha256"
                    ],
                }
                for key, value in expected.items():
                    if str(rows[0].get(key, "")).lower() != str(value).lower():
                        raise RuntimeError(f"stale cached result {path}: {key}")
                print(f"cached {candidate['id']} {dataset} seed={seed} {source}")
                last = path
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, seed, source,
                path.parent, fmt_dim,
            )
            (path.parent / "config_snapshot.yaml").write_text(
                yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8"
            )
            row = _train_one(
                run_spec, dataset, seed, (train, validation, None), stats,
                device, path.parent,
            )
            row.update({
                "optimization_id": candidate["optimization_id"],
                "optimization_recipe_json": json.dumps(
                    candidate["optimization_recipe"], sort_keys=True
                ),
                "upstream_candidate_id": candidate["upstream_candidate_id"],
                "fmt_feature": candidate["fmt_feature"],
                "fmt_dim": fmt_dim,
                "optimization_config_sha256": spec[
                    "optimization_config_sha256"
                ],
                "preflight_manifest_sha256": manifest_hash,
                "upstream_selection_sha256": manifest[
                    "upstream_selection_sha256"
                ],
            })
            _append_csv(path, row)
            last = path
            print(
                f"DONE {candidate['id']} {dataset} seed={seed} {source}: "
                f"F1={row['validation_f1']:.5f} "
                f"AP={row['validation_average_precision']:.5f}",
                flush=True,
            )
    return last


def _decode_job(spec: dict, job_index: int) -> tuple[str, int]:
    per_dataset = len(spec["optimization_candidates"])
    total = len(spec["datasets"]) * per_dataset
    index = int(job_index)
    if not 0 <= index < total:
        raise IndexError(f"optimization job index outside [0,{total})")
    dataset_index, candidate_index = divmod(index, per_dataset)
    return spec["datasets"][dataset_index], candidate_index


def run_job(config_path: str, job_index: int) -> Path:
    spec = _load_optimization_spec(config_path)
    dataset, candidate_index = _decode_job(spec, job_index)
    return run_candidate(config_path, dataset, candidate_index)


def _candidate_summary(spec: dict, manifest: dict, group_name: str,
                       recipe: dict) -> dict:
    datasets = spec["groups"][group_name]["datasets"]
    recipe_index = next(
        index for index, row in enumerate(spec["optimization_candidates"])
        if str(row["id"]) == str(recipe["id"])
    )
    per_dataset = {}
    seed_gains = {int(seed): [] for seed in spec["paired_seeds"]}
    parameter_counts = set()
    for dataset in datasets:
        candidate = _optimization_candidate(
            spec, manifest, dataset, recipe_index
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
                        f"incomplete optimization result {candidate['id']}/"
                        f"{dataset}/seed={seed}/{source}"
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
    return {
        "physical_family": group_name,
        "optimization_id": str(recipe["id"]),
        "optimization_recipe_json": json.dumps(recipe, sort_keys=True),
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
        "worst_dataset_f1_gain": float(min(f1_gains)),
        "worst_seed_f1_gain": float(min(
            np.mean(values) for values in seed_gains.values()
        )),
        "minimum_total_parameter_count": min(parameter_counts),
        "maximum_total_parameter_count": max(parameter_counts),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
        "seed_gains_json": json.dumps({
            str(seed): float(np.mean(values))
            for seed, values in seed_gains.items()
        }, sort_keys=True),
    }


def select(config_path: str) -> Path:
    spec = _load_optimization_spec(config_path)
    manifest = _load_manifest(spec)
    primary, leaderboard = {}, []
    selection = spec["optimization_selection"]
    required = (
        str(selection["primary_metric"]),
        *tuple(selection["tie_breakers"]),
    )
    for group_name in spec["groups"]:
        rows = [
            _candidate_summary(spec, manifest, group_name, recipe)
            for recipe in spec["optimization_candidates"]
        ]
        for key in required:
            if any(key not in row for row in rows):
                raise KeyError(f"unknown optimization selection key {key!r}")
        ranked = sorted(
            rows,
            key=lambda row: tuple(float(row[key]) for key in required),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["rank_within_family"] = rank
            leaderboard.append(row)
        primary[group_name] = ranked[0]
    output_root = Path(spec["output_root"])
    _write_csv(output_root / "optimization_leaderboard.csv", leaderboard)
    dataset_details = []
    for group_name, row in primary.items():
        for dataset, metrics in json.loads(row["datasets_json"]).items():
            dataset_details.append({
                "physical_family": group_name,
                "dataset": dataset,
                "optimization_id": row["optimization_id"],
                **metrics,
            })
    f1_gain = float(np.mean([
        row["f1_gain"] for row in dataset_details
    ]))
    ap_gain = float(np.mean([
        row["average_precision_gain"] for row in dataset_details
    ]))
    target_gain = float(selection["target_dataset_macro_f1_gain"])
    payload = {
        "experiment": spec["experiment"],
        "selection_rule": (
            "family-specific paired optimization: maximize development "
            "dataset-macro F1 gain of FMT over the same-width train-only "
            "Raw-PCA arm; tie-break by AP, positive datasets, worst dataset, "
            "and worst paired seed"
        ),
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "upstream_selection_sha256": manifest[
            "upstream_selection_sha256"
        ],
        "preflight_manifest_sha256": _sha256(_manifest_path(spec)),
        "opened_only_exposed_development_populations": True,
        "confirmation_opened": False,
        "paired_seeds": spec["paired_seeds"],
        "primary_by_group": primary,
        "development_dataset_macro_f1_gain_vs_raw_pca": f1_gain,
        "development_dataset_macro_ap_gain_vs_raw_pca": ap_gain,
        "target_dataset_macro_f1_gain": target_gain,
        "target_reached": f1_gain >= target_gain,
        "dataset_details": dataset_details,
    }
    target = output_root / "optimization_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=("static-preflight", "preflight", "candidate", "select"),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    arguments = parser.parse_args()
    if arguments.mode == "static-preflight":
        static_preflight(arguments.config)
    elif arguments.mode == "preflight":
        preflight(arguments.config)
    elif arguments.mode == "select":
        select(arguments.config)
    elif arguments.job_index is None:
        parser.error("candidate mode requires --job-index")
    else:
        run_job(arguments.config, arguments.job_index)


if __name__ == "__main__":
    main()
