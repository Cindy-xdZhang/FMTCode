"""Replay the frozen Task3 22.1 winner on the exposed 12.2 population.

This is retrospective development analysis, not a new confirmation.  It
loads the already-frozen 22.1 checkpoints, thresholds, residual scales, and
Raw-PCA transforms, then evaluates both paired arms on the third spatial
population that was opened by 12.2.  Nothing is fitted or selected here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from Confirm_Task3_CombinedOptimization_12_1 import _aggregate
from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_residual,
    _load_residual,
)
from Run_Task3_FMTResidual_Frozen_4_1 import _load_confirmation
from Search_Task3_FMTResidual_3D import _read_csv, _write_csv
from Verify_Task3_FMTClassifier import _stack_split


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _under(root: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else root / value


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "status", "source_model",
        "replay_population", "confirmation_roots", "datasets",
        "paired_seeds", "confirmation_count", "expected_ivd_percentile",
        "require_confirmation_reference_match", "batch_size",
        "target_dataset_macro_f1_gain",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing Task3 46.1 config keys: {missing}")
    if str(spec["status"]) != "exposed_development_replay":
        raise ValueError("Task3 46.1 must be marked exposed_development_replay")
    datasets = [str(value) for value in spec["datasets"]]
    seeds = [int(value) for value in spec["paired_seeds"]]
    if len(datasets) != 10 or len(datasets) != len(set(datasets)):
        raise ValueError("Task3 46.1 requires ten unique datasets")
    if seeds != [40, 41]:
        raise ValueError("Task3 46.1 replays the two frozen 22.1 seeds")
    if int(spec["confirmation_count"]) != 4:
        raise ValueError("Task3 46.1 requires four replay slices")
    if not np.isclose(float(spec["expected_ivd_percentile"]), 95.0):
        raise ValueError("Task3 46.1 requires whole-field IVD-p95 labels")
    if not bool(spec["require_confirmation_reference_match"]):
        raise ValueError("Task3 46.1 must require source/reference identity")
    phase = [float(value) for value in spec["replay_population"][
        "seed_grid_phase"
    ]]
    if phase != [0.318359375, 0.4561042524005485, -0.3352]:
        raise ValueError("Task3 46.1 replay phase changed")
    grouped = []
    for group in spec["confirmation_roots"].values():
        grouped.extend(str(value) for value in group.get("datasets", []))
    if len(grouped) != len(set(grouped)) or set(grouped) != set(datasets):
        raise ValueError("Task3 46.1 roots must partition all datasets")
    for section in ("source_model", "replay_population"):
        for key, value in spec[section].get("sha256", {}).items():
            if len(str(value)) != 64:
                raise ValueError(f"{section}.sha256.{key} is not full SHA-256")
    if float(spec["target_dataset_macro_f1_gain"]) != 0.15:
        raise ValueError("Task3 46.1 target must remain +0.15 F1")
    spec["datasets"] = datasets
    spec["paired_seeds"] = seeds
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    return spec


def _repo_root(section: dict, environment_name: str) -> Path:
    override = os.environ.get(environment_name)
    return Path(override if override else section["repo_root"])


def _verify_artifact_hashes(root: Path, section: dict) -> dict[str, Path]:
    paths = {}
    for key, relative in section["paths"].items():
        path = _under(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(section["sha256"][key]).lower()
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen artifact changed: {key} ({path})")
        paths[key] = path
    return paths


def _group_for_dataset(source: dict, dataset: str) -> tuple[str, dict]:
    matches = [
        (name, group) for name, group in source["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} matched {len(matches)} groups")
    return matches[0]


def _selected_candidate(source: dict, selection: dict,
                        dataset: str) -> tuple[str, dict]:
    family, _ = _group_for_dataset(source, dataset)
    selected = selection["primary_by_group"][family]
    candidate_id = str(selected["candidate_id"])
    candidates = {
        str(candidate["id"]): dict(candidate)
        for candidate in source["candidates"]
    }
    if candidate_id not in candidates:
        raise RuntimeError(f"selected 22.1 candidate missing: {candidate_id}")
    candidate = candidates[candidate_id]
    if str(selected["fmt_feature"]) != str(candidate["fmt_feature"]):
        raise RuntimeError(f"selected 22.1 feature changed for {family}")
    return family, candidate


def _source_state(spec: dict) -> tuple:
    section = spec["source_model"]
    root = _repo_root(section, "TASK46_SOURCE_MODEL_ROOT")
    paths = _verify_artifact_hashes(root, section)
    source = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    expected = str(section["expected_experiment"])
    for name, payload in (("config", source), ("preflight", preflight),
                          ("selection", selection)):
        if str(payload.get("experiment")) != expected:
            raise RuntimeError(f"22.1 {name} experiment changed")
    if bool(preflight.get("confirmation_opened", True)):
        raise RuntimeError("22.1 preflight opened confirmation")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("22.1 selection opened confirmation")
    if source.get("datasets") != spec["datasets"]:
        raise RuntimeError("22.1 dataset order changed")
    if set(selection.get("primary_by_group", {})) != set(source["groups"]):
        raise RuntimeError("22.1 selected physical families changed")
    for dataset in spec["datasets"]:
        _selected_candidate(source, selection, dataset)
    return root, paths, source, preflight, selection


def _replay_state(spec: dict) -> tuple:
    section = spec["replay_population"]
    root = _repo_root(section, "TASK46_REPLAY_POPULATION_ROOT")
    paths = _verify_artifact_hashes(root, section)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if str(manifest.get("experiment")) != str(section["source_experiment"]):
        raise RuntimeError("12.2 frozen manifest experiment changed")
    if str(summary.get("experiment")) != str(section["source_experiment"]):
        raise RuntimeError("12.2 summary experiment changed")
    phase = [float(value) for value in section["seed_grid_phase"]]
    if list(manifest.get("confirmation_seed_grid_phase", [])) != phase:
        raise RuntimeError("12.2 frozen replay phase changed")
    if list(summary.get("confirmation_seed_grid_phase", [])) != phase:
        raise RuntimeError("12.2 summary replay phase changed")
    if not bool(summary.get("confirmation_data_was_not_used_for_selection")):
        raise RuntimeError("12.2 selection boundary changed")
    return root, paths, manifest, summary


def _source_result_path(source_root: Path, source: dict, candidate: dict,
                        dataset: str, seed: int, arm: str) -> Path:
    return (
        _under(source_root, source["output_root"]) / "stage1" / "candidates"
        / str(candidate["id"]) / dataset / f"seed{int(seed)}" / arm
        / "per_run.csv"
    )


def _checkpoint_from_result(source_root: Path, result_path: Path,
                            expected: dict) -> tuple[Path, dict]:
    rows = _read_csv(result_path)
    if len(rows) != 1:
        raise RuntimeError(f"source result missing or duplicated: {result_path}")
    row = rows[0]
    for key, value in expected.items():
        if str(row.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"22.1 source result changed: {key}")
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = source_root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_parent = (result_path.parent / "checkpoints").resolve()
    if checkpoint.resolve().parent != expected_parent:
        raise RuntimeError(f"checkpoint escaped result directory: {checkpoint}")
    return checkpoint, row


def _confirmation_group(spec: dict, dataset: str) -> dict:
    matches = [
        group for group in spec["confirmation_roots"].values()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"replay dataset {dataset} matched {len(matches)} roots")
    return matches[0]


def preflight(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    source_root, source_paths, source, _, selection = _source_state(spec)
    replay_root, replay_paths, replay_manifest, replay_summary = _replay_state(spec)
    models = []
    for dataset in spec["datasets"]:
        family, candidate = _selected_candidate(source, selection, dataset)
        for seed in spec["paired_seeds"]:
            paired = []
            for arm, variant in (
                ("fmt", "raw_fmt_residual"),
                ("raw_pca", "raw_pca_residual"),
            ):
                result_path = _source_result_path(
                    source_root, source, candidate, dataset, seed, arm
                )
                checkpoint, row = _checkpoint_from_result(
                    source_root, result_path, {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "auxiliary_source": arm,
                        "candidate_id": candidate["id"],
                        "fmt_feature": candidate["fmt_feature"],
                    },
                )
                item = {
                    "dataset": dataset,
                    "physical_family": family,
                    "seed": int(seed),
                    "source": arm,
                    "variant": variant,
                    "candidate_id": str(candidate["id"]),
                    "fmt_feature": str(candidate["fmt_feature"]),
                    "fmt_dim": int(row["fmt_dim"]),
                    "parameter_count": int(row["parameter_count"]),
                    "trainable_residual_parameter_count": int(
                        row["trainable_residual_parameter_count"]
                    ),
                    "result": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
                models.append(item)
                paired.append(item)
            for key in (
                "fmt_dim", "parameter_count",
                "trainable_residual_parameter_count",
            ):
                if paired[0][key] != paired[1][key]:
                    raise RuntimeError(
                        f"22.1 paired-arm {key} mismatch: {dataset}/seed{seed}"
                    )
    counts = {}
    for dataset in spec["datasets"]:
        group = _confirmation_group(spec, dataset)
        source_dir = Path(group["source_root"]) / dataset
        label_dir = Path(group["label_root"]) / dataset
        source_count = len(list(source_dir.glob("*.npz")))
        label_count = len(list(label_dir.glob("*.npz")))
        if source_count != int(spec["confirmation_count"]):
            raise RuntimeError(f"unexpected replay source count: {dataset}")
        if label_count != int(spec["confirmation_count"]):
            raise RuntimeError(f"unexpected replay label count: {dataset}")
        counts[dataset] = {"source_npz": source_count, "label_npz": label_count}
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": spec["config_sha256"],
        "source_model_repo_root": str(source_root),
        "source_model_artifact_sha256": {
            key: _sha256(path) for key, path in source_paths.items()
        },
        "source_model_experiment": source["experiment"],
        "replay_population_repo_root": str(replay_root),
        "replay_population_artifact_sha256": {
            key: _sha256(path) for key, path in replay_paths.items()
        },
        "replay_population_experiment": replay_manifest["experiment"],
        "replay_population_seed_grid_phase": spec["replay_population"][
            "seed_grid_phase"
        ],
        "replay_population_prior_f1_gain": float(
            replay_summary["dataset_macro_f1_gain_vs_raw_pca"]
        ),
        "fresh_confirmation_opened": False,
        "replayed_population_is_already_exposed": True,
        "dataset_count": len(spec["datasets"]),
        "paired_seed_count": len(spec["paired_seeds"]),
        "paired_arm_count": 2,
        "expected_evaluations": len(models),
        "confirmation_artifact_counts": counts,
        "models": models,
    }
    target = Path(spec["output_root"]) / "preflight_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError("Task3 46.1 preflight manifest changed")
    else:
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(target)
    return target


def _load_manifest(spec: dict) -> tuple[Path, dict]:
    path = Path(spec["output_root"]) / "preflight_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": spec["config_sha256"],
        "fresh_confirmation_opened": False,
        "replayed_population_is_already_exposed": True,
        "expected_evaluations": 40,
    }
    for key, value in expected.items():
        if str(manifest.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"Task3 46.1 preflight changed: {key}")
    return path, manifest


def _model_entry(manifest: dict, dataset: str, seed: int,
                 source: str) -> dict:
    matches = [
        row for row in manifest["models"]
        if row["dataset"] == dataset
        and int(row["seed"]) == int(seed)
        and row["source"] == source
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Task3 46.1 model entry missing: {dataset}/seed{seed}/{source}"
        )
    return matches[0]


def _shard_path(spec: dict, dataset: str) -> Path:
    return Path(spec["output_root"]) / "shards" / f"{dataset}.csv"


def _validate_shard_rows(rows: list[dict], spec: dict, manifest: dict,
                         manifest_hash: str, dataset: str,
                         require_complete: bool) -> None:
    expected_keys = {
        (source, seed)
        for seed in spec["paired_seeds"]
        for source in ("fmt", "raw_pca")
    }
    observed = []
    for row in rows:
        key = (str(row.get("source")), int(row.get("seed", -1)))
        observed.append(key)
        model = _model_entry(manifest, dataset, key[1], key[0])
        expected = {
            "experiment": spec["experiment"],
            "status": spec["status"],
            "config_sha256": spec["config_sha256"],
            "preflight_manifest_sha256": manifest_hash,
            "dataset": dataset,
            "physical_family": model["physical_family"],
            "candidate_id": model["candidate_id"],
            "fmt_feature": model["fmt_feature"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "method": (
                "fmt_residual" if key[0] == "fmt"
                else "raw_pca_residual"
            ),
        }
        for name, value in expected.items():
            if str(row.get(name, "")).lower() != str(value).lower():
                raise RuntimeError(f"stale Task3 46.1 shard: {dataset}/{name}")
        if key not in expected_keys:
            raise RuntimeError(f"unexpected Task3 46.1 key: {dataset}/{key}")
        for metric in ("f1", "average_precision"):
            if not np.isfinite(float(row.get(metric, "nan"))):
                raise RuntimeError(f"non-finite Task3 46.1 {dataset}/{metric}")
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"duplicate Task3 46.1 row: {dataset}")
    if not set(observed).issubset(expected_keys):
        raise RuntimeError(f"unexpected Task3 46.1 rows: {dataset}")
    if require_complete and set(observed) != expected_keys:
        raise RuntimeError(f"incomplete Task3 46.1 shard: {dataset}")
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed, paired in by_seed.items():
        if len(paired) == 2:
            sample_counts = {int(row["sample_count"]) for row in paired}
            positive_fractions = {
                round(float(row["positive_fraction"]), 12) for row in paired
            }
            if len(sample_counts) != 1 or len(positive_fractions) != 1:
                raise RuntimeError(
                    f"Task3 46.1 paired targets differ: {dataset}/seed{seed}"
                )


def run_dataset(config_path: str | Path, dataset: str) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown Task3 46.1 dataset {dataset!r}")
    source_root, _, source, _, selection = _source_state(spec)
    _replay_state(spec)
    manifest_path, manifest = _load_manifest(spec)
    manifest_hash = _sha256(manifest_path)
    family, candidate = _selected_candidate(source, selection, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = _load_confirmation(spec, source, dataset, candidate, device)
    confirmation = _stack_split(
        records, list(range(int(spec["confirmation_count"])))
    )
    target = _shard_path(spec, dataset)
    rows = _read_csv(target)
    _validate_shard_rows(
        rows, spec, manifest, manifest_hash, dataset, require_complete=False
    )
    completed = {(row["source"], int(row["seed"])) for row in rows}
    for seed in spec["paired_seeds"]:
        for arm in ("fmt", "raw_pca"):
            if (arm, seed) in completed:
                continue
            model_entry = _model_entry(manifest, dataset, seed, arm)
            checkpoint_path = Path(model_entry["checkpoint"])
            if _sha256(checkpoint_path) != model_entry["checkpoint_sha256"]:
                raise RuntimeError(
                    f"22.1 checkpoint changed: {dataset}/seed{seed}/{arm}"
                )
            model, checkpoint = _load_residual(
                checkpoint_path, confirmation[1].shape[1], device,
                checkpoint_root=source_root,
            )
            targets, _, metrics = _evaluate_residual(
                model, checkpoint, confirmation, int(spec["batch_size"]),
                int(seed), device,
            )
            rows.append({
                "experiment": spec["experiment"],
                "status": spec["status"],
                "config_sha256": spec["config_sha256"],
                "preflight_manifest_sha256": manifest_hash,
                "dataset": dataset,
                "physical_family": family,
                "candidate_id": candidate["id"],
                "fmt_feature": candidate["fmt_feature"],
                "seed": int(seed),
                "source": arm,
                "method": (
                    "fmt_residual" if arm == "fmt"
                    else "raw_pca_residual"
                ),
                "sample_count": int(len(targets)),
                "positive_fraction": float(targets.mean()),
                "frozen_threshold": float(checkpoint["threshold"]),
                "frozen_alpha": float(checkpoint["alpha"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": model_entry["checkpoint_sha256"],
                **metrics,
            })
            _write_csv(target, rows)
            completed.add((arm, seed))
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"Task3 46.1 {dataset} seed={seed} complete", flush=True)
    return target


def summarize(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    _, _, _, _, selection = _source_state(spec)
    _, replay_paths, _, prior_summary = _replay_state(spec)
    manifest_path, manifest = _load_manifest(spec)
    manifest_hash = _sha256(manifest_path)
    rows = []
    for dataset in spec["datasets"]:
        shard = _read_csv(_shard_path(spec, dataset))
        _validate_shard_rows(
            shard, spec, manifest, manifest_hash, dataset,
            require_complete=True,
        )
        rows.extend(shard)
    output = Path(spec["output_root"])
    _write_csv(output / "per_run.csv", rows)
    aggregate = _aggregate(rows, spec["datasets"])
    result = {
        "experiment": spec["experiment"],
        "status": spec["status"],
        "comparison": (
            "frozen 22.1 FMT residual minus its same-width, same-structure "
            "train-only Raw-PCA residual"
        ),
        "fresh_confirmation_opened": False,
        "replayed_population_is_already_exposed": True,
        "replayed_population_source_experiment": spec[
            "replay_population"
        ]["source_experiment"],
        "replayed_population_summary_sha256": _sha256(
            replay_paths["summary"]
        ),
        "replayed_population_seed_grid_phase": spec[
            "replay_population"
        ]["seed_grid_phase"],
        "preflight_manifest_sha256": manifest_hash,
        "source_model_selection_sha256": spec["source_model"]["sha256"][
            "selection"
        ],
        "paired_seeds": spec["paired_seeds"],
        "source_development_metrics": {
            "raw_pca_f1": float(selection[
                "development_dataset_macro_raw_pca_f1"
            ]),
            "fmt_f1": float(selection["development_dataset_macro_fmt_f1"]),
            "f1_gain": float(selection[
                "development_dataset_macro_f1_gain_vs_raw_pca"
            ]),
            "raw_pca_average_precision": float(selection[
                "development_dataset_macro_raw_pca_ap"
            ]),
            "fmt_average_precision": float(selection[
                "development_dataset_macro_fmt_ap"
            ]),
            "average_precision_gain": float(selection[
                "development_dataset_macro_ap_gain_vs_raw_pca"
            ]),
        },
        "prior_12_2_metrics": {
            "method": "11.1 frozen winner",
            "f1_gain": float(prior_summary[
                "dataset_macro_f1_gain_vs_raw_pca"
            ]),
            "average_precision_gain": float(prior_summary[
                "dataset_macro_ap_gain_vs_raw_pca"
            ]),
        },
        **aggregate,
    }
    target_gain = float(spec["target_dataset_macro_f1_gain"])
    result.update({
        "target_dataset_macro_f1_gain": target_gain,
        "target_reached_on_exposed_replay": (
            result["dataset_macro_f1_gain_vs_raw_pca"] >= target_gain
        ),
        "f1_gain_change_from_22_1_development": (
            result["dataset_macro_f1_gain_vs_raw_pca"]
            - result["source_development_metrics"]["f1_gain"]
        ),
    })
    target = output / "summary.json"
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


def _decode_job(spec: dict, index: int) -> str:
    index = int(index)
    if not 0 <= index < len(spec["datasets"]):
        raise IndexError(f"job index {index} outside [0,{len(spec['datasets'])})")
    return spec["datasets"][index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "dataset", "summary"), required=True
    )
    parser.add_argument("--dataset")
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.config)
    elif args.mode == "dataset":
        spec = _load_spec(args.config)
        dataset = args.dataset
        if dataset is None:
            if args.job_index is None:
                parser.error("dataset mode requires --dataset or --job-index")
            dataset = _decode_job(spec, args.job_index)
        run_dataset(args.config, dataset)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
