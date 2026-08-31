"""Strictly reconstruct missing frozen Raw dependencies for Task3 8.1.

This is an infrastructure repair, not a parameter search.  It never reads a
confirmation label or metric.  Two independent V100 rebuilds must reproduce
the preserved 3.2 validation table and each other before any checkpoint is
installed at the historical path expected by the frozen residual models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
FLOAT_FIELDS = (
    "validation_threshold",
    "train_positive_fraction",
    "validation_positive_fraction",
    "validation_average_precision",
    "validation_roc_auc",
    "validation_f1",
    "validation_balanced_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_predicted_positive_fraction",
)
INTEGER_FIELDS = ("parameter_count", "best_epoch")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a YAML mapping")
    return payload


def _resolve_local(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _paths(spec: dict[str, Any]) -> tuple[Path, Path, Path]:
    source_root = Path(spec["source_repo_root"])
    target_root = Path(spec["target_repo_root"])
    output_root = target_root / spec["target_output_root"]
    return source_root, target_root, output_root


def _require_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != str(expected):
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def _job_map(spec: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(replica), str(group))
        for replica in spec["replicas"]
        for group in spec["groups"]
    ]


def _confirmation_result_paths(output_root: Path) -> list[Path]:
    paths = list((output_root / "shards").glob("*.csv"))
    paths.extend(output_root / name for name in ("per_run.csv", "summary.json"))
    return [path for path in paths if path.exists()]


def preflight(config_path: Path) -> Path:
    spec = _load_yaml(config_path)
    source_root, target_root, output_root = _paths(spec)
    if bool(spec.get("confirmation_metrics_opened")):
        raise RuntimeError("repair config says confirmation metrics were opened")
    if bool(spec.get("confirmation_hyperparameters_changed")):
        raise RuntimeError("repair must not change confirmation hyperparameters")
    existing_results = _confirmation_result_paths(output_root)
    if existing_results:
        raise RuntimeError(
            "confirmation results already exist; repair is no longer blind: "
            + ", ".join(str(path) for path in existing_results)
        )

    hashes = {
        "source_training_script": _require_hash(
            source_root / spec["source_training_script"],
            spec["source_training_script_sha256"],
        ),
        "source_model_module": _require_hash(
            source_root / spec["source_model_module"],
            spec["source_model_module_sha256"],
        ),
    }
    expected_datasets: set[str] = set()
    for group, group_spec in spec["groups"].items():
        source_config = source_root / group_spec["source_config"]
        reference = source_root / group_spec["reference_per_run"]
        hashes[f"{group}_source_config"] = _require_hash(
            source_config, group_spec["source_config_sha256"]
        )
        hashes[f"{group}_reference_per_run"] = _require_hash(
            reference, group_spec["reference_per_run_sha256"]
        )
        source_config_payload = _load_yaml(source_config)
        if source_config_payload["datasets"] != group_spec["datasets"]:
            raise RuntimeError(f"{group}: source dataset order changed")
        expected_datasets.update(map(str, group_spec["datasets"]))

    recipe_path = target_root / spec["frozen_recipe_manifest"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe_datasets = {str(row["dataset"]) for row in recipe["models"]}
    if recipe_datasets != expected_datasets:
        raise RuntimeError("repair datasets differ from frozen recipe datasets")
    manifest = {
        "status": "passed",
        "experiment": spec["experiment"],
        "blind_to_confirmation_metrics": True,
        "confirmation_result_files": 0,
        "job_map": [
            {"job_index": index, "replica": replica, "group": group}
            for index, (replica, group) in enumerate(_job_map(spec))
        ],
        "expected_raw_checkpoints_per_replica": (
            len(expected_datasets) * len(spec["paired_seeds"])
        ),
        "source_sha256": hashes,
        "frozen_recipe_manifest_sha256": _sha256(recipe_path),
    }
    target = output_root / "raw_dependency_rebuild" / "preflight.json"
    return _write_json(target, manifest)


def _runtime_config(spec: dict[str, Any], replica: str, group: str) -> Path:
    source_root, target_root, _ = _paths(spec)
    group_spec = spec["groups"][group]
    source_path = source_root / group_spec["source_config"]
    payload = _load_yaml(source_path)
    for key in ("source_cache_root", "label_cache_root"):
        value = Path(payload[key])
        payload[key] = str(value if value.is_absolute() else source_root / value)
    rebuild_dir = (
        target_root / spec["rebuild_root"] / f"replica_{replica}" / group
        / "baselines"
    )
    payload["experiment"] = (
        f"{spec['experiment']}_replica_{replica}_{group}"
    )
    payload["output_dir"] = str(rebuild_dir)
    payload["variants"] = list(spec["variants"])
    payload["training"]["seeds"] = [int(v) for v in spec["paired_seeds"]]
    runtime_path = rebuild_dir / "rebuild_input.yaml"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return runtime_path


def train_job(config_path: Path, job_index: int) -> None:
    spec = _load_yaml(config_path)
    jobs = _job_map(spec)
    if not 0 <= int(job_index) < len(jobs):
        raise IndexError(f"repair job index outside [0,{len(jobs)})")
    replica, group = jobs[int(job_index)]
    source_root, target_root, output_root = _paths(spec)
    preflight_path = output_root / "raw_dependency_rebuild" / "preflight.json"
    preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_payload.get("status") != "passed":
        raise RuntimeError("repair preflight did not pass")
    runtime_path = _runtime_config(spec, replica, group)
    subprocess.run(
        [
            sys.executable,
            str(source_root / spec["source_training_script"]),
            "--config",
            str(runtime_path),
        ],
        cwd=target_root,
        check=True,
    )


def _read_rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        values = list(csv.DictReader(stream))
    rows: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in values:
        key = (str(row["dataset"]), str(row["variant"]), int(row["seed"]))
        if key in rows:
            raise RuntimeError(f"duplicate row {key} in {path}")
        rows[key] = row
    return rows


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _metric_difference(left: dict[str, str], right: dict[str, str]) -> float:
    for key in INTEGER_FIELDS:
        if int(left[key]) != int(right[key]):
            raise RuntimeError(f"integer metric {key} differs")
    difference = 0.0
    for key in FLOAT_FIELDS:
        a, b = float(left[key]), float(right[key])
        if not math.isfinite(a) or not math.isfinite(b):
            raise RuntimeError(f"non-finite metric {key}")
        difference = max(difference, abs(a - b))
    return difference


def validate(config_path: Path) -> Path:
    spec = _load_yaml(config_path)
    source_root, target_root, output_root = _paths(spec)
    if _confirmation_result_paths(output_root):
        raise RuntimeError("confirmation results appeared before repair validation")
    tolerance = float(spec["metric_tolerance"])
    replicas = [str(value) for value in spec["replicas"]]
    if len(replicas) != 2:
        raise RuntimeError("strict repair requires exactly two replicas")
    replica_a, replica_b = replicas
    copied: list[dict[str, Any]] = []
    maximum_metric_difference = 0.0
    state_hashes: dict[str, dict[str, str]] = {}
    raw_checkpoints: dict[tuple[str, int], Path] = {}

    for group, group_spec in spec["groups"].items():
        reference_all = _read_rows(source_root / group_spec["reference_per_run"])
        expected_keys = {
            (str(dataset), "raw", int(seed))
            for dataset in group_spec["datasets"]
            for seed in spec["paired_seeds"]
        }
        reference = {key: reference_all[key] for key in expected_keys}
        rebuilt: dict[str, dict[tuple[str, str, int], dict[str, str]]] = {}
        checkpoint_roots: dict[str, Path] = {}
        for replica in replicas:
            base = (
                target_root / spec["rebuild_root"] / f"replica_{replica}"
                / group / "baselines"
            )
            rebuilt[replica] = _read_rows(base / "per_run.csv")
            if set(rebuilt[replica]) != expected_keys:
                raise RuntimeError(
                    f"{group}/{replica}: rebuilt row keys are incomplete"
                )
            checkpoint_roots[replica] = base / "checkpoints"

        target_dir = target_root / group_spec["target_checkpoint_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for key in sorted(expected_keys):
            dataset, variant, seed = key
            if variant != "raw":
                raise RuntimeError("repair unexpectedly contains a non-Raw arm")
            for replica in replicas:
                maximum_metric_difference = max(
                    maximum_metric_difference,
                    _metric_difference(reference[key], rebuilt[replica][key]),
                )
            maximum_metric_difference = max(
                maximum_metric_difference,
                _metric_difference(rebuilt[replica_a][key], rebuilt[replica_b][key]),
            )
            paths = {
                replica: checkpoint_roots[replica]
                / f"{dataset}_raw_seed{seed}.pt"
                for replica in replicas
            }
            checkpoints = {
                replica: torch.load(
                    path, map_location="cpu", weights_only=False
                )
                for replica, path in paths.items()
            }
            for replica, checkpoint in checkpoints.items():
                if checkpoint["variant"] != "raw":
                    raise RuntimeError(f"{paths[replica]} is not a Raw checkpoint")
                if checkpoint["dataset"] != dataset or int(checkpoint["seed"]) != seed:
                    raise RuntimeError(f"checkpoint identity changed for {key}")
            hashes = {
                replica: _state_hash(checkpoint["state_dict"])
                for replica, checkpoint in checkpoints.items()
            }
            if hashes[replica_a] != hashes[replica_b]:
                raise RuntimeError(f"independent state rebuilds differ for {key}")
            state_hashes[f"{dataset}/seed{seed}"] = hashes
            destination = target_dir / f"{dataset}_raw_seed{seed}.pt"
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite {destination}")
            shutil.copy2(paths[replica_a], destination)
            raw_checkpoints[(dataset, seed)] = destination
            copied.append({
                "dataset": dataset,
                "seed": seed,
                "group": group,
                "state_dict_sha256": hashes[replica_a],
                "checkpoint_sha256": _sha256(destination),
                "target": str(destination),
            })

    if maximum_metric_difference > tolerance:
        raise RuntimeError(
            f"rebuild metric difference {maximum_metric_difference} exceeds "
            f"tolerance {tolerance}"
        )

    recipe_path = target_root / spec["frozen_recipe_manifest"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    normalization_checks = 0
    for model in recipe["models"]:
        residual = torch.load(
            Path(model["checkpoint"]), map_location="cpu", weights_only=False
        )
        key = (str(model["dataset"]), int(model["seed"]))
        raw = torch.load(
            raw_checkpoints[key], map_location="cpu", weights_only=False
        )
        for name in ("raw_mean", "raw_std"):
            if not np.array_equal(
                np.asarray(residual["normalization"][name]),
                np.asarray(raw["normalization"][name]),
            ):
                raise RuntimeError(
                    f"normalization differs for {key}/{model['source']}/{name}"
                )
        normalization_checks += 1

    result = {
        "status": "passed",
        "experiment": spec["experiment"],
        "blind_to_confirmation_metrics": True,
        "confirmation_result_files": 0,
        "replicas": replicas,
        "raw_checkpoint_count": len(copied),
        "frozen_residual_normalization_checks": normalization_checks,
        "maximum_metric_difference_vs_preserved_3_2": maximum_metric_difference,
        "metric_tolerance": tolerance,
        "all_independent_state_dicts_equal": True,
        "copied_checkpoints": copied,
        "state_dict_sha256_by_replica": state_hashes,
        "frozen_recipe_manifest_sha256": _sha256(recipe_path),
    }
    return _write_json(
        output_root / "raw_dependency_rebuild" / "validated_manifest.json",
        result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "train", "validate"), required=True)
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    config_path = _resolve_local(args.config).resolve()
    if args.mode == "preflight":
        target = preflight(config_path)
        print(target.read_text(encoding="utf-8"))
    elif args.mode == "train":
        if args.job_index is None:
            parser.error("train mode requires --job-index")
        train_job(config_path, args.job_index)
    else:
        target = validate(config_path)
        print(target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
