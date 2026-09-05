"""Outer confirmation for the fixed-pathline Task1/Task2/Task3 search.

The development selector never opens ordinals 8--9.  This module refuses to
run until both the selector and the independent development audit have
completed, then evaluates exactly two preregistered roles: the current
``baseline`` and the frozen ``selected_winner``.  Confirmation results are
reported, never used for another selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.Run_Task1_3D_Main import _fit_score
from experiments.Run_Task2_3D_Main import _latent_score, _prepare_inputs
from experiments.Run_Task123_FixedPathline import (
    _atomic_json,
    _load_spec,
    _sha256,
    _task2_source,
    _task3_backbone_spec,
    _task3_residual_spec,
    _task3_split,
    _variant,
    load_common_records,
)
from experiments.Verify_HighReVAE import _train as train_vae
from experiments.Verify_Task3_FMTClassifier import (
    _normalize_train_only,
    _train_one as train_raw_backbone,
)
from experiments.Verify_Task3_FMTResidual import _train_one as train_residual


def _gate(spec: dict, config_path: Path) -> dict:
    root = Path(spec["output_root"])
    selection_path = root / "selection" / "selection.json"
    audit_path = root / "selection" / "independent_audit.json"
    if not selection_path.is_file() or not audit_path.is_file():
        raise RuntimeError("confirmation requires selection and independent audit")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    config_sha = _sha256(config_path)
    if selection.get("config_sha256") != config_sha:
        raise RuntimeError("selection used another config")
    if selection.get("status") != "development_winner_frozen_before_confirmation":
        raise RuntimeError("development winner is not frozen")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("development selector opened confirmation data")
    if audit.get("status") != "PASS":
        raise RuntimeError("development independent audit did not PASS")
    if audit.get("config_sha256") != config_sha:
        raise RuntimeError("development audit used another config")
    if audit.get("selection_sha256") != _sha256(selection_path):
        raise RuntimeError("development audit did not cover the frozen selection")
    winner = str(selection["winner"]["variant"])
    if winner != str(audit["winner"]["variant"]):
        raise RuntimeError("selector and independent audit winners differ")
    variant_ids = {str(row["id"]) for row in spec["variants"]}
    baseline = str(spec["selection"]["baseline_variant"])
    if baseline not in variant_ids or winner not in variant_ids:
        raise RuntimeError("confirmation variant is absent from preregistration")
    if list(spec["confirmation"]["variants"]) != [
        "baseline", "selected_winner"
    ]:
        raise RuntimeError("confirmation roles changed from preregistration")
    return {
        "selection_path": selection_path,
        "audit_path": audit_path,
        "selection_sha256": _sha256(selection_path),
        "audit_sha256": _sha256(audit_path),
        "baseline": baseline,
        "selected_winner": winner,
    }


def _resolve_role(gate: dict, role: str) -> str:
    if role not in {"baseline", "selected_winner"}:
        raise ValueError(f"unknown confirmation role {role!r}")
    return str(gate[role])


def _identity(
    spec: dict,
    config_path: Path,
    gate: dict,
    task: str,
    role: str,
    variant: str,
    dataset: str,
    seed: int | None,
) -> dict:
    return {
        "experiment": spec["experiment"],
        "phase": "outer_confirmation",
        "config_sha256": _sha256(config_path),
        "development_selection_sha256": gate["selection_sha256"],
        "development_audit_sha256": gate["audit_sha256"],
        "task": task,
        "confirmation_role": role,
        "variant": variant,
        "selected_winner": gate["selected_winner"],
        "dataset": dataset,
        "family": spec["families"][dataset],
        "seed": seed,
        "train_ordinals": sorted(int(value) for value in spec["splits"]["train"]),
        "confirmation_ordinals": sorted(
            int(value) for value in spec["splits"]["confirmation"]
        ),
        "development_validation_opened": False,
        "confirmation_opened": True,
        "selection_performed_on_confirmation": False,
    }


def _existing(path: Path, identity: dict) -> Path | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if {key: payload.get(key) for key in identity} != identity:
        raise RuntimeError(f"stale confirmation identity in {path}")
    print(f"cached {path}", flush=True)
    return path


def _roles(spec: dict) -> list[str]:
    return [str(value) for value in spec["confirmation"]["variants"]]


def _decode_task1(spec: dict, job_index: int) -> tuple[str, str]:
    roles = _roles(spec)
    count = len(roles) * len(spec["datasets"])
    if not 0 <= int(job_index) < count:
        raise IndexError(f"Task1 confirmation index {job_index} outside [0,{count})")
    role_index, dataset_index = divmod(int(job_index), len(spec["datasets"]))
    return roles[role_index], str(spec["datasets"][dataset_index])


def _decode_seeded(
    spec: dict, task: str, job_index: int
) -> tuple[str, str, int]:
    roles = _roles(spec)
    seeds = [int(value) for value in spec[task]["confirmation_training_seeds"]]
    per_role = len(spec["datasets"]) * len(seeds)
    count = len(roles) * per_role
    if not 0 <= int(job_index) < count:
        raise IndexError(f"{task} confirmation index {job_index} outside [0,{count})")
    role_index, remainder = divmod(int(job_index), per_role)
    dataset_index, seed_index = divmod(remainder, len(seeds))
    return roles[role_index], str(spec["datasets"][dataset_index]), seeds[seed_index]


def run_task1(config_path: str | Path, job_index: int) -> Path:
    spec, path = _load_spec(config_path)
    gate = _gate(spec, path)
    role, dataset = _decode_task1(spec, job_index)
    variant_id = _resolve_role(gate, role)
    identity = _identity(
        spec, path, gate, "Task1", role, variant_id, dataset, None
    )
    target = (
        Path(spec["output_root"]) / "confirmation" / "task1" / role
        / f"{dataset}.json"
    )
    if _existing(target, identity):
        return target
    train = load_common_records(spec, variant_id, dataset, spec["splits"]["train"])
    test = load_common_records(
        spec, variant_id, dataset, spec["splits"]["confirmation"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = spec["task1"]
    rows = []
    for seed in settings["confirmation_kmeans_seeds"]:
        for arm, feature, pca_dim in (
            ("raw", "raw", int(settings["raw_pca_dim"])),
            ("fmt", str(settings["fmt_feature"]), int(settings["fmt_pca_dim"])),
        ):
            score = _fit_score(
                train, test, feature, pca_dim, int(seed),
                int(settings["kmeans_n_init"]), device,
            )
            rows.append({
                "arm": arm,
                "feature": feature,
                "pca_dim": pca_dim,
                "kmeans_seed": int(seed),
                **score,
            })
    result = {
        **identity,
        "pathline": _variant(spec, variant_id),
        "train_samples": int(sum(len(row["reference"]) for row in train)),
        "confirmation_samples": int(sum(len(row["reference"]) for row in test)),
        "rows": rows,
    }
    print(f"DONE confirmation Task1 {role}={variant_id}/{dataset}", flush=True)
    return _atomic_json(target, result)


def run_task2(config_path: str | Path, job_index: int) -> Path:
    spec, path = _load_spec(config_path)
    gate = _gate(spec, path)
    role, dataset, seed = _decode_seeded(spec, "task2", job_index)
    variant_id = _resolve_role(gate, role)
    identity = _identity(
        spec, path, gate, "Task2", role, variant_id, dataset, seed
    )
    target = (
        Path(spec["output_root"]) / "confirmation" / "task2" / role
        / dataset / f"seed{seed}.json"
    )
    if _existing(target, identity):
        return target
    train = load_common_records(spec, variant_id, dataset, spec["splits"]["train"])
    test = load_common_records(
        spec, variant_id, dataset, spec["splits"]["confirmation"]
    )
    reference = np.concatenate([row["reference"] for row in test])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = dict(spec["task2"]["architecture"])
    source = _task2_source(spec)
    rows = []
    for arm in ("raw", "fmt"):
        train_x, test_x = _prepare_inputs(
            train, test, arm, str(spec["task2"]["fmt_feature"]), device
        )
        train_mu, test_mu, losses = train_vae(
            train_x, test_x, settings, source, seed, device
        )
        score = _latent_score(
            train_mu, test_mu, reference,
            int(spec["task2"]["kmeans_seed"]),
            int(spec["task2"]["kmeans_n_init"]),
        )
        rows.append({
            "arm": arm,
            "method": "Raw+VAE" if arm == "raw" else "FMT+VAE",
            "feature": (
                "center_relative" if arm == "raw"
                else spec["task2"]["fmt_feature"]
            ),
            "architecture": settings["id"],
            **score,
            **losses,
        })
        del train_x, test_x, train_mu, test_mu
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        **identity,
        "pathline": _variant(spec, variant_id),
        "train_samples": int(sum(len(row["reference"]) for row in train)),
        "confirmation_samples": int(len(reference)),
        "rows": rows,
    }
    print(
        f"DONE confirmation Task2 {role}={variant_id}/{dataset}/seed{seed}",
        flush=True,
    )
    return _atomic_json(target, result)


def run_task3(config_path: str | Path, job_index: int) -> Path:
    spec, path = _load_spec(config_path)
    gate = _gate(spec, path)
    role, dataset, seed = _decode_seeded(spec, "task3", job_index)
    variant_id = _resolve_role(gate, role)
    identity = _identity(
        spec, path, gate, "Task3", role, variant_id, dataset, seed
    )
    job_root = (
        Path(spec["output_root"]) / "confirmation" / "task3" / role
        / dataset / f"seed{seed}"
    )
    target = job_root / "result.json"
    if _existing(target, identity):
        return target
    train_records = load_common_records(
        spec, variant_id, dataset, spec["splits"]["train"]
    )
    test_records = load_common_records(
        spec, variant_id, dataset, spec["splits"]["confirmation"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = str(spec["task3"]["fmt_feature"])
    train = _task3_split(train_records, feature, device)
    test = _task3_split(test_records, feature, device)
    train, test, _, stats = _normalize_train_only(train, test)
    fmt_dim = int(train[1].shape[1])

    backbone_root = job_root / "backbone"
    backbone_spec = _task3_backbone_spec(spec, backbone_root, seed)
    raw_row = train_raw_backbone(
        backbone_spec, dataset, "raw", seed,
        (train, test, None), stats, device, backbone_root,
    )
    raw_wide_row = train_raw_backbone(
        backbone_spec, dataset, "raw_wide", seed,
        (train, test, None), stats, device, backbone_root,
    )
    if int(raw_wide_row["parameter_count"]) != int(
        spec["task3"]["raw_wide_parameter_count"]
    ):
        raise RuntimeError("Task3 Raw-wide parameter count changed")
    checkpoint_dir = backbone_root / "checkpoints"
    rows = [
        {"arm": "raw_backbone", **raw_row},
        {"arm": "raw_wide_guard", **raw_wide_row},
    ]
    for source_name in ("raw_pca", "fmt"):
        output = job_root / source_name
        residual_spec = _task3_residual_spec(
            spec, output, checkpoint_dir, seed, source_name, fmt_dim
        )
        row = train_residual(
            residual_spec, dataset, seed, (train, test, None),
            stats, device, output,
        )
        rows.append({"arm": source_name, **row})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    capacities = {
        int(row["trainable_residual_parameter_count"])
        for row in rows if row["arm"] in {"raw_pca", "fmt"}
    }
    if len(capacities) != 1:
        raise RuntimeError(f"Task3 paired capacities differ: {capacities}")
    result = {
        **identity,
        "pathline": _variant(spec, variant_id),
        "fmt_feature": feature,
        "fmt_dim": fmt_dim,
        "train_samples": int(len(train[2])),
        "confirmation_samples": int(len(test[2])),
        "rows": rows,
    }
    print(
        f"DONE confirmation Task3 {role}={variant_id}/{dataset}/seed{seed}",
        flush=True,
    )
    return _atomic_json(target, result)


def _collect(spec: dict, task: str) -> list[dict]:
    root = Path(spec["output_root"]) / "confirmation" / task.lower()
    paths = sorted(
        root.glob("**/result.json") if task == "Task3" else root.glob("**/*.json")
    )
    multiplier = 1 if task == "Task1" else len(
        spec[task.lower()]["confirmation_training_seeds"]
    )
    expected = len(_roles(spec)) * len(spec["datasets"]) * multiplier
    if len(paths) != expected:
        raise RuntimeError(
            f"{task} confirmation expected {expected} files, found {len(paths)}"
        )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("task") != task for row in payloads):
        raise RuntimeError(f"{task} confirmation contains another task")
    return payloads


def _per_dataset(spec: dict, payloads: dict[str, list[dict]]) -> list[dict]:
    arm_names = {
        "Task1": ("raw", "fmt"),
        "Task2": ("raw", "fmt"),
        "Task3": ("raw_pca", "fmt"),
    }
    result = []
    for task in ("Task1", "Task2", "Task3"):
        for role in _roles(spec):
            for dataset in spec["datasets"]:
                matches = [
                    row for row in payloads[task]
                    if row["confirmation_role"] == role
                    and row["dataset"] == dataset
                ]
                names = arm_names[task]
                f1 = {name: [] for name in names}
                ap = {name: [] for name in names}
                for payload in matches:
                    for row in payload["rows"]:
                        if row["arm"] not in f1:
                            continue
                        key = "validation_f1" if task == "Task3" else "f1"
                        f1[row["arm"]].append(float(row[key]))
                        if task == "Task3":
                            ap[row["arm"]].append(
                                float(row["validation_average_precision"])
                            )
                if any(not f1[name] for name in names):
                    raise RuntimeError(
                        f"incomplete confirmation pair for {task}/{role}/{dataset}"
                    )
                raw_f1 = float(np.mean(f1[names[0]]))
                fmt_f1 = float(np.mean(f1[names[1]]))
                item = {
                    "task": task,
                    "confirmation_role": role,
                    "variant": matches[0]["variant"],
                    "dataset": dataset,
                    "family": spec["families"][dataset],
                    "raw_f1": raw_f1,
                    "fmt_f1": fmt_f1,
                    "f1_gain": fmt_f1 - raw_f1,
                }
                if task == "Task3":
                    raw_ap = float(np.mean(ap[names[0]]))
                    fmt_ap = float(np.mean(ap[names[1]]))
                    item.update({
                        "raw_ap": raw_ap,
                        "fmt_ap": fmt_ap,
                        "ap_gain": fmt_ap - raw_ap,
                    })
                result.append(item)
    return result


def _aggregate(spec: dict, per_dataset: list[dict], role: str) -> dict:
    tasks = {}
    for task in ("Task1", "Task2", "Task3"):
        values = [
            row for row in per_dataset
            if row["confirmation_role"] == role and row["task"] == task
        ]
        task_row = {
            "raw_f1": float(np.mean([row["raw_f1"] for row in values])),
            "fmt_f1": float(np.mean([row["fmt_f1"] for row in values])),
            "f1_gain": float(np.mean([row["f1_gain"] for row in values])),
            "positive_dataset_count": int(sum(row["f1_gain"] > 0 for row in values)),
            "dataset_count": len(values),
        }
        if task == "Task3":
            task_row["ap_gain"] = float(np.mean([row["ap_gain"] for row in values]))
        tasks[task] = task_row
    family_rows = []
    for task in ("Task1", "Task2", "Task3"):
        for family in sorted(set(spec["families"].values())):
            values = [
                row for row in per_dataset
                if row["confirmation_role"] == role
                and row["task"] == task and row["family"] == family
            ]
            family_rows.append({
                "task": task,
                "family": family,
                "raw_f1": float(np.mean([row["raw_f1"] for row in values])),
                "fmt_f1": float(np.mean([row["fmt_f1"] for row in values])),
                "f1_gain": float(np.mean([row["f1_gain"] for row in values])),
            })
    gains = [tasks[task]["f1_gain"] for task in ("Task1", "Task2", "Task3")]
    return {
        "confirmation_role": role,
        "variant": next(
            row["variant"] for row in per_dataset
            if row["confirmation_role"] == role
        ),
        "equal_task_mean_f1_gain": float(np.mean(gains)),
        "minimum_task_f1_gain": float(np.min(gains)),
        "mean_absolute_fmt_f1": float(np.mean([
            tasks[task]["fmt_f1"] for task in ("Task1", "Task2", "Task3")
        ])),
        "tasks": tasks,
        "per_family": family_rows,
    }


def summarize(config_path: str | Path) -> Path:
    spec, path = _load_spec(config_path)
    gate = _gate(spec, path)
    payloads = {
        task: _collect(spec, task) for task in ("Task1", "Task2", "Task3")
    }
    per_dataset = _per_dataset(spec, payloads)
    roles = {role: _aggregate(spec, per_dataset, role) for role in _roles(spec)}
    baseline = roles["baseline"]
    winner = roles["selected_winner"]
    result = {
        "experiment": spec["experiment"],
        "phase": "outer_confirmation",
        "status": "confirmation_complete_no_selection",
        "config_sha256": _sha256(path),
        "development_selection_sha256": gate["selection_sha256"],
        "development_audit_sha256": gate["audit_sha256"],
        "train_ordinals": sorted(spec["splits"]["train"]),
        "confirmation_ordinals": sorted(spec["splits"]["confirmation"]),
        "selection_performed_on_confirmation": False,
        "roles": roles,
        "winner_minus_baseline_equal_task_mean_f1_gain": (
            winner["equal_task_mean_f1_gain"]
            - baseline["equal_task_mean_f1_gain"]
        ),
        "per_dataset": per_dataset,
        "result_file_counts": {
            task: len(payloads[task]) for task in ("Task1", "Task2", "Task3")
        },
    }
    target = Path(spec["output_root"]) / "confirmation" / "summary.json"
    _atomic_json(target, result)
    print(
        "CONFIRMATION baseline="
        f"{baseline['equal_task_mean_f1_gain']:+.6f} winner="
        f"{winner['equal_task_mean_f1_gain']:+.6f} delta="
        f"{result['winner_minus_baseline_equal_task_mean_f1_gain']:+.6f}",
        flush=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Verify_Task123_FixedPathline_1.1.yaml"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=("task1", "task2", "task3", "summarize"),
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "summarize":
        summarize(args.config)
        return
    if args.job_index is None:
        parser.error(f"--job-index is required for --mode {args.mode}")
    {"task1": run_task1, "task2": run_task2, "task3": run_task3}[
        args.mode
    ](args.config, args.job_index)


if __name__ == "__main__":
    main()
