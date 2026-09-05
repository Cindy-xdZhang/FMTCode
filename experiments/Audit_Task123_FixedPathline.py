"""Independently audit the fixed-pathline Task1--Task3 development search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(left, right, label, tolerance=1e-10):
    if not np.isclose(float(left), float(right), rtol=0.0, atol=tolerance):
        raise RuntimeError(f"{label}: {left} != {right}")


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _paths(spec: dict, task: str) -> list[Path]:
    root = Path(spec["output_root"]) / "development" / task.lower()
    matches = sorted(root.glob("**/*.json"))
    seed_count = 1 if task == "Task1" else len(
        spec[task.lower()]["development_training_seeds"]
    )
    expected = len(spec["variants"]) * len(spec["datasets"]) * seed_count
    if len(matches) != expected:
        raise RuntimeError(
            f"{task}: expected {expected} JSON shards, found {len(matches)}"
        )
    return matches


def _read_task(spec: dict, config_sha: str, task: str) -> list[dict]:
    payloads = []
    expected_seeds = (
        {None} if task == "Task1" else
        {int(value) for value in spec[task.lower()]["development_training_seeds"]}
    )
    observed_keys = set()
    for path in _paths(spec, task):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, expected in (
            ("experiment", spec["experiment"]),
            ("config_sha256", config_sha),
            ("task", task),
            ("confirmation_opened", False),
        ):
            if payload.get(key) != expected:
                raise RuntimeError(
                    f"{path}: {key}={payload.get(key)!r}, expected {expected!r}"
                )
        if sorted(payload["opened_ordinals"]) != sorted(
            spec["selection"]["opened_ordinals"]
        ):
            raise RuntimeError(f"{path}: opened ordinal set changed")
        variant = str(payload["variant"])
        dataset = str(payload["dataset"])
        seed = payload.get("seed")
        if variant not in {str(row["id"]) for row in spec["variants"]}:
            raise RuntimeError(f"{path}: unknown variant {variant}")
        if dataset not in spec["datasets"]:
            raise RuntimeError(f"{path}: unknown dataset {dataset}")
        if seed not in expected_seeds:
            raise RuntimeError(f"{path}: unexpected seed {seed}")
        key = (variant, dataset, seed)
        if key in observed_keys:
            raise RuntimeError(f"duplicate {task} shard {key}")
        observed_keys.add(key)
        if int(payload["train_samples"]) <= 0 or int(payload["validation_samples"]) <= 0:
            raise RuntimeError(f"{path}: empty sample split")
        payloads.append(payload)

    expected_keys = {
        (str(variant["id"]), dataset, seed)
        for variant in spec["variants"]
        for dataset in spec["datasets"]
        for seed in expected_seeds
    }
    if observed_keys != expected_keys:
        raise RuntimeError(f"{task}: result-key coverage differs")
    return payloads


def _arm_metrics(payload: dict, task: str) -> dict[str, tuple[float, float | None]]:
    names = {
        "Task1": {"raw", "fmt"},
        "Task2": {"raw", "fmt"},
        "Task3": {"raw_pca", "fmt"},
    }[task]
    result: dict[str, list[tuple[float, float | None]]] = {
        name: [] for name in names
    }
    for row in payload["rows"]:
        arm = str(row["arm"])
        if arm not in names:
            continue
        if task == "Task3":
            f1 = float(row["validation_f1"])
            ap = float(row["validation_average_precision"])
        else:
            f1 = float(row["f1"])
            ap = None
        if not np.isfinite(f1) or ap is not None and not np.isfinite(ap):
            raise RuntimeError(
                f"non-finite {task} metric for {payload['variant']}/"
                f"{payload['dataset']}/{payload.get('seed')}"
            )
        result[arm].append((f1, ap))
    expected_per_arm = (
        len(payload["rows"]) // 2 if task == "Task1" else 1
    )
    if any(len(values) != expected_per_arm for values in result.values()):
        raise RuntimeError(
            f"incomplete {task} arms for {payload['variant']}/"
            f"{payload['dataset']}/{payload.get('seed')}: "
            f"{ {key: len(value) for key, value in result.items()} }"
        )
    if task == "Task3":
        rows = {str(row["arm"]): row for row in payload["rows"]}
        raw_count = int(rows["raw_pca"]["trainable_residual_parameter_count"])
        fmt_count = int(rows["fmt"]["trainable_residual_parameter_count"])
        if raw_count != fmt_count:
            raise RuntimeError(
                f"Task3 capacity mismatch for {payload['variant']}/"
                f"{payload['dataset']}/{payload['seed']}: "
                f"{raw_count} != {fmt_count}"
            )
    return {
        arm: (
            float(np.mean([value[0] for value in values])),
            None if values[0][1] is None else
            float(np.mean([value[1] for value in values])),
        )
        for arm, values in result.items()
    }


def _per_dataset(spec: dict, task_payloads: dict[str, list[dict]]) -> list[dict]:
    result = []
    for task in ("Task1", "Task2", "Task3"):
        raw_name = "raw_pca" if task == "Task3" else "raw"
        for variant in [str(row["id"]) for row in spec["variants"]]:
            for dataset in spec["datasets"]:
                payloads = [
                    row for row in task_payloads[task]
                    if row["variant"] == variant and row["dataset"] == dataset
                ]
                expected = 1 if task == "Task1" else len(
                    spec[task.lower()]["development_training_seeds"]
                )
                if len(payloads) != expected:
                    raise RuntimeError(
                        f"{task} {variant}/{dataset}: {len(payloads)} != {expected}"
                    )
                arm_rows = [_arm_metrics(row, task) for row in payloads]
                raw_f1 = float(np.mean([row[raw_name][0] for row in arm_rows]))
                fmt_f1 = float(np.mean([row["fmt"][0] for row in arm_rows]))
                item = {
                    "task": task, "variant": variant, "dataset": dataset,
                    "raw_f1": raw_f1, "fmt_f1": fmt_f1,
                    "f1_gain": fmt_f1 - raw_f1,
                }
                if task == "Task3":
                    raw_ap = float(np.mean([row[raw_name][1] for row in arm_rows]))
                    fmt_ap = float(np.mean([row["fmt"][1] for row in arm_rows]))
                    item.update({
                        "raw_ap": raw_ap, "fmt_ap": fmt_ap,
                        "ap_gain": fmt_ap - raw_ap,
                    })
                result.append(item)
    return result


def _leaderboard(spec: dict, per_dataset: list[dict]) -> list[dict]:
    result = []
    for variant in [str(row["id"]) for row in spec["variants"]]:
        tasks = {}
        for task in ("Task1", "Task2", "Task3"):
            rows = [row for row in per_dataset
                    if row["variant"] == variant and row["task"] == task]
            if len(rows) != len(spec["datasets"]):
                raise RuntimeError(f"incomplete audit rows for {variant}/{task}")
            tasks[task] = {
                "raw_f1": float(np.mean([row["raw_f1"] for row in rows])),
                "fmt_f1": float(np.mean([row["fmt_f1"] for row in rows])),
                "f1_gain": float(np.mean([row["f1_gain"] for row in rows])),
            }
            if task == "Task3":
                tasks[task]["ap_gain"] = float(np.mean([
                    row["ap_gain"] for row in rows
                ]))
        gains = [tasks[task]["f1_gain"] for task in ("Task1", "Task2", "Task3")]
        result.append({
            "variant": variant,
            "equal_task_mean_f1_gain": float(np.mean(gains)),
            "minimum_task_f1_gain": float(np.min(gains)),
            "task3_ap_gain": tasks["Task3"]["ap_gain"],
            "mean_absolute_fmt_f1": float(np.mean([
                tasks[task]["fmt_f1"] for task in ("Task1", "Task2", "Task3")
            ])),
            "tasks": tasks,
        })
    result.sort(key=lambda row: (
        row["equal_task_mean_f1_gain"], row["minimum_task_f1_gain"],
        row["task3_ap_gain"], row["mean_absolute_fmt_f1"], row["variant"],
    ), reverse=True)
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def audit(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = _sha256(config_path)
    selection_path = Path(spec["output_root"]) / "selection" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("config_sha256") != config_sha:
        raise RuntimeError("selector used another config identity")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("selector opened confirmation data")
    task_payloads = {
        task: _read_task(spec, config_sha, task)
        for task in ("Task1", "Task2", "Task3")
    }
    per_dataset = _per_dataset(spec, task_payloads)
    leaderboard = _leaderboard(spec, per_dataset)
    observed = selection["leaderboard"]
    if len(observed) != len(leaderboard):
        raise RuntimeError("leaderboard row count differs")
    scalar_keys = (
        "equal_task_mean_f1_gain", "minimum_task_f1_gain",
        "task3_ap_gain", "mean_absolute_fmt_f1",
    )
    for expected, actual in zip(leaderboard, observed):
        if expected["variant"] != actual["variant"] or expected["rank"] != actual["rank"]:
            raise RuntimeError("leaderboard order differs")
        for key in scalar_keys:
            _close(expected[key], actual[key], f"{expected['variant']}.{key}")
        for task in ("Task1", "Task2", "Task3"):
            for key, value in expected["tasks"][task].items():
                _close(value, actual["tasks"][task][key],
                       f"{expected['variant']}.{task}.{key}")
    if selection["winner"]["variant"] != leaderboard[0]["variant"]:
        raise RuntimeError("frozen winner differs from independent audit")
    baseline = next(
        row for row in leaderboard
        if row["variant"] == spec["selection"]["baseline_variant"]
    )
    payload = {
        "experiment": spec["experiment"],
        "status": "PASS",
        "config_sha256": config_sha,
        "selection_sha256": _sha256(selection_path),
        "confirmation_data_opened": False,
        "audited_shard_count": sum(len(value) for value in task_payloads.values()),
        "winner": leaderboard[0],
        "baseline": baseline,
        "winner_improvement_over_baseline_equal_task_mean_f1_gain": (
            leaderboard[0]["equal_task_mean_f1_gain"]
            - baseline["equal_task_mean_f1_gain"]
        ),
        "leaderboard": leaderboard,
        "per_dataset": per_dataset,
    }
    target = Path(spec["output_root"]) / "selection" / "independent_audit.json"
    _atomic_json(target, payload)
    print(
        f"PASS winner={leaderboard[0]['variant']} audited_shards="
        f"{payload['audited_shard_count']}", flush=True,
    )
    return target


def cleanup(config_path: str | Path) -> Path:
    """Delete only this experiment's temporary training checkpoints."""
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_path = Path(spec["output_root"]) / "selection" / "independent_audit.json"
    evidence = json.loads(audit_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS":
        raise RuntimeError("checkpoint cleanup requires a PASS independent audit")
    if evidence.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("checkpoint cleanup audit used another config")
    task3_root = Path(spec["output_root"]) / "development" / "task3"
    checkpoints = sorted(task3_root.glob("**/*.pt"))
    for path in checkpoints:
        path.unlink()
    remaining = sorted(task3_root.glob("**/*.pt"))
    if remaining:
        raise RuntimeError(f"checkpoint cleanup left {len(remaining)} files")
    payload = {
        "experiment": spec["experiment"],
        "status": "PASS",
        "config_sha256": _sha256(config_path),
        "independent_audit_sha256": _sha256(audit_path),
        "deleted_checkpoint_count": len(checkpoints),
        "remaining_checkpoint_count": 0,
    }
    target = Path(spec["output_root"]) / "selection" / "cleanup.json"
    _atomic_json(target, payload)
    print(f"PASS deleted {len(checkpoints)} temporary checkpoints", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Verify_Task123_FixedPathline_1.1.yaml"
    )
    parser.add_argument("--mode", choices=("audit", "cleanup"), default="audit")
    args = parser.parse_args()
    {"audit": audit, "cleanup": cleanup}[args.mode](args.config)


if __name__ == "__main__":
    main()
