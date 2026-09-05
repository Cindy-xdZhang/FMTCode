"""Independent audit for fixed-pathline Task1/Task2/Task3 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.audit.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _close(expected: float, actual: float, label: str) -> None:
    if not np.isclose(float(expected), float(actual), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def _gate(spec: dict, config_path: Path) -> dict:
    root = Path(spec["output_root"])
    selection_path = root / "selection" / "selection.json"
    audit_path = root / "selection" / "independent_audit.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    config_sha = _sha256(config_path)
    if selection.get("config_sha256") != config_sha:
        raise RuntimeError("selection config identity differs")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("development selector opened confirmation data")
    if audit.get("status") != "PASS" or audit.get("config_sha256") != config_sha:
        raise RuntimeError("development independent audit is not valid")
    if audit.get("selection_sha256") != _sha256(selection_path):
        raise RuntimeError("development audit did not cover selection")
    winner = str(selection["winner"]["variant"])
    if winner != str(audit["winner"]["variant"]):
        raise RuntimeError("development winner differs in audit")
    return {
        "selection_sha256": _sha256(selection_path),
        "audit_sha256": _sha256(audit_path),
        "baseline": str(spec["selection"]["baseline_variant"]),
        "selected_winner": winner,
    }


def _roles(spec: dict) -> list[str]:
    roles = [str(value) for value in spec["confirmation"]["variants"]]
    if roles != ["baseline", "selected_winner"]:
        raise RuntimeError("confirmation roles differ from preregistration")
    return roles


def _paths(spec: dict, task: str) -> list[Path]:
    root = Path(spec["output_root"]) / "confirmation" / task.lower()
    return sorted(
        root.glob("**/result.json") if task == "Task3" else root.glob("**/*.json")
    )


def _read_task(
    spec: dict, config_path: Path, gate: dict, task: str
) -> tuple[list[dict], list[Path]]:
    paths = _paths(spec, task)
    multiplier = 1 if task == "Task1" else len(
        spec[task.lower()]["confirmation_training_seeds"]
    )
    expected = len(_roles(spec)) * len(spec["datasets"]) * multiplier
    if len(paths) != expected:
        raise RuntimeError(f"{task}: expected {expected} files, found {len(paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected_keys = set()
    task1_seeds = {int(value) for value in spec["task1"]["confirmation_kmeans_seeds"]}
    training_seeds = {
        int(value) for value in spec[task.lower()].get(
            "confirmation_training_seeds", []
        )
    }
    for payload, source_path in zip(payloads, paths):
        role = str(payload.get("confirmation_role"))
        dataset = str(payload.get("dataset"))
        seed = payload.get("seed")
        variant = str(payload.get("variant"))
        if role not in _roles(spec) or dataset not in spec["datasets"]:
            raise RuntimeError(f"unexpected identity in {source_path}")
        if variant != str(gate[role]):
            raise RuntimeError(f"role/variant mismatch in {source_path}")
        required = {
            "experiment": spec["experiment"],
            "phase": "outer_confirmation",
            "config_sha256": _sha256(config_path),
            "development_selection_sha256": gate["selection_sha256"],
            "development_audit_sha256": gate["audit_sha256"],
            "task": task,
            "selected_winner": gate["selected_winner"],
            "family": spec["families"][dataset],
            "train_ordinals": sorted(spec["splits"]["train"]),
            "confirmation_ordinals": sorted(spec["splits"]["confirmation"]),
            "development_validation_opened": False,
            "confirmation_opened": True,
            "selection_performed_on_confirmation": False,
        }
        for key, value in required.items():
            if payload.get(key) != value:
                raise RuntimeError(f"{source_path}: {key} identity differs")
        key = (role, dataset, None if task == "Task1" else int(seed))
        if key in expected_keys:
            raise RuntimeError(f"duplicate confirmation shard {key}")
        expected_keys.add(key)
        arms = [str(row["arm"]) for row in payload["rows"]]
        if task == "Task1":
            if seed is not None:
                raise RuntimeError("Task1 payload-level seed must be null")
            observed_pairs = {
                (str(row["arm"]), int(row["kmeans_seed"]))
                for row in payload["rows"]
            }
            expected_pairs = {
                (arm, value) for arm in ("raw", "fmt") for value in task1_seeds
            }
            if observed_pairs != expected_pairs or len(payload["rows"]) != len(expected_pairs):
                raise RuntimeError(f"Task1 seed/arm coverage differs in {source_path}")
        elif task == "Task2":
            if int(seed) not in training_seeds or set(arms) != {"raw", "fmt"}:
                raise RuntimeError(f"Task2 seed/arm coverage differs in {source_path}")
            if len(arms) != 2:
                raise RuntimeError(f"Task2 duplicate arms in {source_path}")
        else:
            if int(seed) not in training_seeds:
                raise RuntimeError(f"Task3 seed differs in {source_path}")
            expected_arms = {"raw_backbone", "raw_wide_guard", "raw_pca", "fmt"}
            if set(arms) != expected_arms or len(arms) != 4:
                raise RuntimeError(f"Task3 arm coverage differs in {source_path}")
            paired = [row for row in payload["rows"] if row["arm"] in {"raw_pca", "fmt"}]
            trainable = {int(row["trainable_residual_parameter_count"]) for row in paired}
            total = {int(row["parameter_count"]) for row in paired}
            if len(trainable) != 1 or len(total) != 1:
                raise RuntimeError(f"Task3 paired capacities differ in {source_path}")
            guard = next(row for row in payload["rows"] if row["arm"] == "raw_wide_guard")
            if int(guard["parameter_count"]) != int(
                spec["task3"]["raw_wide_parameter_count"]
            ):
                raise RuntimeError(f"Task3 Raw-wide capacity differs in {source_path}")
    return payloads, paths


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
                        arm = row["arm"]
                        if arm not in f1:
                            continue
                        f1[arm].append(float(
                            row["validation_f1"] if task == "Task3" else row["f1"]
                        ))
                        if task == "Task3":
                            ap[arm].append(float(row["validation_average_precision"]))
                if any(not f1[name] for name in names):
                    raise RuntimeError(f"incomplete pair for {task}/{role}/{dataset}")
                raw_f1 = float(np.mean(f1[names[0]]))
                fmt_f1 = float(np.mean(f1[names[1]]))
                item = {
                    "task": task,
                    "confirmation_role": role,
                    "variant": str(matches[0]["variant"]),
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
        tasks[task] = {
            "raw_f1": float(np.mean([row["raw_f1"] for row in values])),
            "fmt_f1": float(np.mean([row["fmt_f1"] for row in values])),
            "f1_gain": float(np.mean([row["f1_gain"] for row in values])),
            "positive_dataset_count": int(sum(row["f1_gain"] > 0 for row in values)),
            "dataset_count": len(values),
        }
        if task == "Task3":
            tasks[task]["ap_gain"] = float(np.mean([row["ap_gain"] for row in values]))
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


def _compare_role(expected: dict, actual: dict, role: str) -> None:
    if expected["variant"] != actual["variant"]:
        raise RuntimeError(f"{role} variant differs in summary")
    for key in (
        "equal_task_mean_f1_gain", "minimum_task_f1_gain", "mean_absolute_fmt_f1"
    ):
        _close(expected[key], actual[key], f"{role}.{key}")
    for task in ("Task1", "Task2", "Task3"):
        for key, value in expected["tasks"][task].items():
            if isinstance(value, float):
                _close(value, actual["tasks"][task][key], f"{role}.{task}.{key}")
            elif value != actual["tasks"][task][key]:
                raise RuntimeError(f"{role}.{task}.{key} differs")


def audit(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = _gate(spec, config_path)
    task_payloads, result_paths = {}, []
    for task in ("Task1", "Task2", "Task3"):
        task_payloads[task], paths = _read_task(spec, config_path, gate, task)
        result_paths.extend(paths)
    per_dataset = _per_dataset(spec, task_payloads)
    roles = {role: _aggregate(spec, per_dataset, role) for role in _roles(spec)}

    summary_path = Path(spec["output_root"]) / "confirmation" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_summary = {
        "experiment": spec["experiment"],
        "phase": "outer_confirmation",
        "status": "confirmation_complete_no_selection",
        "config_sha256": _sha256(config_path),
        "development_selection_sha256": gate["selection_sha256"],
        "development_audit_sha256": gate["audit_sha256"],
        "train_ordinals": sorted(spec["splits"]["train"]),
        "confirmation_ordinals": sorted(spec["splits"]["confirmation"]),
        "selection_performed_on_confirmation": False,
    }
    for key, value in required_summary.items():
        if summary.get(key) != value:
            raise RuntimeError(f"confirmation summary {key} differs")
    for role in _roles(spec):
        _compare_role(roles[role], summary["roles"][role], role)
    delta = (
        roles["selected_winner"]["equal_task_mean_f1_gain"]
        - roles["baseline"]["equal_task_mean_f1_gain"]
    )
    _close(
        delta,
        summary["winner_minus_baseline_equal_task_mean_f1_gain"],
        "winner_minus_baseline_equal_task_mean_f1_gain",
    )
    manifest = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in sorted(result_paths)
    ]
    payload = {
        "experiment": spec["experiment"],
        "phase": "outer_confirmation",
        "status": "PASS",
        "config_sha256": _sha256(config_path),
        "development_selection_sha256": gate["selection_sha256"],
        "development_audit_sha256": gate["audit_sha256"],
        "confirmation_summary_sha256": _sha256(summary_path),
        "audited_result_count": len(result_paths),
        "selection_performed_on_confirmation": False,
        "roles": roles,
        "winner_minus_baseline_equal_task_mean_f1_gain": delta,
        "per_dataset": per_dataset,
        "result_manifest": manifest,
    }
    target = Path(spec["output_root"]) / "confirmation" / "independent_audit.json"
    _atomic_json(target, payload)
    print(
        f"PASS confirmation results={len(result_paths)} delta={delta:+.6f}",
        flush=True,
    )
    return target


def cleanup(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_path = Path(spec["output_root"]) / "confirmation" / "independent_audit.json"
    evidence = json.loads(audit_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS":
        raise RuntimeError("confirmation cleanup requires PASS audit")
    if evidence.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("confirmation audit used another config")
    root = Path(spec["output_root"]) / "confirmation" / "task3"
    checkpoints = sorted(root.glob("**/*.pt"))
    for checkpoint in checkpoints:
        checkpoint.unlink()
    remaining = sorted(root.glob("**/*.pt"))
    if remaining:
        raise RuntimeError(f"cleanup left {len(remaining)} checkpoints")
    payload = {
        "experiment": spec["experiment"],
        "phase": "outer_confirmation_cleanup",
        "status": "PASS",
        "config_sha256": _sha256(config_path),
        "confirmation_audit_sha256": _sha256(audit_path),
        "deleted_checkpoint_count": len(checkpoints),
        "remaining_checkpoint_count": 0,
    }
    target = Path(spec["output_root"]) / "confirmation" / "cleanup.json"
    _atomic_json(target, payload)
    print(f"PASS deleted {len(checkpoints)} confirmation checkpoints", flush=True)
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
