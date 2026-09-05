"""Independently audit Task2/Task3 global FMT selection and confirmation.

This module deliberately does not import either formal search implementation.
It reconstructs every candidate from the completed per-run CSV files, applies
the registered global ranking rule, and verifies the frozen selection JSON.
Selection mode opens no confirmation or outer-development cache. Confirmation
mode reads only final CSV/JSON evidence and never imports the formal runner or
summarizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


TOLERANCE = 1e-12


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise RuntimeError("cannot average an empty collection")
    return sum(values) / len(values)


def _require_close(actual, expected, label: str) -> None:
    if abs(float(actual) - float(expected)) > TOLERANCE:
        raise RuntimeError(
            f"{label} differs: recomputed={actual!r}, frozen={expected!r}"
        )


def _require_identity(row: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        if str(row.get(key, "")) != str(value):
            raise RuntimeError(
                f"{label}.{key} differs: observed={row.get(key)!r}, "
                f"expected={value!r}"
            )


def _task2_candidate(spec: dict, root: Path, feature: dict,
                     architecture: dict, evidence: set[Path]) -> dict:
    seeds = [int(value) for value in spec["screen_seeds"]]
    per_dataset = {}
    seed_gains = {seed: [] for seed in seeds}
    for dataset in spec["datasets"]:
        raw_path = root / "stage1" / "raw" / dataset / (
            f"{architecture['id']}.csv"
        )
        fmt_path = root / "stage1" / "fmt" / dataset / str(
            feature["id"]
        ) / f"{architecture['id']}.csv"
        evidence.update((raw_path, fmt_path))
        raw_rows = _read_csv(raw_path)
        fmt_rows = _read_csv(fmt_path)
        if len(raw_rows) != len(seeds) or len(fmt_rows) != len(seeds):
            raise RuntimeError(
                f"incomplete Task2 pair for {dataset}/{feature['id']}/"
                f"{architecture['id']}"
            )
        raw = {}
        fmt = {}
        for arm, rows, target in (
            ("raw", raw_rows, raw), ("fmt", fmt_rows, fmt)
        ):
            for row in rows:
                seed = int(row["training_seed"])
                if seed in target:
                    raise RuntimeError(
                        f"duplicate Task2 seed {dataset}/{arm}/{seed}"
                    )
                expected = {
                    "dataset": dataset,
                    "arm": arm,
                    "architecture": architecture["id"],
                }
                if arm == "fmt":
                    expected.update({
                        "feature_id": feature["id"],
                        "fmt_feature": feature["name"],
                    })
                _require_identity(row, expected, f"Task2 {dataset}/{arm}/{seed}")
                target[seed] = float(row["f1"])
        if set(raw) != set(seeds) or set(fmt) != set(seeds):
            raise RuntimeError(
                f"Task2 seed set changed for {dataset}/{feature['id']}/"
                f"{architecture['id']}"
            )
        raw_f1 = _mean(raw.values())
        fmt_f1 = _mean(fmt.values())
        gain = fmt_f1 - raw_f1
        per_dataset[dataset] = {
            "raw_f1": raw_f1,
            "fmt_f1": fmt_f1,
            "gain": gain,
        }
        for seed in seeds:
            seed_gains[seed].append(fmt[seed] - raw[seed])
    raw_macro = _mean(row["raw_f1"] for row in per_dataset.values())
    fmt_macro = _mean(row["fmt_f1"] for row in per_dataset.values())
    seed_macro = {
        str(seed): _mean(values) for seed, values in seed_gains.items()
    }
    gains = [row["gain"] for row in per_dataset.values()]
    return {
        "feature_id": str(feature["id"]),
        "fmt_feature": str(feature["name"]),
        "architecture": str(architecture["id"]),
        "raw_f1_macro": raw_macro,
        "fmt_f1_macro": fmt_macro,
        "fmt_minus_raw_f1_macro": fmt_macro - raw_macro,
        "worst_dataset_f1_gain": min(gains),
        "positive_dataset_count": sum(value > 0.0 for value in gains),
        "worst_seed_f1_gain": min(seed_macro.values()),
        "seed_gains": seed_macro,
        "datasets": per_dataset,
    }


def _task2_key(row: dict) -> tuple[float, ...]:
    return (
        float(row["fmt_minus_raw_f1_macro"]),
        float(row["worst_seed_f1_gain"]),
        float(row["fmt_f1_macro"]),
    )


def _audit_task2(spec: dict, config_path: Path) -> dict:
    root = Path(spec["output_root"])
    selection_path = root / "global_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if bool(selection.get("outer_ordinals_opened", True)):
        raise RuntimeError("Task2 selector opened outer ordinals")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("Task2 selector opened confirmation data")
    expected_opened = sorted(
        {int(value) for value in spec["splits"]["selection_train"]}
        | {int(value) for value in spec["splits"]["selection_validation"]}
    )
    if [int(value) for value in selection.get("opened_ordinals", [])] != expected_opened:
        raise RuntimeError("Task2 selector opened an unexpected ordinal set")

    evidence: set[Path] = {config_path, selection_path}
    candidates = [
        _task2_candidate(spec, root, feature, architecture, evidence)
        for feature in spec["features"]
        for architecture in spec["architectures"]
    ]
    ranked = sorted(candidates, key=_task2_key, reverse=True)
    winner = ranked[0]
    frozen = selection["winner"]
    for key in ("feature_id", "fmt_feature", "architecture"):
        if str(winner[key]) != str(frozen[key]):
            raise RuntimeError(f"Task2 global winner differs at {key}")
    for key in (
        "raw_f1_macro", "fmt_f1_macro", "fmt_minus_raw_f1_macro",
        "worst_dataset_f1_gain", "worst_seed_f1_gain",
    ):
        _require_close(winner[key], frozen[key], f"Task2 winner.{key}")
    if int(winner["positive_dataset_count"]) != int(
        frozen["positive_dataset_count"]
    ):
        raise RuntimeError("Task2 positive-dataset count differs")
    leaderboard_path = root / "global_leaderboard.csv"
    evidence.add(leaderboard_path)
    leaderboard = _read_csv(leaderboard_path)
    if len(leaderboard) != len(ranked):
        raise RuntimeError("Task2 global leaderboard row count differs")
    for expected, observed in zip(ranked, leaderboard):
        _require_identity(observed, {
            "feature_id": expected["feature_id"],
            "fmt_feature": expected["fmt_feature"],
            "architecture": expected["architecture"],
        }, "Task2 leaderboard")
        for key in (
            "raw_f1_macro", "fmt_f1_macro", "fmt_minus_raw_f1_macro",
            "worst_dataset_f1_gain", "worst_seed_f1_gain",
        ):
            _require_close(expected[key], observed[key], f"Task2 leaderboard.{key}")
    return {
        "task": "Task2",
        "candidate_count": len(ranked),
        "per_run_row_count": (
            len(spec["datasets"]) * len(spec["screen_seeds"])
            * (len(spec["features"]) * len(spec["architectures"])
               + len(spec["architectures"]))
        ),
        "winner": winner,
        "evidence": evidence,
    }


def _task3_candidate(spec: dict, root: Path, candidate: dict,
                     evidence: set[Path]) -> dict:
    seeds = [int(value) for value in spec["screen_seeds"]]
    metrics = ("f1", "average_precision")
    per_dataset = {}
    seed_gains = {seed: [] for seed in seeds}
    for dataset in spec["datasets"]:
        per_seed = {}
        for seed in seeds:
            arm_rows = {}
            counts = set()
            for source in ("fmt", "raw_pca"):
                path = (
                    root / "stage1" / "candidates" / str(candidate["id"])
                    / dataset / f"seed{seed}" / source / "per_run.csv"
                )
                evidence.add(path)
                rows = _read_csv(path)
                if len(rows) != 1:
                    raise RuntimeError(
                        f"Task3 result count differs for "
                        f"{candidate['id']}/{dataset}/{seed}/{source}"
                    )
                row = rows[0]
                _require_identity(row, {
                    "dataset": dataset,
                    "candidate_id": candidate["id"],
                    "fmt_feature": candidate["fmt_feature"],
                    "seed": seed,
                }, f"Task3 {candidate['id']}/{dataset}/{seed}/{source}")
                counts.add(int(row["trainable_residual_parameter_count"]))
                arm_rows[source] = {
                    metric: float(row[f"validation_{metric}"])
                    for metric in metrics
                }
            if len(counts) != 1:
                raise RuntimeError(
                    f"Task3 paired capacity differs for "
                    f"{candidate['id']}/{dataset}/{seed}"
                )
            per_seed[seed] = arm_rows
            seed_gains[seed].append(
                arm_rows["fmt"]["f1"] - arm_rows["raw_pca"]["f1"]
            )
        value = {}
        for source in ("fmt", "raw_pca"):
            value[source] = {
                metric: _mean(
                    per_seed[seed][source][metric] for seed in seeds
                )
                for metric in metrics
            }
        value["f1_gain"] = value["fmt"]["f1"] - value["raw_pca"]["f1"]
        value["ap_gain"] = (
            value["fmt"]["average_precision"]
            - value["raw_pca"]["average_precision"]
        )
        per_dataset[dataset] = value

    def macro(source: str, metric: str) -> float:
        return _mean(
            value[source][metric] for value in per_dataset.values()
        )

    fmt_f1 = macro("fmt", "f1")
    raw_f1 = macro("raw_pca", "f1")
    fmt_ap = macro("fmt", "average_precision")
    raw_ap = macro("raw_pca", "average_precision")
    seed_macro = {
        str(seed): _mean(values) for seed, values in seed_gains.items()
    }
    f1_gains = [value["f1_gain"] for value in per_dataset.values()]
    ap_gains = [value["ap_gain"] for value in per_dataset.values()]
    return {
        "candidate_id": str(candidate["id"]),
        "fmt_feature": str(candidate["fmt_feature"]),
        "fmt_f1_macro": fmt_f1,
        "raw_pca_f1_macro": raw_f1,
        "fmt_minus_raw_pca_f1_macro": fmt_f1 - raw_f1,
        "fmt_ap_macro": fmt_ap,
        "raw_pca_ap_macro": raw_ap,
        "fmt_minus_raw_pca_ap_macro": fmt_ap - raw_ap,
        "positive_f1_dataset_count": sum(value > 0.0 for value in f1_gains),
        "positive_ap_dataset_count": sum(value > 0.0 for value in ap_gains),
        "worst_dataset_f1_gain": min(f1_gains),
        "worst_dataset_ap_gain": min(ap_gains),
        "worst_seed_f1_gain": min(seed_macro.values()),
        "seed_gains": seed_macro,
        "datasets": per_dataset,
    }


def _task3_key(row: dict) -> tuple[float, ...]:
    return (
        float(row["fmt_f1_macro"]),
        float(row["fmt_minus_raw_pca_f1_macro"]),
        float(row["fmt_ap_macro"]),
        float(row["worst_dataset_f1_gain"]),
        float(row["worst_seed_f1_gain"]),
    )


def _audit_task3(spec: dict, config_path: Path) -> dict:
    root = Path(spec["output_root"])
    selection_path = root / "global_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if bool(selection.get("outer_ordinals_opened", True)):
        raise RuntimeError("Task3 selector opened outer ordinals")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("Task3 selector opened confirmation data")
    expected_opened = sorted(
        {int(value) for value in spec["screen_split"]["train_ordinals"]}
        | {int(value) for value in spec["screen_split"]["validation_ordinals"]}
    )
    if [int(value) for value in selection.get("opened_ordinals", [])] != expected_opened:
        raise RuntimeError("Task3 selector opened an unexpected ordinal set")

    evidence: set[Path] = {config_path, selection_path}
    candidates = [
        _task3_candidate(spec, root, candidate, evidence)
        for candidate in spec["candidates"]
    ]
    minimum_gain = float(spec["selection"].get("minimum_macro_f1_gain", 0.0))
    eligible = [
        row for row in candidates
        if row["fmt_minus_raw_pca_f1_macro"] >= minimum_gain
    ]
    if not eligible:
        raise RuntimeError("independent audit found no eligible Task3 candidate")
    ranked = sorted(eligible, key=_task3_key, reverse=True)
    ineligible = sorted(
        [row for row in candidates if row not in eligible],
        key=_task3_key,
        reverse=True,
    )
    ordered = [*ranked, *ineligible]
    winner = ranked[0]
    frozen = selection["winner"]
    for key in ("candidate_id", "fmt_feature"):
        if str(winner[key]) != str(frozen[key]):
            raise RuntimeError(f"Task3 global winner differs at {key}")
    metric_keys = (
        "fmt_f1_macro", "raw_pca_f1_macro",
        "fmt_minus_raw_pca_f1_macro", "fmt_ap_macro", "raw_pca_ap_macro",
        "fmt_minus_raw_pca_ap_macro", "worst_dataset_f1_gain",
        "worst_dataset_ap_gain", "worst_seed_f1_gain",
    )
    for key in metric_keys:
        _require_close(winner[key], frozen[key], f"Task3 winner.{key}")
    leaderboard_path = root / "global_leaderboard.csv"
    evidence.add(leaderboard_path)
    leaderboard = _read_csv(leaderboard_path)
    if len(leaderboard) != len(ordered):
        raise RuntimeError("Task3 global leaderboard row count differs")
    for expected, observed in zip(ordered, leaderboard):
        _require_identity(observed, {
            "candidate_id": expected["candidate_id"],
            "fmt_feature": expected["fmt_feature"],
        }, "Task3 leaderboard")
        for key in metric_keys:
            _require_close(expected[key], observed[key], f"Task3 leaderboard.{key}")
    return {
        "task": "Task3",
        "candidate_count": len(ordered),
        "eligible_candidate_count": len(ranked),
        "per_run_row_count": (
            len(spec["datasets"]) * len(spec["screen_seeds"])
            * len(spec["candidates"]) * 2
        ),
        "winner": winner,
        "evidence": evidence,
    }


def audit(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = str(spec.get("task", ""))
    if task == "Task2":
        result = _audit_task2(spec, config_path)
    elif task == "Task3":
        result = _audit_task3(spec, config_path)
    else:
        raise ValueError("uniform selection audit supports Task2 or Task3")
    evidence = sorted(result.pop("evidence"))
    payload = {
        "schema": 1,
        "status": "PASS",
        "experiment": str(spec["experiment"]),
        "task": task,
        "audit_is_independent_of_formal_selector": True,
        "selection_scope": "one unchanged recipe across all datasets",
        "outer_or_confirmation_data_opened": False,
        "config_sha256": _sha256(config_path),
        "evidence_file_count": len(evidence),
        "evidence_sha256": {
            path.as_posix(): _sha256(path) for path in evidence
        },
        **result,
    }
    target = Path(spec["output_root"]) / "independent_selection_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


def _family(source: dict, dataset: str) -> str:
    matches = [
        name for name, group in source["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"dataset {dataset} has {len(matches)} families")
    return matches[0]


def _confirmation_metric(row: dict, task: str, metric: str) -> float:
    key = f"confirmation_{metric}" if task == "Task2" else f"test_{metric}"
    return float(row[key])


def audit_confirmation(config_path: str | Path) -> Path:
    """Rebuild the frozen confirmation table without importing its runner."""
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "experiment", "task", "selection_config", "selection_path",
        "selection_audit_path", "output_root", "splits", "training_seeds",
        "expected_slices", "checkpoint_policy",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"confirmation audit misses config keys: {missing}")
    task = str(spec["task"])
    if task not in {"Task2", "Task3"}:
        raise ValueError("confirmation audit supports Task2 or Task3")
    source_config = Path(spec["selection_config"])
    selection_path = Path(spec["selection_path"])
    selection_audit_path = Path(spec["selection_audit_path"])
    manifest_path = Path(spec["output_root"]) / "frozen_recipe_manifest.json"
    per_run_path = Path(spec["output_root"]) / "confirmation" / "per_run.csv"
    table_path = Path(spec["output_root"]) / "confirmation" / "paper_table.csv"
    summary_path = Path(spec["output_root"]) / "confirmation" / "summary.json"
    evidence = {
        config_path, source_config, selection_path, selection_audit_path,
        manifest_path, per_run_path, table_path, summary_path,
    }
    for path in evidence:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_audit = json.loads(
        selection_audit_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if selection_audit.get("status") != "PASS":
        raise RuntimeError("selection audit did not pass")
    if bool(selection.get("outer_ordinals_opened", True)):
        raise RuntimeError("selector opened outer ordinals")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("selector opened confirmation data")
    identities = {
        "confirmation_config_sha256": _sha256(config_path),
        "selection_config_sha256": _sha256(source_config),
        "selection_sha256": _sha256(selection_path),
        "selection_audit_sha256": _sha256(selection_audit_path),
    }
    for key, expected in identities.items():
        if str(manifest.get(key, "")).lower() != expected:
            raise RuntimeError(f"confirmation manifest differs at {key}")
    if manifest.get("winner") != selection.get("winner"):
        raise RuntimeError("confirmation winner differs from frozen selection")
    if manifest.get("splits") != spec.get("splits"):
        raise RuntimeError("confirmation split differs from frozen manifest")
    seeds = [int(value) for value in spec["training_seeds"]]
    if manifest.get("training_seeds") != seeds:
        raise RuntimeError("confirmation seeds differ from frozen manifest")
    rows = _read_csv(per_run_path)
    expected_rows = len(source["datasets"]) * len(seeds) * 2
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"confirmation expected {expected_rows} rows, found {len(rows)}"
        )
    manifest_hash = _sha256(manifest_path)
    if {row["recipe_manifest_sha256"] for row in rows} != {manifest_hash}:
        raise RuntimeError("confirmation rows mix recipe manifests")
    winner = selection["winner"]
    arm_names = ("raw", "fmt") if task == "Task2" else ("raw_pca", "fmt")
    arm_key = "arm" if task == "Task2" else "source"
    seed_key = "training_seed" if task == "Task2" else "seed"
    dataset_details = {}
    seed_f1_gains = {seed: [] for seed in seeds}
    for dataset in source["datasets"]:
        subset = [row for row in rows if row["dataset"] == dataset]
        if len(subset) != len(seeds) * 2:
            raise RuntimeError(f"{dataset} confirmation row count differs")
        arms = {}
        for arm in arm_names:
            arm_rows = [row for row in subset if row[arm_key] == arm]
            observed_seeds = [int(row[seed_key]) for row in arm_rows]
            if sorted(observed_seeds) != sorted(seeds):
                raise RuntimeError(f"{dataset}/{arm} seed set differs")
            arms[arm] = arm_rows
        if task == "Task2":
            for row in subset:
                _require_identity(row, {
                    "fmt_feature": winner["fmt_feature"],
                    "feature_id": winner["feature_id"],
                    "architecture": winner["architecture"],
                }, f"Task2 confirmation {dataset}")
        else:
            for row in subset:
                _require_identity(row, {
                    "candidate_id": winner["candidate_id"],
                    "fmt_feature": winner["fmt_feature"],
                }, f"Task3 confirmation {dataset}")
            for seed in seeds:
                counts = {
                    int(next(
                        row["trainable_residual_parameter_count"]
                        for row in arms[arm] if int(row[seed_key]) == seed
                    )) for arm in arm_names
                }
                if len(counts) != 1:
                    raise RuntimeError(
                        f"Task3 capacity differs for {dataset}/seed{seed}"
                    )
        metrics = ("f1",) if task == "Task2" else (
            "f1", "average_precision"
        )
        detail = {"family": _family(source, dataset)}
        for metric in metrics:
            values = {}
            for arm in arm_names:
                values[arm] = [
                    _confirmation_metric(row, task, metric)
                    for row in arms[arm]
                ]
                detail[f"{arm}_{metric}"] = _mean(values[arm])
                mean = detail[f"{arm}_{metric}"]
                detail[f"{arm}_{metric}_std"] = (
                    _mean([(value - mean) ** 2 for value in values[arm]])
                    ** 0.5
                )
            detail[f"fmt_minus_{arm_names[0]}_{metric}"] = (
                detail[f"fmt_{metric}"] - detail[f"{arm_names[0]}_{metric}"]
            )
        for seed in seeds:
            raw = next(
                row for row in arms[arm_names[0]]
                if int(row[seed_key]) == seed
            )
            fmt = next(
                row for row in arms["fmt"] if int(row[seed_key]) == seed
            )
            seed_f1_gains[seed].append(
                _confirmation_metric(fmt, task, "f1")
                - _confirmation_metric(raw, task, "f1")
            )
        dataset_details[dataset] = detail

    table_rows = _read_csv(table_path)
    if len(table_rows) != len(source["datasets"]):
        raise RuntimeError("confirmation paper table row count differs")
    table_by_dataset = {row["dataset"]: row for row in table_rows}
    if set(table_by_dataset) != set(source["datasets"]):
        raise RuntimeError("confirmation paper table dataset set differs")
    for dataset, expected in dataset_details.items():
        observed = table_by_dataset[dataset]
        _require_identity(
            observed, {"family": expected["family"]},
            f"confirmation table {dataset}",
        )
        for key, value in expected.items():
            if key != "family":
                _require_close(value, observed[key], f"table {dataset}.{key}")

    raw_f1 = _mean(
        detail[f"{arm_names[0]}_f1"] for detail in dataset_details.values()
    )
    fmt_f1 = _mean(detail["fmt_f1"] for detail in dataset_details.values())
    seed_macro = {
        str(seed): _mean(values) for seed, values in seed_f1_gains.items()
    }
    families = sorted({detail["family"] for detail in dataset_details.values()})
    family_gains = {
        family: _mean(
            detail[f"fmt_minus_{arm_names[0]}_f1"]
            for detail in dataset_details.values()
            if detail["family"] == family
        ) for family in families
    }
    expected_summary = {
        "record_count": expected_rows,
        "dataset_count": len(source["datasets"]),
        "training_seed_count": len(seeds),
        "raw_f1_dataset_macro": raw_f1,
        "fmt_f1_dataset_macro": fmt_f1,
        "fmt_minus_raw_f1_dataset_macro": fmt_f1 - raw_f1,
        "positive_f1_dataset_count": sum(
            detail[f"fmt_minus_{arm_names[0]}_f1"] > 0.0
            for detail in dataset_details.values()
        ),
        "positive_f1_family_count": sum(
            value > 0.0 for value in family_gains.values()
        ),
        "positive_f1_seed_count": sum(
            value > 0.0 for value in seed_macro.values()
        ),
    }
    if task == "Task3":
        raw_ap = _mean(
            detail[f"{arm_names[0]}_average_precision"]
            for detail in dataset_details.values()
        )
        fmt_ap = _mean(
            detail["fmt_average_precision"]
            for detail in dataset_details.values()
        )
        expected_summary.update({
            "raw_average_precision_dataset_macro": raw_ap,
            "fmt_average_precision_dataset_macro": fmt_ap,
            "fmt_minus_raw_average_precision_dataset_macro": fmt_ap - raw_ap,
            "positive_average_precision_dataset_count": sum(
                detail[
                    f"fmt_minus_{arm_names[0]}_average_precision"
                ] > 0.0 for detail in dataset_details.values()
            ),
        })
    for key, value in expected_summary.items():
        if isinstance(value, int):
            if int(summary[key]) != value:
                raise RuntimeError(f"summary.{key} differs")
        else:
            _require_close(value, summary[key], f"summary.{key}")
    frozen_datasets = dict(summary.get("datasets", {}))
    if set(frozen_datasets) != set(dataset_details):
        raise RuntimeError("summary dataset set differs")
    for dataset, expected in dataset_details.items():
        observed = frozen_datasets[dataset]
        if observed.get("family") != expected["family"]:
            raise RuntimeError(f"summary {dataset}.family differs")
        for key, value in expected.items():
            if key != "family":
                _require_close(
                    value, observed[key], f"summary {dataset}.{key}"
                )
    frozen_family = dict(summary.get("family_f1_gains", {}))
    if set(frozen_family) != set(family_gains):
        raise RuntimeError("summary family gain keys differ")
    for family, value in family_gains.items():
        _require_close(
            value, frozen_family[family], f"summary family {family}"
        )
    frozen_seed = dict(summary.get("seed_f1_gains", {}))
    if set(frozen_seed) != set(seed_macro):
        raise RuntimeError("summary seed gain keys differ")
    for seed, value in seed_macro.items():
        _require_close(value, frozen_seed[seed], f"summary seed {seed}")
    if str(summary.get("per_run_sha256", "")).lower() != _sha256(per_run_path):
        raise RuntimeError("summary per-run hash differs")
    if str(summary.get("paper_table_sha256", "")).lower() != _sha256(table_path):
        raise RuntimeError("summary paper-table hash differs")
    temporary_checkpoints = list(
        (Path(spec["output_root"]) / "temporary_training").rglob("*.pt")
    )
    if task == "Task2" and temporary_checkpoints:
        raise RuntimeError("Task2 confirmation unexpectedly wrote checkpoints")
    payload = {
        "schema": 1,
        "status": "PASS",
        "experiment": spec["experiment"],
        "task": task,
        "audit_is_independent_of_formal_summarizer": True,
        "selection_scope": "one unchanged recipe across all datasets",
        "record_count": expected_rows,
        "dataset_count": len(source["datasets"]),
        "seed_count": len(seeds),
        "recomputed": expected_summary,
        "temporary_checkpoint_count_before_cleanup": len(temporary_checkpoints),
        "checkpoint_policy": spec["checkpoint_policy"],
        "evidence_sha256": {
            path.as_posix(): _sha256(path) for path in sorted(evidence)
        },
    }
    target = (
        Path(spec["output_root"]) / "confirmation"
        / "independent_confirmation_audit.json"
    )
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--confirmation-config")
    args = parser.parse_args()
    if args.confirmation_config:
        audit_confirmation(args.confirmation_config)
    else:
        audit(args.config)


if __name__ == "__main__":
    main()
