"""Independently audit strong baselines, FMT ablations, and noise tests.

This script intentionally does not import any formal experiment runner or
summarizer.  It reconstructs record capacities and dataset-macro statistics
directly from the frozen YAML contracts and per-run CSV evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml


def _load_yaml(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty evidence CSV: {path}")
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_count(name: str, rows: list[dict], expected: int) -> None:
    if len(rows) != int(expected):
        raise RuntimeError(f"{name}: found {len(rows)} rows, expected {expected}")


def _assert_unique(name: str, rows: list[dict], fields: tuple[str, ...]) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{name}: duplicate keys for {fields}")


def _assert_metric(rows: list[dict], field: str) -> None:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    # F1/AP are probabilities in [0, 1].  Adjusted Rand Index is chance
    # corrected and can legitimately be negative, with range [-1, 1].
    lower = -1.0 if field == "ari" else 0.0
    if not np.isfinite(values).all() or np.any(values < lower) or np.any(values > 1.0):
        raise RuntimeError(f"invalid {field} values")


def _dataset_macro(rows: list[dict], arm_field: str, arm: str, metric: str) -> float:
    selected = [row for row in rows if row[arm_field] == arm]
    if not selected:
        raise RuntimeError(f"no rows for {arm_field}={arm}")
    means = []
    for dataset in sorted({row["dataset"] for row in selected}):
        values = [
            float(row[metric]) for row in selected if row["dataset"] == dataset
        ]
        means.append(float(np.mean(values)))
    return float(np.mean(means))


def _close(name: str, observed: float, expected: float, tolerance: float = 1e-10) -> None:
    if not np.isclose(float(observed), float(expected), rtol=0.0, atol=tolerance):
        raise RuntimeError(f"{name}: {observed} != {expected} (tol={tolerance})")


def _task1_families(spec: dict) -> list[str]:
    task = spec["task1"]
    families = [
        str(value) for value in task.get(
            "baseline_families", ("time_domain", "plain_fourier")
        )
    ]
    declared = {str(row["family"]) for row in task["representations"]}
    if len(set(families)) != len(families) or declared != set(families):
        raise RuntimeError("task1 baseline families disagree with representations")
    return families


def _replay_tolerance(spec: dict) -> float:
    """Recipe-drift tolerance for replaying a main result with new seeds."""
    value = float(spec.get("replay_tolerance", 0.02))
    if not 0.0 < value <= 0.05:
        raise RuntimeError(f"replay_tolerance {value} outside (0, 0.05]")
    return value


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.as_posix(): _sha256(path)
        for path in sorted(set(path.resolve() for path in paths), key=str)
    }


def _write_audit(spec: dict, report: dict, evidence: list[Path]) -> Path:
    report = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": "PASS",
        "audit_is_independent_of_formal_summarizer": True,
        "evidence_sha256": _hashes(evidence),
        **report,
    }
    target = Path(spec["output_root"]) / "independent_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return target


def audit_strong(config_path: str | Path) -> Path:
    spec = _load_yaml(config_path)
    root = Path(spec["output_root"])
    source1 = _load_yaml(spec["task1"]["source_config"])
    source2 = _load_yaml(spec["task2"]["source_config"])
    source3 = _load_yaml(spec["task3"]["source_config"])
    datasets = list(source1["datasets"])
    if datasets != list(source2["datasets"]) or datasets != list(source3["datasets"]):
        raise RuntimeError("strong-baseline task dataset orders differ")

    selection_path = root / "task1" / "development_selection.csv"
    selection = _read_csv(selection_path)
    _assert_count(
        "Task1 development selection", selection,
        len(datasets) * len(spec["task1"]["representations"])
        * len(spec["task1"]["pca_dims"]),
    )
    _assert_unique("Task1 development selection", selection,
                   ("dataset", "baseline_id", "pca_dim"))
    for field in ("f1", "ari"):
        _assert_metric(selection, field)
    independent_winners = {}
    families = _task1_families(spec)
    for family in families:
        candidates = sorted({
            (row["baseline_id"], row["representation"], row["pca_dim"])
            for row in selection if row["baseline_family"] == family
        })
        ranked = []
        for baseline_id, representation, pca_dim in candidates:
            rows = [
                row for row in selection
                if row["baseline_id"] == baseline_id and row["pca_dim"] == pca_dim
            ]
            f1 = np.asarray([float(row["f1"]) for row in rows])
            ari = np.asarray([float(row["ari"]) for row in rows])
            ranked.append({
                "baseline_family": family, "baseline_id": baseline_id,
                "representation": representation, "pca_dim": pca_dim,
                "dataset_macro_f1": float(f1.mean()),
                "worst_dataset_f1": float(f1.min()),
                "dataset_macro_ari": float(ari.mean()),
            })
        ranked.sort(key=lambda row: (
            row["dataset_macro_f1"], row["worst_dataset_f1"],
            row["dataset_macro_ari"], row["baseline_id"], str(row["pca_dim"]),
        ), reverse=True)
        independent_winners[family] = ranked[0]
    frozen_path = root / "task1" / "frozen_baselines.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for family, winner in independent_winners.items():
        for key, value in winner.items():
            if isinstance(value, float):
                _close(f"Task1 frozen {family}/{key}", frozen["winners"][family][key], value)
            elif frozen["winners"][family][key] != value:
                raise RuntimeError(f"Task1 frozen winner mismatch for {family}/{key}")

    task1_path = root / "task1" / "confirmation_runs.csv"
    task1 = _read_csv(task1_path)
    _assert_count("Task1 confirmation", task1,
                  len(datasets) * len(families)
                  * len(spec["task1"]["final_kmeans_seeds"]))
    _assert_unique("Task1 confirmation", task1,
                   ("dataset", "baseline_family", "kmeans_seed"))
    _assert_metric(task1, "f1")

    task2, task2_paths = [], []
    for dataset in datasets:
        for arm in spec["task2"]["arms"]:
            path = root / "task2" / "shards" / f"{dataset}_{arm['id']}.csv"
            task2_paths.append(path); task2.extend(_read_csv(path))
    _assert_count("Task2 confirmation", task2,
                  len(datasets) * len(spec["task2"]["arms"])
                  * len(spec["task2"]["training_seeds"]))
    _assert_unique("Task2 confirmation", task2, ("dataset", "arm", "training_seed"))
    _assert_metric(task2, "f1")

    task3, task3_paths = [], []
    for dataset in datasets:
        path = root / "task3" / "shards" / f"{dataset}.csv"
        task3_paths.append(path); task3.extend(_read_csv(path))
    _assert_count("Task3 confirmation", task3,
                  len(datasets) * len(spec["task3"]["baselines"])
                  * len(spec["task3"]["seeds"]))
    _assert_unique("Task3 confirmation", task3, ("dataset", "baseline", "seed"))
    for field in ("f1", "average_precision"):
        _assert_metric(task3, field)

    macro = {
        **{
            f"task1_{family}_f1": _dataset_macro(task1, "baseline_family", family, "f1")
            for family in families
        },
        **{
            f"task2_{arm['id']}_f1": _dataset_macro(task2, "arm", arm["id"], "f1")
            for arm in spec["task2"]["arms"]
        },
        "task3_raw_f1": _dataset_macro(task3, "baseline", "raw", "f1"),
        "task3_raw_wide_f1": _dataset_macro(task3, "baseline", "raw_wide", "f1"),
        "task3_raw_ap": _dataset_macro(task3, "baseline", "raw", "average_precision"),
        "task3_raw_wide_ap": _dataset_macro(task3, "baseline", "raw_wide", "average_precision"),
    }
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, value in macro.items():
        _close(f"strong summary {key}", summary["dataset_macro"][key], value)
    return _write_audit(spec, {
        "record_counts": {"Task1": len(task1), "Task2": len(task2), "Task3": len(task3)},
        "task1_independently_reconstructed_winners": independent_winners,
        "independently_reconstructed_dataset_macro": macro,
    }, [Path(config_path), selection_path, frozen_path, task1_path,
        *task2_paths, *task3_paths, summary_path])


def _collect_ablation(spec: dict):
    root = Path(spec["output_root"])
    source1 = _load_yaml(spec["task1"]["source_config"])
    source2 = _load_yaml(spec["task2"]["source_config"])
    source3 = _load_yaml(spec["task3"]["source_config"])
    paths = [root / "task1" / "per_run.csv"]
    task1 = _read_csv(paths[0])
    task2 = []
    for dataset in source2["datasets"]:
        for variant in spec["canonical_variants"]:
            path = root / "task2" / "shards" / f"{dataset}_{variant['id']}.csv"
            paths.append(path); task2.extend(_read_csv(path))
    task3 = []
    for dataset in source3["datasets"]:
        for variant in spec["task3"]["variants"]:
            path = root / "task3" / "shards" / f"{dataset}_{variant['id']}.csv"
            paths.append(path); task3.extend(_read_csv(path))
    expected = {
        "Task1": len(source1["datasets"]) * len(spec["canonical_variants"])
        * len(spec["task1"]["kmeans_seeds"]),
        "Task2": len(source2["datasets"]) * len(spec["canonical_variants"])
        * len(spec["task2"]["training_seeds"]),
        "Task3": len(source3["datasets"]) * len(spec["task3"]["variants"])
        * len(spec["task3"]["seeds"]) * len(spec["task3"]["paired_sources"]),
    }
    for name, rows in (("Task1", task1), ("Task2", task2), ("Task3", task3)):
        _assert_count(name, rows, expected[name])
    _assert_unique("Task1", task1, ("dataset", "variant", "seed"))
    _assert_unique("Task2", task2, ("dataset", "variant", "seed"))
    _assert_unique("Task3", task3,
                   ("dataset", "component_variant", "source", "seed"))
    _assert_metric(task1, "f1"); _assert_metric(task2, "f1")
    _assert_metric(task3, "test_f1"); _assert_metric(task3, "test_average_precision")
    for dataset in source3["datasets"]:
        for variant in spec["task3"]["variants"]:
            for seed in spec["task3"]["seeds"]:
                pair = [
                    row for row in task3 if row["dataset"] == dataset
                    and row["component_variant"] == variant["id"]
                    and int(row["seed"]) == int(seed)
                ]
                capacities = {int(row["trainable_residual_parameter_count"]) for row in pair}
                if len(pair) != 2 or len(capacities) != 1:
                    raise RuntimeError(f"Task3 capacity mismatch: {dataset}/{variant['id']}/{seed}")
    return (task1, task2, task3), paths, expected


def audit_ablation(config_path: str | Path) -> Path:
    spec = _load_yaml(config_path)
    (task1, task2, task3), paths, expected = _collect_ablation(spec)
    reconstructed = {}
    for task_name, rows, variant_field, source_field, metric in (
        ("Task1", task1, "variant", None, "f1"),
        ("Task2", task2, "variant", None, "f1"),
        ("Task3", task3, "component_variant", "source", "test_f1"),
    ):
        keys = sorted({
            (row[variant_field], "" if source_field is None else row[source_field])
            for row in rows
        })
        for variant, source in keys:
            selected = [row for row in rows if row[variant_field] == variant
                        and (source_field is None or row[source_field] == source)]
            key = f"{task_name}|{variant}|{source}"
            reconstructed[key] = {
                "dataset_macro_f1": _dataset_macro(
                    [{**row, "audit_arm": key} for row in selected],
                    "audit_arm", key, metric,
                )
            }
            if task_name == "Task3":
                reconstructed[key]["dataset_macro_average_precision"] = _dataset_macro(
                    [{**row, "audit_arm": key} for row in selected],
                    "audit_arm", key, "test_average_precision",
                )
    summary_path = Path(spec["output_root"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    formal = {
        f"{row['task']}|{row['variant']}|{row['source']}": row
        for row in summary["rows"]
    }
    if set(formal) != set(reconstructed):
        raise RuntimeError("ablation summary keys differ from independent reconstruction")
    for key, metrics in reconstructed.items():
        for metric, value in metrics.items():
            _close(f"ablation {key}/{metric}", formal[key][metric], value)

    main1 = json.loads(Path(spec["task1"]["main_summary"]).read_text(encoding="utf-8"))
    main2 = json.loads(Path(spec["task2"]["main_summary"]).read_text(encoding="utf-8"))
    main3 = json.loads(Path(spec["task3"]["main_summary"]).read_text(encoding="utf-8"))
    full_checks = {
        "Task1 full": (reconstructed["Task1|full|"]["dataset_macro_f1"],
                       main1["dataset_macro"]["fmt_f1"]),
        "Task2 full": (reconstructed["Task2|full|"]["dataset_macro_f1"],
                       main2["fmt_f1_dataset_macro"]),
        "Task3 dft fmt F1": (reconstructed["Task3|dft|fmt"]["dataset_macro_f1"],
                             main3["fmt_f1_dataset_macro"]),
        "Task3 dft raw-pca F1": (reconstructed["Task3|dft|raw_pca"]["dataset_macro_f1"],
                                 main3["raw_f1_dataset_macro"]),
    }
    # GPU kernels can differ slightly across the old main run and this new
    # replay; these checks detect recipe drift, not bitwise device identity.
    tolerance = _replay_tolerance(spec)
    for name, (observed, reference) in full_checks.items():
        _close(name, observed, reference, tolerance=tolerance)
    return _write_audit(spec, {
        "record_counts": expected,
        "replay_tolerance": tolerance,
        "paired_task3_capacity_equal": True,
        "independently_reconstructed_rows": reconstructed,
        "full_recipe_replay_checks": {
            key: {"observed": value[0], "main_reference": value[1]}
            for key, value in full_checks.items()
        },
    }, [Path(config_path), *paths, summary_path])


def _collect_noise(spec: dict):
    root = Path(spec["output_root"])
    sources = {
        task: _load_yaml(spec[task]["source_config"])
        for task in ("task1", "task2", "task3")
    }
    paths = [root / "task1" / "per_run.csv"]
    rows = {"Task1": _read_csv(paths[0]), "Task2": [], "Task3": []}
    for task_name in ("Task2", "Task3"):
        task = task_name.lower()
        for dataset in sources[task]["datasets"]:
            for seed in spec[task]["training_seeds"]:
                path = root / task / "shards" / f"{dataset}_seed{int(seed)}.csv"
                paths.append(path); rows[task_name].extend(_read_csv(path))
    expected = {
        "Task1": len(sources["task1"]["datasets"]) * len(spec["task1"]["arms"])
        * len(spec["task1"]["kmeans_seeds"]) * len(spec["corruptions"]),
        "Task2": len(sources["task2"]["datasets"]) * len(spec["task2"]["arms"])
        * len(spec["task2"]["training_seeds"]) * len(spec["corruptions"]),
        "Task3": len(sources["task3"]["datasets"]) * len(spec["task3"]["arms"])
        * len(spec["task3"]["training_seeds"]) * len(spec["corruptions"]),
    }
    for name in rows:
        _assert_count(name, rows[name], expected[name])
        _assert_metric(rows[name], "f1")
    _assert_unique("Task1", rows["Task1"], ("dataset", "arm", "seed", "condition"))
    _assert_unique("Task2", rows["Task2"],
                   ("dataset", "arm", "training_seed", "condition"))
    _assert_unique("Task3", rows["Task3"],
                   ("dataset", "arm", "training_seed", "condition"))
    _assert_metric(rows["Task3"], "average_precision")
    return rows, paths, expected


def audit_noise(config_path: str | Path) -> Path:
    spec = _load_yaml(config_path)
    rows, paths, expected = _collect_noise(spec)
    table = {}
    for task_name, values in rows.items():
        arm_field = "arm"
        for condition in [row["id"] for row in spec["corruptions"]]:
            for arm in sorted({row[arm_field] for row in values}):
                selected = [row for row in values
                            if row["condition"] == condition and row[arm_field] == arm]
                key = f"{task_name}|{condition}|{arm}"
                table[key] = {
                    "dataset_macro_f1": _dataset_macro(
                        [{**row, "audit_arm": key} for row in selected],
                        "audit_arm", key, "f1",
                    )
                }
                if task_name == "Task3":
                    table[key]["dataset_macro_average_precision"] = _dataset_macro(
                        [{**row, "audit_arm": key} for row in selected],
                        "audit_arm", key, "average_precision",
                    )
    summary_path = Path(spec["output_root"]) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    formal = {
        f"{row['task']}|{row['condition']}|{row['arm']}": row
        for row in summary["table"]
    }
    if set(formal) != set(table):
        raise RuntimeError("noise summary keys differ from independent reconstruction")
    for key, metrics in table.items():
        for metric, value in metrics.items():
            _close(f"noise {key}/{metric}", formal[key][metric], value)

    main1 = json.loads(Path(spec["task1"]["main_summary"]).read_text(encoding="utf-8"))
    main2 = json.loads(Path(spec["task2"]["main_summary"]).read_text(encoding="utf-8"))
    main3 = json.loads(Path(spec["task3"]["main_summary"]).read_text(encoding="utf-8"))
    clean_checks = {
        "Task1 raw": (table["Task1|clean|raw"]["dataset_macro_f1"],
                      main1["dataset_macro"]["raw_f1"]),
        "Task1 fmt": (table["Task1|clean|fmt"]["dataset_macro_f1"],
                      main1["dataset_macro"]["fmt_f1"]),
        "Task2 raw": (table["Task2|clean|raw"]["dataset_macro_f1"],
                      main2["raw_f1_dataset_macro"]),
        "Task2 fmt": (table["Task2|clean|fmt"]["dataset_macro_f1"],
                      main2["fmt_f1_dataset_macro"]),
        "Task3 raw-pca": (table["Task3|clean|raw_pca"]["dataset_macro_f1"],
                          main3["raw_f1_dataset_macro"]),
        "Task3 fmt": (table["Task3|clean|fmt"]["dataset_macro_f1"],
                      main3["fmt_f1_dataset_macro"]),
    }
    for name, (observed, reference) in clean_checks.items():
        _close(name, observed, reference, tolerance=0.02)
    return _write_audit(spec, {
        "record_counts": expected,
        "clean_train_corrupted_confirmation": True,
        "paired_corruption_realizations_across_arms": bool(
            spec["randomization"]["realization_is_paired_across_arms"]
        ),
        "independently_reconstructed_table": table,
        "clean_recipe_replay_checks": {
            key: {"observed": value[0], "main_reference": value[1]}
            for key, value in clean_checks.items()
        },
    }, [Path(config_path), *paths, summary_path])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("strong", "ablation", "noise"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.kind == "strong":
        audit_strong(args.config)
    elif args.kind == "ablation":
        audit_ablation(args.config)
    else:
        audit_noise(args.config)


if __name__ == "__main__":
    main()
