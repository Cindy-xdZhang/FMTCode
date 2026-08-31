"""Independently audit the frozen 3D Task2 and Task3 core F1 claims.

The audit reads per-run CSV files rather than trusting either experiment's
summary implementation.  It checks paired coverage, recomputes dataset-,
family-, and seed-level means, compares the recomputation with the published
summary, and evaluates the shared 15-percentage-point target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_close(actual: float, expected: float, label: str) -> float:
    difference = abs(actual - expected)
    if difference > TOLERANCE:
        raise ValueError(
            f"{label} differs by {difference:.17g}: "
            f"recomputed={actual:.17g}, summary={expected:.17g}"
        )
    return difference


def _audit_task2(root: Path, target: float) -> dict:
    per_run_path = root / "per_run.csv"
    summary_path = root / "summary.json"
    rows = [row for row in _read_csv(per_run_path) if row["recipe"] == "selected"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({int(row["training_seed"]) for row in rows})
    arms = sorted({row["arm"] for row in rows})
    if len(datasets) != 10 or len(seeds) != 5 or arms != ["fmt", "raw"]:
        raise ValueError(
            f"Task2 coverage mismatch: datasets={len(datasets)}, "
            f"seeds={len(seeds)}, arms={arms}"
        )
    if len(rows) != len(datasets) * len(seeds) * len(arms):
        raise ValueError(f"Task2 expected 100 selected rows, found {len(rows)}")

    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    group_by_dataset: dict[str, str] = {}
    for row in rows:
        dataset = row["dataset"]
        seed = int(row["training_seed"])
        arm = row["arm"]
        if arm in paired[(dataset, seed)]:
            raise ValueError(f"duplicate Task2 row for {(dataset, seed, arm)}")
        paired[(dataset, seed)][arm] = float(row["confirmation_f1"])
        previous_group = group_by_dataset.setdefault(dataset, row["group"])
        if previous_group != row["group"]:
            raise ValueError(f"Task2 family changed within {dataset}")

    if any(set(pair) != {"raw", "fmt"} for pair in paired.values()):
        raise ValueError("Task2 contains an incomplete Raw/FMT pair")

    dataset_details = {}
    for dataset in datasets:
        raw = _mean(paired[(dataset, seed)]["raw"] for seed in seeds)
        fmt = _mean(paired[(dataset, seed)]["fmt"] for seed in seeds)
        dataset_details[dataset] = {
            "physical_family": group_by_dataset[dataset],
            "raw_f1": raw,
            "fmt_f1": fmt,
            "f1_gain": fmt - raw,
        }

    raw_macro = _mean(item["raw_f1"] for item in dataset_details.values())
    fmt_macro = _mean(item["fmt_f1"] for item in dataset_details.values())
    gain = fmt_macro - raw_macro
    seed_gains = {
        str(seed): _mean(
            paired[(dataset, seed)]["fmt"] - paired[(dataset, seed)]["raw"]
            for dataset in datasets
        )
        for seed in seeds
    }
    family_values: dict[str, list[float]] = defaultdict(list)
    for item in dataset_details.values():
        family_values[item["physical_family"]].append(item["f1_gain"])
    family_gains = {
        family: _mean(values) for family, values in sorted(family_values.items())
    }

    selected = summary["selected"]
    differences = {
        "raw_f1": _require_close(raw_macro, float(selected["raw_f1"]), "Task2 Raw F1"),
        "fmt_f1": _require_close(fmt_macro, float(selected["fmt_f1"]), "Task2 FMT F1"),
        "f1_gain": _require_close(
            gain, float(selected["fmt_minus_raw_f1"]), "Task2 F1 gain"
        ),
    }
    for dataset, item in dataset_details.items():
        published = selected["datasets"][dataset]
        _require_close(item["raw_f1"], float(published["raw_f1"]), f"Task2 {dataset} Raw")
        _require_close(item["fmt_f1"], float(published["fmt_f1"]), f"Task2 {dataset} FMT")
        _require_close(
            item["f1_gain"],
            float(published["fmt_minus_raw_f1"]),
            f"Task2 {dataset} gain",
        )

    return {
        "experiment": summary["experiment"],
        "comparison": summary["comparison"],
        "counts": {
            "selected_rows": len(rows),
            "datasets": len(datasets),
            "families": len(family_gains),
            "paired_seeds": len(seeds),
        },
        "dataset_macro": {"raw_f1": raw_macro, "fmt_f1": fmt_macro, "f1_gain": gain},
        "family_macro_f1_gain": _mean(family_gains.values()),
        "family_gains": family_gains,
        "seed_macro_gains": seed_gains,
        "positive_dataset_count": sum(
            item["f1_gain"] > 0.0 for item in dataset_details.values()
        ),
        "positive_family_count": sum(value > 0.0 for value in family_gains.values()),
        "positive_seed_count": sum(value > 0.0 for value in seed_gains.values()),
        "minimum_dataset_f1_gain": min(
            item["f1_gain"] for item in dataset_details.values()
        ),
        "target": target,
        "target_reached": gain >= target,
        "maximum_absolute_difference_vs_summary": max(differences.values()),
        "sha256": {"per_run_csv": _sha256(per_run_path), "summary": _sha256(summary_path)},
    }


def _audit_task3(root: Path, target: float) -> dict:
    per_run_path = root / "per_run.csv"
    summary_path = root / "summary.json"
    rows = _read_csv(per_run_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    sources = sorted({row["source"] for row in rows})
    if len(datasets) != 10 or len(seeds) != 2 or sources != ["fmt", "raw_pca"]:
        raise ValueError(
            f"Task3 coverage mismatch: datasets={len(datasets)}, "
            f"seeds={len(seeds)}, sources={sources}"
        )
    if len(rows) != len(datasets) * len(seeds) * len(sources):
        raise ValueError(f"Task3 expected 40 rows, found {len(rows)}")

    paired: dict[tuple[str, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    family_by_dataset: dict[str, str] = {}
    for row in rows:
        dataset = row["dataset"]
        seed = int(row["seed"])
        source = row["source"]
        if source in paired[(dataset, seed)]:
            raise ValueError(f"duplicate Task3 row for {(dataset, seed, source)}")
        paired[(dataset, seed)][source] = (
            float(row["f1"]),
            float(row["average_precision"]),
        )
        previous_family = family_by_dataset.setdefault(dataset, row["physical_family"])
        if previous_family != row["physical_family"]:
            raise ValueError(f"Task3 family changed within {dataset}")

    if any(set(pair) != {"raw_pca", "fmt"} for pair in paired.values()):
        raise ValueError("Task3 contains an incomplete Raw-PCA/FMT pair")

    dataset_details = {}
    for dataset in datasets:
        raw_f1 = _mean(paired[(dataset, seed)]["raw_pca"][0] for seed in seeds)
        fmt_f1 = _mean(paired[(dataset, seed)]["fmt"][0] for seed in seeds)
        raw_ap = _mean(paired[(dataset, seed)]["raw_pca"][1] for seed in seeds)
        fmt_ap = _mean(paired[(dataset, seed)]["fmt"][1] for seed in seeds)
        dataset_details[dataset] = {
            "physical_family": family_by_dataset[dataset],
            "raw_pca_f1": raw_f1,
            "fmt_f1": fmt_f1,
            "f1_gain": fmt_f1 - raw_f1,
            "raw_pca_ap": raw_ap,
            "fmt_ap": fmt_ap,
            "ap_gain": fmt_ap - raw_ap,
        }

    raw_f1 = _mean(item["raw_pca_f1"] for item in dataset_details.values())
    fmt_f1 = _mean(item["fmt_f1"] for item in dataset_details.values())
    f1_gain = fmt_f1 - raw_f1
    raw_ap = _mean(item["raw_pca_ap"] for item in dataset_details.values())
    fmt_ap = _mean(item["fmt_ap"] for item in dataset_details.values())
    ap_gain = fmt_ap - raw_ap
    family_values: dict[str, list[float]] = defaultdict(list)
    for item in dataset_details.values():
        family_values[item["physical_family"]].append(item["f1_gain"])
    family_gains = {
        family: _mean(values) for family, values in sorted(family_values.items())
    }
    seed_gains = {
        str(seed): _mean(
            paired[(dataset, seed)]["fmt"][0]
            - paired[(dataset, seed)]["raw_pca"][0]
            for dataset in datasets
        )
        for seed in seeds
    }

    differences = {
        "raw_f1": _require_close(
            raw_f1, float(summary["dataset_macro_raw_pca_f1"]), "Task3 Raw-PCA F1"
        ),
        "fmt_f1": _require_close(
            fmt_f1, float(summary["dataset_macro_fmt_f1"]), "Task3 FMT F1"
        ),
        "f1_gain": _require_close(
            f1_gain,
            float(summary["dataset_macro_f1_gain_vs_raw_pca"]),
            "Task3 F1 gain",
        ),
        "raw_ap": _require_close(
            raw_ap, float(summary["dataset_macro_raw_pca_ap"]), "Task3 Raw-PCA AP"
        ),
        "fmt_ap": _require_close(
            fmt_ap, float(summary["dataset_macro_fmt_ap"]), "Task3 FMT AP"
        ),
        "ap_gain": _require_close(
            ap_gain,
            float(summary["dataset_macro_ap_gain_vs_raw_pca"]),
            "Task3 AP gain",
        ),
    }

    return {
        "experiment": summary["experiment"],
        "comparison": "FMT residual versus paired equal-capacity Raw-PCA residual",
        "counts": {
            "rows": len(rows),
            "datasets": len(datasets),
            "families": len(family_gains),
            "paired_seeds": len(seeds),
        },
        "dataset_macro": {
            "raw_pca_f1": raw_f1,
            "fmt_f1": fmt_f1,
            "f1_gain": f1_gain,
            "raw_pca_average_precision": raw_ap,
            "fmt_average_precision": fmt_ap,
            "average_precision_gain": ap_gain,
        },
        "family_macro_f1_gain": _mean(family_gains.values()),
        "family_gains": family_gains,
        "seed_macro_gains": seed_gains,
        "positive_dataset_count": sum(
            item["f1_gain"] > 0.0 for item in dataset_details.values()
        ),
        "positive_family_count": sum(value > 0.0 for value in family_gains.values()),
        "positive_seed_count": sum(value > 0.0 for value in seed_gains.values()),
        "minimum_dataset_f1_gain": min(
            item["f1_gain"] for item in dataset_details.values()
        ),
        "target": target,
        "target_reached": f1_gain >= target,
        "maximum_absolute_difference_vs_summary": max(differences.values()),
        "sha256": {"per_run_csv": _sha256(per_run_path), "summary": _sha256(summary_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2-dir", required=True, type=Path)
    parser.add_argument("--task3-dir", required=True, type=Path)
    parser.add_argument("--target", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    task2 = _audit_task2(args.task2_dir, args.target)
    task3 = _audit_task3(args.task3_dir, args.target)
    report = {
        "schema": 1,
        "status": "passed" if task2["target_reached"] and task3["target_reached"] else "failed",
        "shared_target": args.target,
        "both_tasks_reached_target": task2["target_reached"] and task3["target_reached"],
        "minimum_task_gain": min(
            task2["dataset_macro"]["f1_gain"], task3["dataset_macro"]["f1_gain"]
        ),
        "task2": task2,
        "task3": task3,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
