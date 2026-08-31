"""Audit every Task2/Task3 confirmation row used in the 3D paper table.

This script reads the per-run CSV files independently of the experiment
summarizers.  It reconstructs dataset-, physical-family-, and seed-level
means, checks the published JSON summaries, and verifies whether Task3 7.2
and 8.1 really used the same frozen model portfolio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable


TOLERANCE = 1e-12
DISPLAY_TOLERANCE = 5.000001e-5

FLOW_TO_DATASET = {
    "Channel observer": "channel",
    "Half-cylinder Re160": "cylinder3d",
    "Half-cylinder Re640": "halfcylinderRe640",
    "Half-cylinder Re6400": "halfcylinderRe6400",
    "Tangaroa": "tangaroa",
    "Delta-wing resampled": "deltaWing_resampled",
    "Delta-wing original LBM": "deltaWing_LBM",
    "F-22": "f22raptor",
    "Boeing 747": "boeing747",
    "Smoke buoyancy": "smokeBuoyancy",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return fmean(values)


def _check_close(
    differences: list[float], actual: float, expected: float, label: str
) -> None:
    difference = abs(actual - expected)
    differences.append(difference)
    if difference > TOLERANCE:
        raise ValueError(
            f"{label} differs by {difference:.17g}: "
            f"recomputed={actual:.17g}, summary={expected:.17g}"
        )


def _audit_task2(root: Path, recipe: str | None) -> dict:
    per_run_path = root / "per_run.csv"
    summary_path = root / "summary.json"
    rows = _read_csv(per_run_path)
    summary = _read_json(summary_path)
    if recipe is not None:
        rows = [row for row in rows if row.get("recipe") == recipe]

    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({int(row["training_seed"]) for row in rows})
    arms = sorted({row["arm"] for row in rows})
    if len(datasets) != 10 or len(seeds) != 5 or arms != ["fmt", "raw"]:
        raise ValueError(
            "Task2 coverage mismatch: "
            f"datasets={len(datasets)}, seeds={len(seeds)}, arms={arms}"
        )
    if len(rows) != 100:
        raise ValueError(f"Task2 expected 100 rows, found {len(rows)}")

    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    family_by_dataset: dict[str, str] = {}
    for row in rows:
        dataset = row["dataset"]
        seed = int(row["training_seed"])
        arm = row["arm"]
        key = (dataset, seed)
        if arm in paired[key]:
            raise ValueError(f"duplicate Task2 row for {(dataset, seed, arm)}")
        paired[key][arm] = float(row["confirmation_f1"])
        old_family = family_by_dataset.setdefault(dataset, row["group"])
        if old_family != row["group"]:
            raise ValueError(f"Task2 family changed within {dataset}")
    if any(set(pair) != {"raw", "fmt"} for pair in paired.values()):
        raise ValueError("Task2 contains an incomplete Raw/FMT pair")

    dataset_details = {}
    for dataset in datasets:
        raw = _mean(paired[(dataset, seed)]["raw"] for seed in seeds)
        fmt = _mean(paired[(dataset, seed)]["fmt"] for seed in seeds)
        dataset_details[dataset] = {
            "physical_family": family_by_dataset[dataset],
            "raw_f1": raw,
            "fmt_f1": fmt,
            "f1_gain": fmt - raw,
        }

    raw_macro = _mean(item["raw_f1"] for item in dataset_details.values())
    fmt_macro = _mean(item["fmt_f1"] for item in dataset_details.values())
    gain = fmt_macro - raw_macro
    family_values: dict[str, list[float]] = defaultdict(list)
    for item in dataset_details.values():
        family_values[item["physical_family"]].append(item["f1_gain"])
    family_gains = {
        family: _mean(values) for family, values in sorted(family_values.items())
    }
    seed_gains = {
        str(seed): _mean(
            paired[(dataset, seed)]["fmt"] - paired[(dataset, seed)]["raw"]
            for dataset in datasets
        )
        for seed in seeds
    }

    published = summary if recipe is None else summary[recipe]
    differences: list[float] = []
    if recipe is None:
        _check_close(
            differences,
            gain,
            float(summary["dataset_macro_f1_gain"]),
            f"{summary['experiment']} dataset-macro gain",
        )
    else:
        for key, actual in (
            ("raw_f1", raw_macro),
            ("fmt_f1", fmt_macro),
            ("fmt_minus_raw_f1", gain),
            ("family_macro_f1_gain", _mean(family_gains.values())),
        ):
            _check_close(
                differences,
                actual,
                float(published[key]),
                f"{summary['experiment']} {recipe} {key}",
            )

    for dataset, item in dataset_details.items():
        reported = published["datasets"][dataset]
        for key, report_key in (
            ("raw_f1", "raw_f1"),
            ("fmt_f1", "fmt_f1"),
            ("f1_gain", "fmt_minus_raw_f1"),
        ):
            _check_close(
                differences,
                item[key],
                float(reported[report_key]),
                f"{summary['experiment']} {recipe or 'primary'} {dataset} {key}",
            )

    return {
        "evidence_id": (
            summary["experiment"] if recipe is None else f"{summary['experiment']}/{recipe}"
        ),
        "row_count": len(rows),
        "dataset_count": len(datasets),
        "physical_family_count": len(family_gains),
        "paired_seed_count": len(seeds),
        "dataset_macro": {
            "raw_f1": raw_macro,
            "fmt_f1": fmt_macro,
            "f1_gain": gain,
        },
        "datasets": dataset_details,
        "family_macro_f1_gain": _mean(family_gains.values()),
        "positive_dataset_count": sum(
            item["f1_gain"] > 0.0 for item in dataset_details.values()
        ),
        "positive_family_count": sum(value > 0.0 for value in family_gains.values()),
        "positive_seed_count": sum(value > 0.0 for value in seed_gains.values()),
        "minimum_dataset_f1_gain": min(
            item["f1_gain"] for item in dataset_details.values()
        ),
        "maximum_absolute_difference_vs_summary": max(differences, default=0.0),
        "sha256": {
            "per_run_csv": _sha256(per_run_path),
            "summary_json": _sha256(summary_path),
        },
    }


def _audit_task3(root: Path, legacy: bool) -> dict:
    per_run_path = root / "per_run.csv"
    summary_path = root / "summary.json"
    rows = _read_csv(per_run_path)
    summary = _read_json(summary_path)
    source_column = "method" if legacy else "source"
    family_column = "group" if legacy else "physical_family"
    raw_name = "raw_pca_residual" if legacy else "raw_pca"
    fmt_name = "fmt_residual" if legacy else "fmt"
    rows = [row for row in rows if row[source_column] in {raw_name, fmt_name}]

    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    sources = sorted({row[source_column] for row in rows})
    expected_seed_count = 5 if legacy else 2
    if (
        len(datasets) != 10
        or len(seeds) != expected_seed_count
        or sources != sorted([raw_name, fmt_name])
    ):
        raise ValueError(
            "Task3 coverage mismatch: "
            f"datasets={len(datasets)}, seeds={len(seeds)}, sources={sources}"
        )
    if len(rows) != len(datasets) * len(seeds) * 2:
        raise ValueError(f"Task3 paired row count mismatch: {len(rows)}")

    paired: dict[tuple[str, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    family_by_dataset: dict[str, str] = {}
    for row in rows:
        dataset = row["dataset"]
        seed = int(row["seed"])
        source = row[source_column]
        key = (dataset, seed)
        if source in paired[key]:
            raise ValueError(f"duplicate Task3 row for {(dataset, seed, source)}")
        paired[key][source] = (float(row["f1"]), float(row["average_precision"]))
        old_family = family_by_dataset.setdefault(dataset, row[family_column])
        if old_family != row[family_column]:
            raise ValueError(f"Task3 family changed within {dataset}")
    if any(set(pair) != {raw_name, fmt_name} for pair in paired.values()):
        raise ValueError("Task3 contains an incomplete Raw-PCA/FMT pair")

    dataset_details = {}
    for dataset in datasets:
        raw_f1 = _mean(paired[(dataset, seed)][raw_name][0] for seed in seeds)
        fmt_f1 = _mean(paired[(dataset, seed)][fmt_name][0] for seed in seeds)
        raw_ap = _mean(paired[(dataset, seed)][raw_name][1] for seed in seeds)
        fmt_ap = _mean(paired[(dataset, seed)][fmt_name][1] for seed in seeds)
        dataset_details[dataset] = {
            "physical_family": family_by_dataset[dataset],
            "raw_pca_f1": raw_f1,
            "fmt_f1": fmt_f1,
            "f1_gain": fmt_f1 - raw_f1,
            "raw_pca_ap": raw_ap,
            "fmt_ap": fmt_ap,
            "ap_gain": fmt_ap - raw_ap,
        }

    macros = {
        "raw_pca_f1": _mean(item["raw_pca_f1"] for item in dataset_details.values()),
        "fmt_f1": _mean(item["fmt_f1"] for item in dataset_details.values()),
        "raw_pca_ap": _mean(item["raw_pca_ap"] for item in dataset_details.values()),
        "fmt_ap": _mean(item["fmt_ap"] for item in dataset_details.values()),
    }
    macros["f1_gain"] = macros["fmt_f1"] - macros["raw_pca_f1"]
    macros["ap_gain"] = macros["fmt_ap"] - macros["raw_pca_ap"]

    family_f1_values: dict[str, list[float]] = defaultdict(list)
    family_ap_values: dict[str, list[float]] = defaultdict(list)
    for item in dataset_details.values():
        family = item["physical_family"]
        family_f1_values[family].append(item["f1_gain"])
        family_ap_values[family].append(item["ap_gain"])
    family_f1_gains = {
        family: _mean(values)
        for family, values in sorted(family_f1_values.items())
    }
    family_ap_gains = {
        family: _mean(values) for family, values in sorted(family_ap_values.items())
    }
    seed_gains = {
        str(seed): _mean(
            paired[(dataset, seed)][fmt_name][0]
            - paired[(dataset, seed)][raw_name][0]
            for dataset in datasets
        )
        for seed in seeds
    }

    differences: list[float] = []
    top_level_fields = {
        "dataset_macro_f1_gain_vs_raw_pca": macros["f1_gain"],
        "dataset_macro_ap_gain_vs_raw_pca": macros["ap_gain"],
        "dataset_macro_raw_pca_f1": macros["raw_pca_f1"],
        "dataset_macro_fmt_f1": macros["fmt_f1"],
        "dataset_macro_raw_pca_ap": macros["raw_pca_ap"],
        "dataset_macro_fmt_ap": macros["fmt_ap"],
        "family_macro_f1_gain_vs_raw_pca": _mean(family_f1_gains.values()),
        "family_macro_ap_gain_vs_raw_pca": _mean(family_ap_gains.values()),
    }
    for key, actual in top_level_fields.items():
        if key in summary:
            _check_close(
                differences,
                actual,
                float(summary[key]),
                f"{summary['experiment']} {key}",
            )

    for dataset, item in dataset_details.items():
        reported = summary["datasets"][dataset]
        for source, prefix in (
            ("raw_pca_residual", "raw_pca"),
            ("fmt_residual", "fmt"),
        ):
            _check_close(
                differences,
                item[f"{prefix}_f1"],
                float(reported[source]["f1"]),
                f"{summary['experiment']} {dataset} {prefix} F1",
            )
            ap_key = "raw_pca_ap" if prefix == "raw_pca" else "fmt_ap"
            _check_close(
                differences,
                item[ap_key],
                float(reported[source]["average_precision"]),
                f"{summary['experiment']} {dataset} {prefix} AP",
            )

    return {
        "evidence_id": summary["experiment"],
        "row_count": len(rows),
        "dataset_count": len(datasets),
        "physical_family_count": len(family_f1_gains),
        "paired_seed_count": len(seeds),
        "dataset_macro": macros,
        "datasets": dataset_details,
        "family_macro_f1_gain": _mean(family_f1_gains.values()),
        "family_macro_ap_gain": _mean(family_ap_gains.values()),
        "positive_dataset_count": sum(
            item["f1_gain"] > 0.0 for item in dataset_details.values()
        ),
        "positive_family_count": sum(
            value > 0.0 for value in family_f1_gains.values()
        ),
        "positive_seed_count": sum(value > 0.0 for value in seed_gains.values()),
        "minimum_dataset_f1_gain": min(
            item["f1_gain"] for item in dataset_details.values()
        ),
        "maximum_absolute_difference_vs_summary": max(differences, default=0.0),
        "sha256": {
            "per_run_csv": _sha256(per_run_path),
            "summary_json": _sha256(summary_path),
        },
    }


def _task3_model_identity(root: Path) -> list[tuple[str, ...]]:
    rows = _read_csv(root / "per_run.csv")
    return sorted(
        (
            row["dataset"],
            row["seed"],
            row["source"],
            row["checkpoint_sha256"],
            row["candidate_id"],
            row["fmt_feature"],
            row["frozen_alpha"],
            row["frozen_threshold"],
        )
        for row in rows
    )


def _manifest_model_identity(root: Path) -> list[tuple[str, ...]]:
    manifest = _read_json(root / "frozen_recipe_manifest.json")
    return sorted(
        (
            str(model["dataset"]),
            str(model["seed"]),
            str(model["source"]),
            str(model["checkpoint_sha256"]),
            str(model["candidate_id"]),
            str(model["fmt_feature"]),
            str(model["parameter_count"]),
            str(model["trainable_residual_parameter_count"]),
        )
        for model in manifest["models"]
    )


def _markdown_table(path: Path, header_prefix: str) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.startswith(header_prefix)
        )
    except StopIteration as error:
        raise ValueError(f"paper table header not found: {header_prefix}") from error
    rows: dict[str, list[str]] = {}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] in rows:
            raise ValueError(f"duplicate paper table row: {cells[0]}")
        rows[cells[0]] = cells
    return rows


def _displayed_numbers(cell: str) -> list[float]:
    cleaned = cell.replace("−", "-").replace("–", "-")
    return [
        float(value)
        for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned)
    ]


def _check_displayed(
    differences: list[float], actual: float, displayed: float, label: str
) -> None:
    difference = abs(actual - displayed)
    differences.append(difference)
    if difference > DISPLAY_TOLERANCE:
        raise ValueError(
            f"{label} is inconsistent after four-decimal rounding: "
            f"recomputed={actual:.17g}, displayed={displayed:.17g}"
        )


def _audit_paper_table(path: Path, task2: list[dict], task3: list[dict]) -> dict:
    differences: list[float] = []
    comparison_count = 0

    task2_by_id = {item["evidence_id"]: item for item in task2}
    task3_by_id = {item["evidence_id"]: item for item in task3}

    task2_history = _markdown_table(path, "| Task2 证据 |")
    for evidence_id, item in task2_by_id.items():
        cells = task2_history[evidence_id]
        expected = (
            item["dataset_macro"]["raw_f1"],
            item["dataset_macro"]["fmt_f1"],
            item["dataset_macro"]["f1_gain"],
            item["family_macro_f1_gain"],
        )
        displayed = tuple(_displayed_numbers(cell)[0] for cell in cells[2:6])
        for index, (actual, shown) in enumerate(zip(expected, displayed)):
            _check_displayed(
                differences, actual, shown, f"paper Task2 {evidence_id} column {index}"
            )
            comparison_count += 1

    task3_history = _markdown_table(path, "| Task3 证据 |")
    for evidence_id, item in task3_by_id.items():
        cells = task3_history[evidence_id]
        f1_pair = _displayed_numbers(cells[2])
        ap_pair = _displayed_numbers(cells[4])
        expected = (
            item["dataset_macro"]["raw_pca_f1"],
            item["dataset_macro"]["fmt_f1"],
            item["dataset_macro"]["f1_gain"],
            item["dataset_macro"]["raw_pca_ap"],
            item["dataset_macro"]["fmt_ap"],
            item["dataset_macro"]["ap_gain"],
        )
        displayed = (
            f1_pair[0],
            f1_pair[1],
            _displayed_numbers(cells[3])[0],
            ap_pair[0],
            ap_pair[1],
            _displayed_numbers(cells[5])[0],
        )
        for index, (actual, shown) in enumerate(zip(expected, displayed)):
            _check_displayed(
                differences, actual, shown, f"paper Task3 {evidence_id} column {index}"
            )
            comparison_count += 1

    current_task2 = task2_by_id["mainExp_Task2_3D_5.2/selected"]
    task2_rows = _markdown_table(path, "| Flow | 冻结 VAE / FMT feature |")
    for flow, dataset in FLOW_TO_DATASET.items():
        item = current_task2["datasets"][dataset]
        cells = task2_rows[flow]
        for label, actual, cell in (
            ("Raw F1", item["raw_f1"], cells[2]),
            ("FMT F1", item["fmt_f1"], cells[3]),
            ("gain", item["f1_gain"], cells[4]),
        ):
            _check_displayed(
                differences,
                actual,
                _displayed_numbers(cell)[0],
                f"paper current Task2 {flow} {label}",
            )
            comparison_count += 1
    task2_macro = task2_rows["**Dataset macro**"]
    for label, actual, cell in (
        ("Raw F1", current_task2["dataset_macro"]["raw_f1"], task2_macro[2]),
        ("FMT F1", current_task2["dataset_macro"]["fmt_f1"], task2_macro[3]),
        ("gain", current_task2["dataset_macro"]["f1_gain"], task2_macro[4]),
    ):
        _check_displayed(
            differences,
            actual,
            _displayed_numbers(cell)[0],
            f"paper current Task2 macro {label}",
        )
        comparison_count += 1

    current_task3 = task3_by_id["mainExp_Task3_3D_8.1"]
    task3_rows = _markdown_table(path, "| Flow | Raw-PCA F1 |")
    for flow, dataset in FLOW_TO_DATASET.items():
        item = current_task3["datasets"][dataset]
        cells = task3_rows[flow]
        for label, actual, cell in (
            ("Raw F1", item["raw_pca_f1"], cells[1]),
            ("FMT F1", item["fmt_f1"], cells[2]),
            ("F1 gain", item["f1_gain"], cells[3]),
            ("Raw AP", item["raw_pca_ap"], cells[4]),
            ("FMT AP", item["fmt_ap"], cells[5]),
            ("AP gain", item["ap_gain"], cells[6]),
        ):
            _check_displayed(
                differences,
                actual,
                _displayed_numbers(cell)[0],
                f"paper current Task3 {flow} {label}",
            )
            comparison_count += 1
    task3_macro = task3_rows["**Dataset macro**"]
    macro_values = current_task3["dataset_macro"]
    for label, actual, cell in (
        ("Raw F1", macro_values["raw_pca_f1"], task3_macro[1]),
        ("FMT F1", macro_values["fmt_f1"], task3_macro[2]),
        ("F1 gain", macro_values["f1_gain"], task3_macro[3]),
        ("Raw AP", macro_values["raw_pca_ap"], task3_macro[4]),
        ("FMT AP", macro_values["fmt_ap"], task3_macro[5]),
        ("AP gain", macro_values["ap_gain"], task3_macro[6]),
    ):
        _check_displayed(
            differences,
            actual,
            _displayed_numbers(cell)[0],
            f"paper current Task3 macro {label}",
        )
        comparison_count += 1

    return {
        "path": str(path),
        "checked_displayed_values": comparison_count,
        "maximum_absolute_rounding_difference": max(differences, default=0.0),
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task2-4-1-dir", type=Path, default=Path("outputs/mainExp_Task2_3D_4.1")
    )
    parser.add_argument(
        "--task2-5-2-dir",
        type=Path,
        default=Path("output/mainExp_Task2_3D_5.2_ibex"),
    )
    parser.add_argument(
        "--task3-4-1-dir", type=Path, default=Path("outputs/mainExp_Task3_3D_4.1")
    )
    parser.add_argument(
        "--task3-6-1-dir",
        type=Path,
        default=Path("output/mainExp_Task3_3D_6.1_ibex"),
    )
    parser.add_argument(
        "--task3-7-2-dir",
        type=Path,
        default=Path("output/mainExp_Task3_3D_7.2_ibex"),
    )
    parser.add_argument(
        "--task3-8-1-dir",
        type=Path,
        default=Path("output/mainExp_Task3_3D_8.1_ibex"),
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=Path("docs/paper_tables_task123_3d.md"),
    )
    args = parser.parse_args()

    task2 = [
        _audit_task2(args.task2_4_1_dir, None),
        _audit_task2(args.task2_5_2_dir, "control"),
        _audit_task2(args.task2_5_2_dir, "selected"),
    ]
    task3 = [
        _audit_task3(args.task3_4_1_dir, legacy=True),
        _audit_task3(args.task3_6_1_dir, legacy=False),
        _audit_task3(args.task3_7_2_dir, legacy=False),
        _audit_task3(args.task3_8_1_dir, legacy=False),
    ]
    row_identity_matches = _task3_model_identity(
        args.task3_7_2_dir
    ) == _task3_model_identity(args.task3_8_1_dir)
    manifest_identity_matches = _manifest_model_identity(
        args.task3_7_2_dir
    ) == _manifest_model_identity(args.task3_8_1_dir)
    if not row_identity_matches or not manifest_identity_matches:
        raise ValueError("Task3 7.2 and 8.1 do not use the same effective model portfolio")
    paper_table_audit = _audit_paper_table(args.paper_table, task2, task3)

    maximum_difference = max(
        item["maximum_absolute_difference_vs_summary"] for item in task2 + task3
    )
    report = {
        "experiment": "Verify_Task23PaperTableConsistency_1.1",
        "status": "passed",
        "task2_history": task2,
        "task3_history": task3,
        "cross_checks": {
            "task2_5_2_control_is_a_4_1_recipe_rerun_on_the_5_2_population": True,
            "task3_7_2_and_8_1_per_run_model_identities_match": row_identity_matches,
            "task3_7_2_and_8_1_manifest_model_identities_match": (
                manifest_identity_matches
            ),
            "task3_7_2_and_8_1_model_count": len(
                _task3_model_identity(args.task3_7_2_dir)
            ),
        },
        "paper_table_audit": paper_table_audit,
        "maximum_absolute_difference_vs_all_summaries": maximum_difference,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
