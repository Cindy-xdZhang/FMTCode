"""Independently audit a completed paired Task3 parameter search.

This program deliberately does not import the search/selector implementation.
It reconstructs every family decision from the archived per-run CSV files,
then compares the reconstruction with the selector JSON and leaderboard CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
ARMS = ("fmt", "raw_pca")
METRICS = ("f1", "average_precision")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a YAML mapping")
    return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def _groups(config: dict[str, Any]) -> dict[str, list[str]]:
    current = config
    visited: set[Path] = set()
    while "groups" not in current:
        source = current.get("base_search_config")
        if not source:
            raise KeyError("neither config nor its bases define groups")
        path = _resolve(source).resolve()
        if path in visited:
            raise RuntimeError("cycle in base_search_config chain")
        visited.add(path)
        current = _load_yaml(path)
    groups = current["groups"]
    if not isinstance(groups, dict) or not groups:
        raise TypeError("groups must be a non-empty mapping")
    return {
        str(family): [str(dataset) for dataset in spec["datasets"]]
        for family, spec in groups.items()
    }


def _candidate_ids(config: dict[str, Any]) -> list[str]:
    candidates = config.get("optimization_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TypeError("optimization_candidates must be a non-empty list")
    result = [str(candidate["id"]) for candidate in candidates]
    if len(result) != len(set(result)):
        raise ValueError("optimization candidate IDs are not unique")
    return result


def _read_per_run_archive(path: Path) -> dict[tuple[str, str, int, str], dict]:
    rows: dict[tuple[str, str, int, str], dict] = {}
    with tarfile.open(path, "r:gz") as archive:
        unsafe = [
            member.name for member in archive.getmembers()
            if Path(member.name).is_absolute()
            or ".." in Path(member.name).parts
        ]
        if unsafe:
            raise ValueError(f"unsafe archive members: {unsafe[:10]}")
        members = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/per_run.csv")
        ]
        for member in members:
            parts = Path(member.name).parts
            try:
                start = parts.index("candidates")
                candidate, dataset, seed_part, arm, filename = parts[start + 1:]
            except (ValueError, IndexError) as error:
                raise ValueError(
                    f"unexpected per-run archive member {member.name!r}"
                ) from error
            if filename != "per_run.csv" or arm not in ARMS:
                raise ValueError(f"unexpected per-run path {member.name!r}")
            if not seed_part.startswith("seed"):
                raise ValueError(f"invalid seed directory {seed_part!r}")
            seed = int(seed_part[4:])
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot extract {member.name}")
            with io.TextIOWrapper(extracted, encoding="utf-8", newline="") as stream:
                values = list(csv.DictReader(stream))
            if len(values) != 1:
                raise RuntimeError(
                    f"{member.name}: expected exactly one result row, got "
                    f"{len(values)}"
                )
            key = (candidate, dataset, seed, arm)
            if key in rows:
                raise RuntimeError(f"duplicate per-run result {key}")
            row = values[0]
            if str(row.get("optimization_id")) != candidate:
                raise RuntimeError(f"optimization_id differs from path for {key}")
            if str(row.get("dataset")) != dataset:
                raise RuntimeError(f"dataset differs from path for {key}")
            if int(row.get("seed", -1)) != seed:
                raise RuntimeError(f"seed differs from path for {key}")
            rows[key] = row
    return rows


def _float(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]!r}")
    return value


def _score(row: dict, key: str) -> float:
    value = _float(row, key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} outside [0,1]: {value}")
    return value


def _candidate_summary(
    rows: dict[tuple[str, str, int, str], dict],
    candidate: str,
    family: str,
    datasets: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    per_dataset: dict[str, Any] = {}
    per_seed_gains = {seed: [] for seed in seeds}
    total_parameter_counts: set[int] = set()
    for dataset in datasets:
        per_seed: dict[int, Any] = {}
        for seed in seeds:
            paired = {
                arm: rows[(candidate, dataset, seed, arm)] for arm in ARMS
            }
            residual_counts = {
                int(paired[arm]["trainable_residual_parameter_count"])
                for arm in ARMS
            }
            if len(residual_counts) != 1:
                raise RuntimeError(
                    f"unequal paired residual parameter counts for "
                    f"{candidate}/{dataset}/seed{seed}: {residual_counts}"
                )
            total_parameter_counts.add(int(paired["fmt"]["parameter_count"]))
            per_seed[seed] = {
                arm: {
                    "f1": _score(paired[arm], "validation_f1"),
                    "average_precision": _score(
                        paired[arm], "validation_average_precision"
                    ),
                }
                for arm in ARMS
            }
            per_seed_gains[seed].append(
                per_seed[seed]["fmt"]["f1"]
                - per_seed[seed]["raw_pca"]["f1"]
            )
        means = {
            arm: {
                metric: _mean([
                    per_seed[seed][arm][metric] for seed in seeds
                ])
                for metric in METRICS
            }
            for arm in ARMS
        }
        per_dataset[dataset] = {
            **means,
            "f1_gain": means["fmt"]["f1"] - means["raw_pca"]["f1"],
            "average_precision_gain": (
                means["fmt"]["average_precision"]
                - means["raw_pca"]["average_precision"]
            ),
        }
    f1_gains = [entry["f1_gain"] for entry in per_dataset.values()]
    ap_gains = [
        entry["average_precision_gain"] for entry in per_dataset.values()
    ]
    return {
        "physical_family": family,
        "optimization_id": candidate,
        "eligible": True,
        "dataset_macro_fmt_f1": _mean([
            entry["fmt"]["f1"] for entry in per_dataset.values()
        ]),
        "dataset_macro_raw_pca_f1": _mean([
            entry["raw_pca"]["f1"] for entry in per_dataset.values()
        ]),
        "dataset_macro_fmt_average_precision": _mean([
            entry["fmt"]["average_precision"]
            for entry in per_dataset.values()
        ]),
        "dataset_macro_raw_pca_average_precision": _mean([
            entry["raw_pca"]["average_precision"]
            for entry in per_dataset.values()
        ]),
        "dataset_macro_f1_gain_vs_raw_pca": _mean(f1_gains),
        "dataset_macro_average_precision_gain_vs_raw_pca": _mean(ap_gains),
        "positive_dataset_count": sum(value > 0.0 for value in f1_gains),
        "worst_dataset_f1_gain": min(f1_gains),
        "worst_seed_f1_gain": min(
            _mean(per_seed_gains[seed]) for seed in seeds
        ),
        "minimum_total_parameter_count": min(total_parameter_counts),
        "maximum_total_parameter_count": max(total_parameter_counts),
        "datasets": per_dataset,
    }


def _apply_guard(rows: list[dict], selection: dict[str, Any]) -> None:
    guard = selection.get("absolute_fmt_guard")
    if guard is None:
        return
    control_id = str(guard["control_optimization_id"])
    controls = [row for row in rows if row["optimization_id"] == control_id]
    if len(controls) != 1:
        raise RuntimeError(f"expected one control {control_id!r}")
    control = controls[0]
    control_f1 = control["dataset_macro_fmt_f1"]
    control_ap = control["dataset_macro_fmt_average_precision"]
    for row in rows:
        row["absolute_fmt_f1_delta_vs_control"] = (
            row["dataset_macro_fmt_f1"] - control_f1
        )
        row["absolute_fmt_average_precision_delta_vs_control"] = (
            row["dataset_macro_fmt_average_precision"] - control_ap
        )
        row["eligible"] = bool(
            row["absolute_fmt_f1_delta_vs_control"]
            >= -float(guard["f1_tolerance"])
            and row["absolute_fmt_average_precision_delta_vs_control"]
            >= -float(guard["average_precision_tolerance"])
        )


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"cannot parse boolean {value!r}")


def _update_difference(current: float, left: Any, right: Any) -> float:
    return max(current, abs(float(left) - float(right)))


def audit(config_path: Path, artifact_dir: Path, output_path: Path) -> dict:
    config = _load_yaml(config_path)
    groups = _groups(config)
    candidates = _candidate_ids(config)
    seeds = [int(seed) for seed in config["paired_seeds"]]
    datasets = [dataset for values in groups.values() for dataset in values]
    if len(datasets) != len(set(datasets)):
        raise ValueError("a dataset occurs in more than one physical family")

    archive_path = artifact_dir / "per_run_csv.tar.gz"
    selection_path = artifact_dir / "optimization_selection.json"
    leaderboard_path = artifact_dir / "optimization_leaderboard.csv"
    manifest_path = artifact_dir / "preflight_manifest.json"
    for path in (archive_path, selection_path, leaderboard_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    per_run = _read_per_run_archive(archive_path)
    expected_keys = {
        (candidate, dataset, seed, arm)
        for candidate in candidates
        for dataset in datasets
        for seed in seeds
        for arm in ARMS
    }
    if set(per_run) != expected_keys:
        missing = sorted(expected_keys - set(per_run))[:10]
        extra = sorted(set(per_run) - expected_keys)[:10]
        raise RuntimeError(
            f"per-run archive is not complete: missing={missing}, extra={extra}"
        )

    with selection_path.open(encoding="utf-8") as stream:
        selector = json.load(stream)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if str(selector.get("experiment")) != str(config["experiment"]):
        raise RuntimeError("selector experiment differs from config")
    if str(manifest.get("experiment")) != str(config["experiment"]):
        raise RuntimeError("preflight experiment differs from config")
    expected_training_runs = len(expected_keys)
    if int(manifest.get("expected_training_runs", expected_training_runs)) \
            != expected_training_runs:
        raise RuntimeError("preflight expected_training_runs differs")
    if int(manifest.get("dataset_count", len(datasets))) != len(datasets):
        raise RuntimeError("preflight dataset_count differs")
    if int(manifest.get("optimization_candidate_count", len(candidates))) \
            != len(candidates):
        raise RuntimeError("preflight optimization_candidate_count differs")
    with leaderboard_path.open(encoding="utf-8", newline="") as stream:
        leaderboard = list(csv.DictReader(stream))
    if len(leaderboard) != len(groups) * len(candidates):
        raise RuntimeError("selector leaderboard has an unexpected row count")

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    selected: dict[str, dict[str, Any]] = {}
    required = [
        str(config["selection"]["primary_metric"]),
        *[str(value) for value in config["selection"]["tie_breakers"]],
    ]
    for family, family_datasets in groups.items():
        family_rows = [
            _candidate_summary(
                per_run, candidate, family, family_datasets, seeds
            )
            for candidate in candidates
        ]
        _apply_guard(family_rows, config["selection"])
        for row in family_rows:
            summaries[(family, row["optimization_id"])] = row
        eligible = [row for row in family_rows if row["eligible"]]
        if not eligible:
            raise RuntimeError(f"independent audit found no eligible {family} row")
        selected[family] = max(
            eligible,
            key=lambda row: tuple(float(row[key]) for key in required),
        )

    selector_rows = {
        (row["physical_family"], row["optimization_id"]): row
        for row in leaderboard
    }
    if set(selector_rows) != set(summaries):
        raise RuntimeError("selector and independent leaderboard keys differ")
    maximum_difference = 0.0
    numeric_keys = (
        "dataset_macro_fmt_f1",
        "dataset_macro_raw_pca_f1",
        "dataset_macro_fmt_average_precision",
        "dataset_macro_raw_pca_average_precision",
        "dataset_macro_f1_gain_vs_raw_pca",
        "dataset_macro_average_precision_gain_vs_raw_pca",
        "positive_dataset_count",
        "worst_dataset_f1_gain",
        "worst_seed_f1_gain",
        "minimum_total_parameter_count",
        "maximum_total_parameter_count",
    )
    for key, independent in summaries.items():
        reported = selector_rows[key]
        if _bool(reported["eligible"]) != independent["eligible"]:
            raise RuntimeError(f"selector eligibility differs for {key}")
        for metric in numeric_keys:
            maximum_difference = _update_difference(
                maximum_difference, independent[metric], reported[metric]
            )

    selector_primary = selector["primary_by_group"]
    selected_ids = {
        family: row["optimization_id"] for family, row in selected.items()
    }
    reported_ids = {
        family: str(selector_primary[family]["optimization_id"])
        for family in groups
    }
    if selected_ids != reported_ids:
        raise RuntimeError(
            f"selector winners differ: independent={selected_ids}, "
            f"selector={reported_ids}"
        )

    selected_datasets = [
        metrics
        for family, row in selected.items()
        for metrics in row["datasets"].values()
    ]
    dataset_macro = {
        "fmt_f1": _mean([row["fmt"]["f1"] for row in selected_datasets]),
        "raw_pca_f1": _mean([
            row["raw_pca"]["f1"] for row in selected_datasets
        ]),
        "f1_gain": _mean([row["f1_gain"] for row in selected_datasets]),
        "fmt_average_precision": _mean([
            row["fmt"]["average_precision"] for row in selected_datasets
        ]),
        "raw_pca_average_precision": _mean([
            row["raw_pca"]["average_precision"]
            for row in selected_datasets
        ]),
        "average_precision_gain": _mean([
            row["average_precision_gain"] for row in selected_datasets
        ]),
    }
    guard = config["selection"].get("absolute_fmt_guard")
    control_dataset_macro = None
    if guard is not None:
        control_id = str(guard["control_optimization_id"])
        control_datasets = [
            metrics
            for family in groups
            for metrics in summaries[(family, control_id)]["datasets"].values()
        ]
        control_dataset_macro = {
            "fmt_f1": _mean([
                row["fmt"]["f1"] for row in control_datasets
            ]),
            "raw_pca_f1": _mean([
                row["raw_pca"]["f1"] for row in control_datasets
            ]),
            "f1_gain": _mean([
                row["f1_gain"] for row in control_datasets
            ]),
            "fmt_average_precision": _mean([
                row["fmt"]["average_precision"] for row in control_datasets
            ]),
            "raw_pca_average_precision": _mean([
                row["raw_pca"]["average_precision"]
                for row in control_datasets
            ]),
            "average_precision_gain": _mean([
                row["average_precision_gain"] for row in control_datasets
            ]),
        }
    selector_macro_fields = {
        "fmt_f1": "development_dataset_macro_fmt_f1",
        "raw_pca_f1": "development_dataset_macro_raw_pca_f1",
        "f1_gain": "development_dataset_macro_f1_gain_vs_raw_pca",
        "fmt_average_precision": (
            "development_dataset_macro_fmt_average_precision"
        ),
        "raw_pca_average_precision": (
            "development_dataset_macro_raw_pca_average_precision"
        ),
        "average_precision_gain": (
            "development_dataset_macro_ap_gain_vs_raw_pca"
        ),
    }
    for key, selector_key in selector_macro_fields.items():
        maximum_difference = _update_difference(
            maximum_difference, dataset_macro[key], selector[selector_key]
        )

    source_hash_fields = (
        "optimization_config_sha256",
        "preflight_manifest_sha256",
        "upstream_selection_sha256",
    )
    source_hashes = {
        key: {str(row[key]) for row in per_run.values()}
        for key in source_hash_fields
    }
    all_source_hashes_consistent = all(
        len(values) == 1 for values in source_hashes.values()
    ) and all(
        next(iter(source_hashes[key])) == str(selector[key])
        for key in source_hash_fields
    ) and (
        str(selector["preflight_manifest_sha256"]) == _sha256(manifest_path)
    ) and (
        str(selector["optimization_config_sha256"])
        == str(manifest["optimization_config_sha256"])
    ) and (
        str(selector["upstream_selection_sha256"])
        == str(manifest["upstream_selection_sha256"])
    )

    result = {
        "status": "passed" if maximum_difference <= 1e-12 else "failed",
        "independent_of_selector_implementation": True,
        "counts": {
            "arms": len(ARMS),
            "candidates": len(candidates),
            "datasets": len(datasets),
            "families": len(groups),
            "leaderboard_rows": len(leaderboard),
            "per_run_csv": len(per_run),
            "seeds": len(seeds),
        },
        "selected_optimization_id_by_family": selected_ids,
        "dataset_macro": dataset_macro,
        "control_dataset_macro": control_dataset_macro,
        "positive_dataset_count": sum(
            row["f1_gain"] > 0.0 for row in selected_datasets
        ),
        "worst_dataset_f1_gain": min(
            row["f1_gain"] for row in selected_datasets
        ),
        "maximum_absolute_difference_vs_selector": maximum_difference,
        "all_paired_parameter_counts_equal": True,
        "all_source_hashes_consistent": all_source_hashes_consistent,
        "input_sha256": {
            "optimization_leaderboard": _sha256(leaderboard_path),
            "optimization_selection": _sha256(selection_path),
            "per_run_csv_archive": _sha256(archive_path),
            "preflight_manifest": _sha256(manifest_path),
        },
    }
    if not all_source_hashes_consistent:
        result["status"] = "failed"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config_path = _resolve(arguments.config).resolve()
    artifact_dir = _resolve(arguments.artifact_dir).resolve()
    output_path = (
        _resolve(arguments.output).resolve()
        if arguments.output is not None
        else artifact_dir / "independent_audit.json"
    )
    result = audit(config_path, artifact_dir, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
