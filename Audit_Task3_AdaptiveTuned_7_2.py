"""Independently audit the sealed Task3 7.2 confirmation artifacts.

This module deliberately does not import the confirmation, evaluation, or
summary implementation.  It reconstructs every aggregate from ``per_run.csv``
and verifies the frozen model/evaluation evidence by content hash.  A passed
audit means the evidence is internally consistent; it does not require the
scientific gain target to be reached.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


TOLERANCE = 1e-12
EXPECTED_EXPERIMENT = "mainExp_Task3_3D_7.2"
EXPECTED_SOURCE_EXPERIMENT = "Verify_Task3_AdaptivePortfolio_52.1"
EXPECTED_SOURCES = ("fmt", "raw_pca")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise RuntimeError("cannot average an empty sequence")
    return sum(values) / len(values)


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{label} differs: actual={actual!r}, expected={expected!r}"
        )


def _require_close(actual: float, expected: float, label: str) -> float:
    difference = abs(float(actual) - float(expected))
    if difference > TOLERANCE:
        raise RuntimeError(
            f"{label} differs by {difference:.17g}: "
            f"recomputed={float(actual):.17g}, published={float(expected):.17g}"
        )
    return difference


def _finite_unit(value: str | float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{label} is not finite in [0,1]: {value!r}")
    return number


def _finite(value: str | float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return number


def _load_config(config_path: Path) -> dict:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise RuntimeError("Task3 7.2 config is not a mapping")
    required = {
        "experiment", "task", "status", "datasets", "paired_seeds",
        "confirmation_count", "expected_ivd_percentile", "phase_key",
        "phase_key_sha256", "halton_index", "confirmation_seed_grid_phase",
        "target_dataset_macro_f1_gain",
        "aspirational_dataset_macro_f1_gain", "source_model",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise RuntimeError(f"Task3 7.2 config keys missing: {missing}")
    _require_equal(spec["experiment"], EXPECTED_EXPERIMENT, "config experiment")
    _require_equal(spec["task"], "Task3", "config task")
    _require_equal(
        spec["status"], "fresh_spatial_confirmation", "config status"
    )
    datasets = [str(value) for value in spec["datasets"]]
    seeds = [int(value) for value in spec["paired_seeds"]]
    if len(datasets) != 10 or len(set(datasets)) != 10:
        raise RuntimeError("Task3 7.2 requires ten unique datasets")
    _require_equal(seeds, [40, 41], "config paired seeds")
    _require_equal(int(spec["confirmation_count"]), 4, "confirmation count")
    _require_close(
        float(spec["expected_ivd_percentile"]), 95.0, "IVD percentile"
    )
    _require_equal(
        spec["source_model"]["expected_experiment"],
        EXPECTED_SOURCE_EXPERIMENT,
        "source portfolio experiment",
    )
    if not bool(spec.get("new_spatial_primitive_population", False)):
        raise RuntimeError("Task3 7.2 is not declared as a new population")
    if bool(spec.get("confirmation_opened_before_freeze", True)):
        raise RuntimeError("Task3 7.2 confirmation was opened before freeze")
    spec = dict(spec)
    spec["datasets"] = datasets
    spec["paired_seeds"] = seeds
    spec["config_sha256"] = _sha256(config_path)
    return spec


def _model_index(manifest: dict, spec: dict) -> dict[tuple[str, int, str], dict]:
    models = manifest.get("models", [])
    expected = {
        (dataset, seed, source)
        for dataset in spec["datasets"]
        for seed in spec["paired_seeds"]
        for source in EXPECTED_SOURCES
    }
    observed: dict[tuple[str, int, str], dict] = {}
    for model in models:
        key = (
            str(model.get("dataset")),
            int(model.get("seed", -1)),
            str(model.get("source")),
        )
        if key not in expected or key in observed:
            raise RuntimeError(f"unexpected or duplicate frozen model: {key}")
        for path_key, hash_key in (
            ("result", "result_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            path = Path(str(model.get(path_key, "")))
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = _sha256(path)
            expected_hash = str(model.get(hash_key, "")).lower()
            _require_equal(actual_hash, expected_hash, f"{key} {path_key} hash")
        observed[key] = model
    _require_equal(set(observed), expected, "frozen model key set")

    for dataset in spec["datasets"]:
        for seed in spec["paired_seeds"]:
            pair = [observed[(dataset, seed, source)] for source in EXPECTED_SOURCES]
            for field in (
                "fmt_dim", "parameter_count",
                "trainable_residual_parameter_count",
            ):
                values = {int(model[field]) for model in pair}
                if len(values) != 1:
                    raise RuntimeError(
                        f"paired {field} differs: {dataset}/seed{seed}"
                    )
            for field in ("physical_family", "candidate_id", "fmt_feature"):
                values = {str(model[field]) for model in pair}
                if len(values) != 1:
                    raise RuntimeError(
                        f"paired {field} differs: {dataset}/seed{seed}"
                    )
    return observed


def _validate_evidence(
    spec: dict,
    manifest_path: Path,
    manifest: dict,
    evaluation_path: Path,
    evaluation: dict,
) -> tuple[str, str, dict[tuple[str, int, str], dict]]:
    manifest_hash = _sha256(manifest_path)
    evaluation_hash = _sha256(evaluation_path)
    for name, payload in (("manifest", manifest), ("evaluation", evaluation)):
        _require_equal(payload.get("experiment"), spec["experiment"], f"{name} experiment")
        _require_equal(payload.get("status"), spec["status"], f"{name} status")
        _require_equal(
            str(payload.get("config_sha256", "")).lower(),
            spec["config_sha256"],
            f"{name} config hash",
        )
    _require_equal(
        str(evaluation.get("recipe_manifest_sha256", "")).lower(),
        manifest_hash,
        "evaluation manifest hash",
    )
    _require_equal(
        evaluation.get("source_model_selection_sha256"),
        manifest.get("source_model_selection_sha256"),
        "evaluation source selection hash",
    )
    _require_equal(
        evaluation.get("confirmation_seed_grid_phase"),
        spec["confirmation_seed_grid_phase"],
        "evaluation spatial phase",
    )
    _require_equal(
        bool(evaluation.get("confirmation_was_generated_after_recipe_freeze")),
        True,
        "evaluation generated after freeze",
    )
    _require_equal(int(evaluation.get("expected_evaluations", -1)), 40, "evaluation count")

    _require_equal(manifest.get("phase_key"), spec["phase_key"], "manifest phase key")
    _require_equal(
        manifest.get("phase_key_sha256"),
        spec["phase_key_sha256"],
        "manifest phase-key hash",
    )
    _require_equal(int(manifest.get("halton_index", -1)), int(spec["halton_index"]), "manifest Halton index")
    _require_equal(
        manifest.get("confirmation_seed_grid_phase"),
        spec["confirmation_seed_grid_phase"],
        "manifest spatial phase",
    )
    _require_equal(bool(manifest.get("confirmation_data_opened")), False, "manifest confirmation state")
    _require_equal(bool(manifest.get("new_spatial_primitive_population")), True, "manifest population state")
    _require_equal(evaluation.get("models"), manifest.get("models"), "evaluation model manifest")
    return manifest_hash, evaluation_hash, _model_index(manifest, spec)


def _validate_rows(
    rows: list[dict[str, str]],
    spec: dict,
    models: dict[tuple[str, int, str], dict],
    manifest_hash: str,
    evaluation_hash: str,
) -> dict[tuple[str, int], dict[str, dict]]:
    expected = set(models)
    observed: dict[tuple[str, int, str], dict] = {}
    paired: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("dataset")), int(row.get("seed", -1)), str(row.get("source")))
        if key not in expected or key in observed:
            raise RuntimeError(f"unexpected or duplicate result row: {key}")
        model = models[key]
        expected_fields = {
            "experiment": spec["experiment"],
            "status": spec["status"],
            "config_sha256": spec["config_sha256"],
            "recipe_manifest_sha256": manifest_hash,
            "evaluation_preflight_sha256": evaluation_hash,
            "dataset": key[0],
            "physical_family": model["physical_family"],
            "candidate_id": model["candidate_id"],
            "fmt_feature": model["fmt_feature"],
            "source": key[2],
            "method": "fmt_residual" if key[2] == "fmt" else "raw_pca_residual",
            "checkpoint": model["checkpoint"],
            "checkpoint_sha256": model["checkpoint_sha256"],
        }
        for field, value in expected_fields.items():
            _require_equal(
                str(row.get(field, "")).lower(),
                str(value).lower(),
                f"{key} {field}",
            )
        count = int(row.get("sample_count", 0))
        if count <= 0:
            raise RuntimeError(f"{key} has no evaluation samples")
        parsed = {
            **row,
            "sample_count": count,
            "positive_fraction": _finite_unit(row["positive_fraction"], f"{key} positive fraction"),
            "frozen_threshold": _finite_unit(row["frozen_threshold"], f"{key} threshold"),
            "frozen_alpha": _finite(row["frozen_alpha"], f"{key} alpha"),
            "f1": _finite_unit(row["f1"], f"{key} F1"),
            "average_precision": _finite_unit(row["average_precision"], f"{key} average precision"),
        }
        observed[key] = parsed
        paired[(key[0], key[1])][key[2]] = parsed
    _require_equal(set(observed), expected, "result row key set")
    for pair_key, pair in paired.items():
        _require_equal(set(pair), set(EXPECTED_SOURCES), f"{pair_key} paired arms")
        _require_equal(
            pair["fmt"]["sample_count"], pair["raw_pca"]["sample_count"],
            f"{pair_key} paired sample count",
        )
        _require_close(
            pair["fmt"]["positive_fraction"],
            pair["raw_pca"]["positive_fraction"],
            f"{pair_key} paired positive fraction",
        )
    return paired


def _aggregate(paired: dict, spec: dict) -> dict:
    datasets = {}
    seed_f1_gains: dict[str, list[float]] = defaultdict(list)
    seed_ap_gains: dict[str, list[float]] = defaultdict(list)
    for dataset in spec["datasets"]:
        family_values = {
            str(paired[(dataset, seed)][source]["physical_family"])
            for seed in spec["paired_seeds"] for source in EXPECTED_SOURCES
        }
        if len(family_values) != 1:
            raise RuntimeError(f"physical family changes within {dataset}")
        methods = {}
        for source in EXPECTED_SOURCES:
            methods[source] = {
                metric: _mean(
                    paired[(dataset, seed)][source][metric]
                    for seed in spec["paired_seeds"]
                )
                for metric in ("f1", "average_precision")
            }
        datasets[dataset] = {
            "physical_family": next(iter(family_values)),
            "raw_pca_residual": methods["raw_pca"],
            "fmt_residual": methods["fmt"],
            "f1_gain": methods["fmt"]["f1"] - methods["raw_pca"]["f1"],
            "average_precision_gain": (
                methods["fmt"]["average_precision"]
                - methods["raw_pca"]["average_precision"]
            ),
        }
        for seed in spec["paired_seeds"]:
            pair = paired[(dataset, seed)]
            seed_f1_gains[str(seed)].append(
                pair["fmt"]["f1"] - pair["raw_pca"]["f1"]
            )
            seed_ap_gains[str(seed)].append(
                pair["fmt"]["average_precision"]
                - pair["raw_pca"]["average_precision"]
            )

    family_rows: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for dataset, values in datasets.items():
        family_rows[values["physical_family"]].append((dataset, values))
    families = {
        family: {
            "datasets": sorted(dataset for dataset, _ in values),
            "f1_gain": _mean(row["f1_gain"] for _, row in values),
            "average_precision_gain": _mean(
                row["average_precision_gain"] for _, row in values
            ),
        }
        for family, values in sorted(family_rows.items())
    }
    raw_f1 = _mean(row["raw_pca_residual"]["f1"] for row in datasets.values())
    fmt_f1 = _mean(row["fmt_residual"]["f1"] for row in datasets.values())
    raw_ap = _mean(
        row["raw_pca_residual"]["average_precision"] for row in datasets.values()
    )
    fmt_ap = _mean(
        row["fmt_residual"]["average_precision"] for row in datasets.values()
    )
    minimum_dataset = min(datasets, key=lambda name: datasets[name]["f1_gain"])
    return {
        "datasets": datasets,
        "families": families,
        "dataset_macro_raw_pca_f1": raw_f1,
        "dataset_macro_fmt_f1": fmt_f1,
        "dataset_macro_raw_pca_ap": raw_ap,
        "dataset_macro_fmt_ap": fmt_ap,
        "dataset_macro_f1_gain_vs_raw_pca": fmt_f1 - raw_f1,
        "dataset_macro_ap_gain_vs_raw_pca": fmt_ap - raw_ap,
        "family_macro_f1_gain_vs_raw_pca": _mean(
            row["f1_gain"] for row in families.values()
        ),
        "family_macro_ap_gain_vs_raw_pca": _mean(
            row["average_precision_gain"] for row in families.values()
        ),
        "positive_dataset_f1_gain_count": sum(
            row["f1_gain"] > 0.0 for row in datasets.values()
        ),
        "positive_family_f1_gain_count": sum(
            row["f1_gain"] > 0.0 for row in families.values()
        ),
        "minimum_dataset": minimum_dataset,
        "minimum_dataset_f1_gain": datasets[minimum_dataset]["f1_gain"],
        "seed_macro_f1_gains": {
            seed: _mean(values) for seed, values in sorted(seed_f1_gains.items())
        },
        "seed_macro_ap_gains": {
            seed: _mean(values) for seed, values in sorted(seed_ap_gains.items())
        },
    }


def _compare_summary(summary: dict, aggregate: dict, spec: dict,
                     manifest_hash: str, evaluation_hash: str,
                     manifest: dict) -> float:
    exact = {
        "experiment": spec["experiment"],
        "status": spec["status"],
        "source_search_experiment": EXPECTED_SOURCE_EXPERIMENT,
        "source_portfolio_experiment": EXPECTED_SOURCE_EXPERIMENT,
        "fresh_confirmation": True,
        "confirmation_data_was_not_used_for_selection": True,
        "source_portfolio_artifacts_copied_before_source_cleanup": True,
        "recipe_manifest_sha256": manifest_hash,
        "evaluation_preflight_sha256": evaluation_hash,
        "source_model_selection_sha256": manifest["source_model_selection_sha256"],
        "confirmation_seed_grid_phase": spec["confirmation_seed_grid_phase"],
        "phase_key_sha256": spec["phase_key_sha256"],
        "halton_index": int(spec["halton_index"]),
        "paired_seeds": spec["paired_seeds"],
    }
    for field, value in exact.items():
        _require_equal(summary.get(field), value, f"summary {field}")

    differences = []
    for field in (
        "dataset_macro_raw_pca_f1", "dataset_macro_fmt_f1",
        "dataset_macro_raw_pca_ap", "dataset_macro_fmt_ap",
        "dataset_macro_f1_gain_vs_raw_pca",
        "dataset_macro_ap_gain_vs_raw_pca",
        "family_macro_f1_gain_vs_raw_pca",
        "family_macro_ap_gain_vs_raw_pca", "minimum_dataset_f1_gain",
    ):
        differences.append(_require_close(
            aggregate[field], float(summary[field]), f"summary {field}"
        ))
    for field in (
        "positive_dataset_f1_gain_count", "positive_family_f1_gain_count",
    ):
        _require_equal(int(summary[field]), int(aggregate[field]), f"summary {field}")

    for dataset, values in aggregate["datasets"].items():
        published = summary["datasets"][dataset]
        _require_equal(
            published["physical_family"], values["physical_family"],
            f"summary {dataset} physical family",
        )
        for field in ("f1_gain", "average_precision_gain"):
            differences.append(_require_close(
                values[field], published[field], f"summary {dataset} {field}"
            ))
        for arm in ("raw_pca_residual", "fmt_residual"):
            for metric in ("f1", "average_precision"):
                differences.append(_require_close(
                    values[arm][metric], published[arm][metric],
                    f"summary {dataset} {arm} {metric}",
                ))
    _require_equal(set(summary["datasets"]), set(aggregate["datasets"]), "summary dataset set")

    for family, values in aggregate["families"].items():
        published = summary["families"][family]
        _require_equal(published["datasets"], values["datasets"], f"summary {family} datasets")
        for field in ("f1_gain", "average_precision_gain"):
            differences.append(_require_close(
                values[field], published[field], f"summary {family} {field}"
            ))
    _require_equal(set(summary["families"]), set(aggregate["families"]), "summary family set")

    primary = float(spec["target_dataset_macro_f1_gain"])
    aspirational = float(spec["aspirational_dataset_macro_f1_gain"])
    _require_close(float(summary["target_dataset_macro_f1_gain"]), primary, "summary primary target")
    _require_close(float(summary["aspirational_dataset_macro_f1_gain"]), aspirational, "summary aspirational target")
    gain = aggregate["dataset_macro_f1_gain_vs_raw_pca"]
    _require_equal(bool(summary["target_reached"]), gain >= primary, "summary primary decision")
    _require_equal(
        bool(summary["aspirational_target_reached"]),
        gain >= aspirational,
        "summary aspirational decision",
    )
    return max(differences, default=0.0)


def audit(config_path: str | Path, artifact_dir: str | Path,
          output: str | Path | None = None) -> dict:
    config_path = Path(config_path)
    artifact_dir = Path(artifact_dir)
    spec = _load_config(config_path)
    paths = {
        "per_run_csv": artifact_dir / "per_run.csv",
        "summary": artifact_dir / "summary.json",
        "manifest": artifact_dir / "frozen_recipe_manifest.json",
        "evaluation_preflight": artifact_dir / "evaluation_preflight.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    evaluation = json.loads(
        paths["evaluation_preflight"].read_text(encoding="utf-8")
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    manifest_hash, evaluation_hash, models = _validate_evidence(
        spec, paths["manifest"], manifest,
        paths["evaluation_preflight"], evaluation,
    )
    rows = _read_csv(paths["per_run_csv"])
    paired = _validate_rows(
        rows, spec, models, manifest_hash, evaluation_hash
    )
    aggregate = _aggregate(paired, spec)
    maximum_difference = _compare_summary(
        summary, aggregate, spec, manifest_hash, evaluation_hash, manifest
    )
    primary = float(spec["target_dataset_macro_f1_gain"])
    aspirational = float(spec["aspirational_dataset_macro_f1_gain"])
    gain = aggregate["dataset_macro_f1_gain_vs_raw_pca"]
    report = {
        "schema": 1,
        "status": "passed",
        "experiment": spec["experiment"],
        "comparison": (
            "FMT residual versus paired same-recipe, same-capacity "
            "train-only Raw-PCA residual"
        ),
        "counts": {
            "rows": len(rows),
            "datasets": len(aggregate["datasets"]),
            "families": len(aggregate["families"]),
            "paired_seeds": len(spec["paired_seeds"]),
            "frozen_models": len(models),
        },
        "dataset_macro": {
            "raw_pca_f1": aggregate["dataset_macro_raw_pca_f1"],
            "fmt_f1": aggregate["dataset_macro_fmt_f1"],
            "f1_gain": gain,
            "raw_pca_average_precision": aggregate["dataset_macro_raw_pca_ap"],
            "fmt_average_precision": aggregate["dataset_macro_fmt_ap"],
            "average_precision_gain": aggregate["dataset_macro_ap_gain_vs_raw_pca"],
        },
        "family_macro_f1_gain": aggregate["family_macro_f1_gain_vs_raw_pca"],
        "family_macro_average_precision_gain": aggregate["family_macro_ap_gain_vs_raw_pca"],
        "positive_dataset_f1_gain_count": aggregate["positive_dataset_f1_gain_count"],
        "positive_family_f1_gain_count": aggregate["positive_family_f1_gain_count"],
        "minimum_dataset": aggregate["minimum_dataset"],
        "minimum_dataset_f1_gain": aggregate["minimum_dataset_f1_gain"],
        "seed_macro_f1_gains": aggregate["seed_macro_f1_gains"],
        "seed_macro_average_precision_gains": aggregate["seed_macro_ap_gains"],
        "primary_target": primary,
        "primary_target_reached": gain >= primary,
        "aspirational_target": aspirational,
        "aspirational_target_reached": gain >= aspirational,
        "maximum_absolute_difference_vs_summary": maximum_difference,
        "sha256": {name: _sha256(path) for name, path in paths.items()},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit(args.config, args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
