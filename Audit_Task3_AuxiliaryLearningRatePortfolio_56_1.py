"""Independently audit Task3 portfolio 56.1 without importing its selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from Search_Task3_FMTResidual_3D import _read_csv


EXPERIMENT = "Verify_Task3_AuxiliaryLearningRatePortfolio_56.1"
CURRENT_SOURCE_NAME = "current_portfolio"
AUXILIARY_SOURCE_NAME = "auxiliary_learning_rate"
AUXILIARY_ARCHIVE_COUNT = 540
CURRENT_SOURCE_LABEL = "54.1"
AUXILIARY_SOURCE_LABEL = "55.1"
PORTFOLIO_LABEL = "56.1"
ARMS = ("fmt", "raw_pca")
MACRO_FIELDS = {
    "development_dataset_macro_f1_gain_vs_raw_pca": "f1_gain",
    "development_dataset_macro_ap_gain_vs_raw_pca": "average_precision_gain",
    "development_dataset_macro_fmt_f1": ("fmt", "f1"),
    "development_dataset_macro_raw_pca_f1": ("raw_pca", "f1"),
    "development_dataset_macro_fmt_average_precision": (
        "fmt", "average_precision"
    ),
    "development_dataset_macro_raw_pca_average_precision": (
        "raw_pca", "average_precision"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _under(root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def _family_map(selection: dict) -> dict[str, list[str]]:
    return {
        str(family): list(json.loads(str(row["datasets_json"])))
        for family, row in dict(selection["primary_by_group"]).items()
    }


def _metric(detail: dict, route) -> float:
    if isinstance(route, tuple):
        return _finite(detail[route[0]][route[1]], "/".join(route))
    return _finite(detail[route], str(route))


def _source_config(name: str, section: dict) -> tuple[Path, Path, dict, dict]:
    root = Path(section["repo_root"])
    paths = {key: _under(root, value) for key, value in section["paths"].items()}
    config = paths["config"]
    if not config.is_file():
        raise FileNotFoundError(config)
    if _canonical_sha256(config) != str(section["expected_config_canonical_sha256"]):
        raise RuntimeError(f"{name}: canonical config hash changed")
    overlay = yaml.safe_load(config.read_text(encoding="utf-8"))
    if str(overlay.get("experiment")) != str(section["expected_experiment"]):
        raise RuntimeError(f"{name}: experiment identity changed")
    return root, config, overlay, paths


def _current_source(section: dict) -> dict:
    root, config, overlay, paths = _source_config(CURRENT_SOURCE_NAME, section)
    selection = _json(paths["selection"])
    audit = _json(paths["audit"])
    if selection.get("experiment") != overlay["experiment"] or bool(
        selection.get("confirmation_opened", True)
    ):
        raise RuntimeError(
            f"{CURRENT_SOURCE_LABEL} source selection identity changed"
        )
    if audit.get("status") != "passed" or not audit.get("all_frozen_hashes_verified"):
        raise RuntimeError(f"{CURRENT_SOURCE_LABEL} source audit failed")
    if audit["input_sha256"].get("config") != _sha256(config):
        raise RuntimeError(f"{CURRENT_SOURCE_LABEL} audit config hash differs")
    if audit["input_sha256"].get("portfolio_selection") != _sha256(paths["selection"]):
        raise RuntimeError(
            f"{CURRENT_SOURCE_LABEL} audit selection hash differs"
        )
    frozen_root = paths["frozen_root"].resolve()
    model_map = {}
    for model in selection["models"]:
        key = (str(model["dataset"]), int(model["seed"]), str(model["source"]))
        if key in model_map:
            raise RuntimeError(f"duplicate {CURRENT_SOURCE_LABEL} model {key}")
        for field in ("result", "checkpoint"):
            path = Path(model[field]).resolve()
            try:
                path.relative_to(frozen_root)
            except ValueError as error:
                raise RuntimeError(
                    f"{CURRENT_SOURCE_LABEL} source {field} escaped frozen root"
                ) from error
            if not path.is_file() or _sha256(path) != str(model[f"{field}_sha256"]):
                raise RuntimeError(
                    f"{CURRENT_SOURCE_LABEL} source {field} changed: {key}"
                )
        model_map[key] = dict(model)
    if len(model_map) != 40:
        raise RuntimeError(f"{CURRENT_SOURCE_LABEL} model set is incomplete")
    return {
        "root": root, "overlay": overlay, "selection": selection,
        "models": model_map,
        "hashes": {
            "config": _sha256(config), "selection": _sha256(paths["selection"]),
            "audit": _sha256(paths["audit"]),
        },
    }


def _auxiliary_source(section: dict) -> dict:
    root, config, overlay, paths = _source_config(
        AUXILIARY_SOURCE_NAME, section
    )
    preflight = _json(paths["preflight"])
    selection = _json(paths["selection"])
    audit = _json(paths["audit"])
    evidence = _json(paths["evidence"])
    expected_experiment = str(overlay["experiment"])
    if any(
        str(payload.get("experiment")) != expected_experiment
        for payload in (preflight, selection)
    ):
        raise RuntimeError(
            f"{AUXILIARY_SOURCE_LABEL} source selection identity changed"
        )
    if bool(preflight.get("confirmation_opened", True)) or bool(
        selection.get("confirmation_opened", True)
    ):
        raise RuntimeError(
            f"{AUXILIARY_SOURCE_LABEL} source opened confirmation"
        )
    if audit.get("status") != "passed" or not audit.get(
        "all_source_hashes_consistent"
    ):
        raise RuntimeError(f"{AUXILIARY_SOURCE_LABEL} source audit failed")
    input_hashes = audit["input_sha256"]
    required = {
        "optimization_selection": _sha256(paths["selection"]),
        "preflight_manifest": _sha256(paths["preflight"]),
        "per_run_csv_archive": _sha256(paths["archive"]),
    }
    if any(str(input_hashes.get(key)) != value for key, value in required.items()):
        raise RuntimeError(
            f"{AUXILIARY_SOURCE_LABEL} source audit hashes differ"
        )
    if evidence.get("status") != "passed" or int(
        evidence.get("archived_per_run_csv", -1)
    ) != AUXILIARY_ARCHIVE_COUNT:
        raise RuntimeError(f"{AUXILIARY_SOURCE_LABEL} evidence is incomplete")
    if evidence.get("stable_archive_sha256") != _sha256(paths["archive"]):
        raise RuntimeError(f"{AUXILIARY_SOURCE_LABEL} archive hash differs")
    return {
        "root": root, "overlay": overlay, "selection": selection,
        "candidate_root": paths["candidate_root"],
        "hashes": {
            "config": _sha256(config), "preflight": _sha256(paths["preflight"]),
            "selection": _sha256(paths["selection"]), "audit": _sha256(paths["audit"]),
            "evidence": _sha256(paths["evidence"]), "archive": _sha256(paths["archive"]),
        },
    }


def _source_model(source_name: str, source: dict, row: dict,
                  family: str, dataset: str, seed: int, arm: str) -> dict:
    if source_name == CURRENT_SOURCE_NAME:
        return source["models"][(dataset, seed, arm)]
    candidate = str(row["optimization_id"])
    result_path = (
        source["candidate_root"] / candidate / dataset / f"seed{seed}"
        / arm / "per_run.csv"
    )
    rows = _read_csv(result_path)
    if len(rows) != 1:
        raise RuntimeError(
            f"{AUXILIARY_SOURCE_LABEL} missing source result {result_path}"
        )
    result = rows[0]
    expected_variant = "raw_fmt_residual" if arm == "fmt" else "raw_pca_residual"
    expected_fields = {
        "dataset": dataset, "seed": seed, "auxiliary_source": arm,
        "variant": expected_variant,
        "optimization_id": str(row["optimization_id"]),
    }
    for field, expected in expected_fields.items():
        if str(result.get(field, "")).lower() != str(expected).lower():
            raise RuntimeError(
                f"{AUXILIARY_SOURCE_LABEL} source result changed: {field}"
            )
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = source["root"] / checkpoint
    if not checkpoint.is_file() or checkpoint.resolve().parent != (
        result_path.parent / "checkpoints"
    ).resolve():
        raise RuntimeError(
            f"{AUXILIARY_SOURCE_LABEL} source checkpoint changed: {checkpoint}"
        )
    return {
        "dataset": dataset, "seed": seed, "source": arm,
        "physical_family": family,
        "fmt_dim": int(result["fmt_dim"]),
        "parameter_count": int(result["parameter_count"]),
        "trainable_residual_parameter_count": int(
            result["trainable_residual_parameter_count"]
        ),
        "result": str(result_path), "result_sha256": _sha256(result_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
    }


def audit(config_path: Path, artifact_dir: Path, output_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != EXPERIMENT:
        raise RuntimeError(f"{EXPERIMENT} experiment identity changed")
    if bool(config.get("confirmation_opened", True)):
        raise RuntimeError(f"{EXPERIMENT} opened confirmation")
    current = _current_source(config["sources"][CURRENT_SOURCE_NAME])
    auxiliary = _auxiliary_source(config["sources"][AUXILIARY_SOURCE_NAME])
    sources = {CURRENT_SOURCE_NAME: current, AUXILIARY_SOURCE_NAME: auxiliary}
    metrics = [
        str(config["selection"]["primary_metric"]),
        *[str(value) for value in config["selection"]["tie_breakers"]],
    ]
    maps = {name: _family_map(source["selection"]) for name, source in sources.items()}
    reference = maps[CURRENT_SOURCE_NAME]
    if maps[AUXILIARY_SOURCE_NAME] != reference:
        raise RuntimeError(f"{EXPERIMENT} source family maps differ")

    winners = {}
    expected_details = {}
    for family, datasets in reference.items():
        candidates = []
        for name, source in sources.items():
            row = dict(source["selection"]["primary_by_group"][family])
            if not bool(row.get("eligible", True)) or row.get(
                "absolute_fmt_guard_passed"
            ) is False:
                raise RuntimeError(f"{name}/{family}: source winner failed guard")
            score = tuple(
                _finite(row[metric], f"{name}/{family}/{metric}")
                for metric in metrics
            )
            candidates.append((score, name, row))
        _, name, row = max(candidates, key=lambda value: value[0])
        winners[family] = (name, row)
        details = json.loads(str(row["datasets_json"]))
        if list(details) != datasets:
            raise RuntimeError(f"{family}: source dataset order changed")
        expected_details.update(details)

    portfolio_path = artifact_dir / "portfolio_selection.json"
    portfolio = _json(portfolio_path)
    if portfolio.get("config_sha256") != _sha256(config_path):
        raise RuntimeError(f"{EXPERIMENT} portfolio/config hash mismatch")
    if portfolio.get("source_artifact_sha256") != {
        name: source["hashes"] for name, source in sources.items()
    }:
        raise RuntimeError(f"{EXPERIMENT} source artifact hashes differ")

    maximum_difference = 0.0
    reported_primary = portfolio["primary_by_group"]
    for family, (name, row) in winners.items():
        reported = reported_primary[family]
        if reported.get("portfolio_source") != name:
            raise RuntimeError(f"{family}: selected source differs")
        if str(reported.get("optimization_id")) != str(row["optimization_id"]):
            raise RuntimeError(f"{family}: selected candidate differs")
        for metric in metrics:
            maximum_difference = max(
                maximum_difference,
                abs(float(reported[metric]) - float(row[metric])),
            )

    reported_details = {
        str(row["dataset"]): row for row in portfolio["dataset_details"]
    }
    if set(reported_details) != set(expected_details):
        raise RuntimeError(f"{EXPERIMENT} dataset details differ")
    for dataset, expected in expected_details.items():
        observed = reported_details[dataset]
        for route in MACRO_FIELDS.values():
            maximum_difference = max(
                maximum_difference,
                abs(_metric(observed, route) - _metric(expected, route)),
            )
    for field, route in MACRO_FIELDS.items():
        expected = sum(
            _metric(detail, route) for detail in expected_details.values()
        ) / len(expected_details)
        maximum_difference = max(
            maximum_difference, abs(float(portfolio[field]) - expected)
        )

    model_map = {}
    recorded_files = set()
    frozen_root = Path(portfolio["frozen_artifact_root"]).resolve()
    for model in portfolio["models"]:
        key = (str(model["dataset"]), int(model["seed"]), str(model["source"]))
        if key in model_map:
            raise RuntimeError(f"duplicate {EXPERIMENT} model {key}")
        family = str(model["physical_family"])
        source_name, row = winners[family]
        if model.get("portfolio_source") != source_name:
            raise RuntimeError(f"{key}: model source differs")
        expected = _source_model(
            source_name, sources[source_name], row, family, key[0], key[1], key[2]
        )
        for field in ("result", "checkpoint"):
            path = Path(model[field]).resolve()
            try:
                path.relative_to(frozen_root)
            except ValueError as error:
                raise RuntimeError(f"{key}: frozen {field} escaped root") from error
            expected_hash = str(expected[f"{field}_sha256"])
            if not path.is_file() or _sha256(path) != expected_hash:
                raise RuntimeError(f"{key}: frozen {field} hash differs")
            if path in recorded_files:
                raise RuntimeError(f"duplicate frozen file: {path}")
            recorded_files.add(path)
        for field in ("fmt_dim", "parameter_count", "trainable_residual_parameter_count"):
            if int(model[field]) != int(expected[field]):
                raise RuntimeError(f"{key}: {field} differs")
        model_map[key] = model
    expected_keys = {
        (dataset, seed, arm)
        for datasets in reference.values() for dataset in datasets
        for seed in [40, 41] for arm in ARMS
    }
    if set(model_map) != expected_keys:
        raise RuntimeError(f"{EXPERIMENT} frozen model keys differ")
    actual_files = {path.resolve() for path in frozen_root.rglob("*") if path.is_file()}
    if actual_files != recorded_files or len(recorded_files) != 80:
        raise RuntimeError(f"{EXPERIMENT} frozen artifact tree differs")

    result = {
        "status": "passed" if maximum_difference <= 1e-12 else "failed",
        "independent_of_portfolio_selector_implementation": True,
        "counts": {
            "datasets": 10, "families": len(reference), "sources": 2,
            "frozen_models": len(model_map),
            "frozen_artifact_files": len(recorded_files),
        },
        "selected_source_by_family": {
            family: value[0] for family, value in winners.items()
        },
        "maximum_absolute_difference_vs_portfolio": maximum_difference,
        "all_frozen_hashes_verified": True,
        "all_paired_parameter_counts_equal": all(
            all(
                int(model_map[(dataset, seed, "fmt")][field])
                == int(model_map[(dataset, seed, "raw_pca")][field])
                for field in (
                    "fmt_dim", "parameter_count",
                    "trainable_residual_parameter_count",
                )
            )
            for datasets in reference.values() for dataset in datasets
            for seed in [40, 41]
        ),
        "input_sha256": {
            "config": _sha256(config_path),
            "portfolio_selection": _sha256(portfolio_path),
        },
    }
    if not result["all_paired_parameter_counts_equal"]:
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
    output = arguments.output or arguments.artifact_dir / "independent_audit.json"
    result = audit(
        arguments.config.resolve(), arguments.artifact_dir.resolve(), output.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
