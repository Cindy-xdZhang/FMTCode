"""Independently audit a frozen Task3 adaptive portfolio.

The audit deliberately does not import either portfolio selector.  It reads
the registered source selections, reconstructs the physical-family winners,
recomputes all dataset-macro metrics, and verifies every frozen result and
checkpoint by content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
ARMS = ("fmt", "raw_pca")
MACRO_FIELDS = {
    "development_dataset_macro_f1_gain_vs_raw_pca": "f1_gain",
    "development_dataset_macro_ap_gain_vs_raw_pca": (
        "average_precision_gain"
    ),
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


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a YAML mapping")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"cannot parse boolean {value!r}")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _source_state(name: str, section: dict[str, Any]) -> dict[str, Any]:
    root = Path(section["repo_root"])
    paths = {
        key: _under(root, value)
        for key, value in dict(section["paths"]).items()
    }
    required = {"config", "preflight", "selection"}
    if set(paths) != required:
        raise RuntimeError(f"{name}: source paths changed: {sorted(paths)}")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if _canonical_sha256(paths["config"]) != str(
        section["expected_config_canonical_sha256"]
    ):
        raise RuntimeError(f"{name}: canonical config hash changed")

    overlay = _load_yaml(paths["config"])
    preflight = _load_json(paths["preflight"])
    selection = _load_json(paths["selection"])
    expected_experiment = str(section["expected_experiment"])
    if any(
        str(payload.get("experiment")) != expected_experiment
        for payload in (overlay, preflight, selection)
    ):
        raise RuntimeError(f"{name}: source experiment identity changed")
    if _as_bool(preflight.get("confirmation_opened", True)):
        raise RuntimeError(f"{name}: preflight opened confirmation")
    if _as_bool(selection.get("confirmation_opened", True)):
        raise RuntimeError(f"{name}: selection opened confirmation")
    if selection.get("absolute_fmt_guard") is None:
        raise RuntimeError(f"{name}: source lacks an absolute FMT guard")

    hashes = {key: _sha256(path) for key, path in paths.items()}
    if str(preflight.get("optimization_config_sha256", "")).lower() != (
        hashes["config"]
    ):
        raise RuntimeError(f"{name}: preflight/config hash mismatch")
    if str(selection.get("optimization_config_sha256", "")).lower() != (
        hashes["config"]
    ):
        raise RuntimeError(f"{name}: selection/config hash mismatch")
    if str(selection.get("preflight_manifest_sha256", "")).lower() != (
        hashes["preflight"]
    ):
        raise RuntimeError(f"{name}: selection/preflight hash mismatch")
    return {
        "name": name,
        "experiment": expected_experiment,
        "selection": selection,
        "hashes": hashes,
    }


def _family_datasets(selection: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for family, row in dict(selection["primary_by_group"]).items():
        details = json.loads(str(row["datasets_json"]))
        if not isinstance(details, dict) or not details:
            raise RuntimeError(f"{family}: invalid datasets_json")
        result[str(family)] = [str(value) for value in details]
    return result


def _dataset_details(row: dict[str, Any]) -> dict[str, Any]:
    details = json.loads(str(row["datasets_json"]))
    if not isinstance(details, dict):
        raise TypeError("datasets_json must contain a JSON object")
    return {str(key): value for key, value in details.items()}


def _metric(detail: dict[str, Any], route: str | tuple[str, str]) -> float:
    if isinstance(route, tuple):
        return _finite(detail[route[0]][route[1]], "/".join(route))
    return _finite(detail[route], route)


def audit(config_path: Path, artifact_dir: Path, output_path: Path) -> dict:
    config = _load_yaml(config_path)
    if _as_bool(config.get("confirmation_opened", True)):
        raise RuntimeError("portfolio config opened confirmation")
    source_sections = dict(config["sources"])
    sources = {
        name: _source_state(str(name), dict(section))
        for name, section in source_sections.items()
    }
    selection_spec = dict(config["selection"])
    if not _as_bool(selection_spec["require_source_absolute_fmt_guard"]):
        raise RuntimeError("portfolio does not require source guards")
    selection_metrics = [
        str(selection_spec["primary_metric"]),
        *[str(value) for value in selection_spec["tie_breakers"]],
    ]
    if len(selection_metrics) != len(set(selection_metrics)):
        raise RuntimeError("portfolio selection metrics are duplicated")

    family_maps = {
        name: _family_datasets(source["selection"])
        for name, source in sources.items()
    }
    reference = next(iter(family_maps.values()))
    if any(value != reference for value in family_maps.values()):
        raise RuntimeError("source family/dataset maps differ")
    datasets = [dataset for values in reference.values() for dataset in values]
    registered = [str(value) for value in config["datasets"]]
    if len(datasets) != len(set(datasets)) or set(datasets) != set(registered):
        raise RuntimeError("source family map does not partition datasets")

    winners: dict[str, dict[str, Any]] = {}
    for family in reference:
        candidates = []
        for source_name, source in sources.items():
            row = dict(source["selection"]["primary_by_group"][family])
            if not _as_bool(row.get("eligible", True)):
                raise RuntimeError(f"{source_name}/{family}: winner ineligible")
            if row.get("absolute_fmt_guard_passed") is False:
                raise RuntimeError(f"{source_name}/{family}: FMT guard failed")
            score = tuple(
                _finite(row[metric], f"{source_name}/{family}/{metric}")
                for metric in selection_metrics
            )
            candidates.append((score, source_name, source, row))
        _, source_name, source, row = max(candidates, key=lambda item: item[0])
        winners[family] = {
            "source_name": source_name,
            "source": source,
            "row": row,
            "datasets": _dataset_details(row),
        }

    portfolio_path = artifact_dir / "portfolio_selection.json"
    if not portfolio_path.is_file():
        raise FileNotFoundError(portfolio_path)
    portfolio = _load_json(portfolio_path)
    if str(portfolio.get("experiment")) != str(config["experiment"]):
        raise RuntimeError("portfolio experiment differs from config")
    if _as_bool(portfolio.get("confirmation_opened", True)):
        raise RuntimeError("portfolio opened confirmation")
    if int(portfolio.get("training_runs", -1)) != 0:
        raise RuntimeError("portfolio unexpectedly trained models")
    if str(portfolio.get("config_sha256", "")).lower() != _sha256(config_path):
        raise RuntimeError("portfolio/config hash mismatch")

    expected_source_hashes = {
        name: source["hashes"] for name, source in sources.items()
    }
    if portfolio.get("source_artifact_sha256") != expected_source_hashes:
        raise RuntimeError("portfolio source artifact hashes differ")

    maximum_difference = 0.0
    reported_primary = dict(portfolio["primary_by_group"])
    if set(reported_primary) != set(winners):
        raise RuntimeError("portfolio family keys differ")
    expected_details: dict[str, dict[str, Any]] = {}
    dataset_family: dict[str, str] = {}
    for family, winner in winners.items():
        reported = dict(reported_primary[family])
        source = winner["source"]
        expected_identity = {
            "portfolio_source": winner["source_name"],
            "source_experiment": source["experiment"],
            "source_selection_sha256": source["hashes"]["selection"],
            "optimization_id": str(winner["row"]["optimization_id"]),
        }
        for key, value in expected_identity.items():
            if str(reported.get(key)) != str(value):
                raise RuntimeError(f"{family}: portfolio {key} differs")
        if _dataset_details(reported) != winner["datasets"]:
            raise RuntimeError(f"{family}: portfolio datasets_json differs")
        for metric in selection_metrics:
            maximum_difference = max(
                maximum_difference,
                abs(_finite(reported[metric], metric)
                    - _finite(winner["row"][metric], metric)),
            )
        for dataset, detail in winner["datasets"].items():
            expected_details[dataset] = detail
            dataset_family[dataset] = family

    for field, route in MACRO_FIELDS.items():
        independent = _mean([
            _metric(detail, route) for detail in expected_details.values()
        ])
        maximum_difference = max(
            maximum_difference,
            abs(independent - _finite(portfolio[field], field)),
        )

    reported_details = {
        str(row["dataset"]): row for row in portfolio["dataset_details"]
    }
    if set(reported_details) != set(expected_details):
        raise RuntimeError("portfolio dataset detail keys differ")
    for dataset, detail in expected_details.items():
        row = reported_details[dataset]
        winner = winners[dataset_family[dataset]]
        if str(row["physical_family"]) != dataset_family[dataset]:
            raise RuntimeError(f"{dataset}: physical family differs")
        if str(row["portfolio_source"]) != winner["source_name"]:
            raise RuntimeError(f"{dataset}: portfolio source differs")
        if str(row["optimization_id"]) != str(
            winner["row"]["optimization_id"]
        ):
            raise RuntimeError(f"{dataset}: optimization ID differs")
        for route in MACRO_FIELDS.values():
            expected_value = _metric(detail, route)
            if isinstance(route, tuple):
                observed = _finite(row[route[0]][route[1]], str(route))
            else:
                observed = _finite(row[route], route)
            maximum_difference = max(
                maximum_difference, abs(expected_value - observed)
            )

    seeds = [int(value) for value in config["frozen_confirmation_seeds"]]
    expected_model_keys = {
        (dataset, seed, arm)
        for dataset in datasets for seed in seeds for arm in ARMS
    }
    models = list(portfolio["models"])
    model_map: dict[tuple[str, int, str], dict[str, Any]] = {}
    frozen_root = (artifact_dir / "frozen_artifacts").resolve()
    recorded_files: set[Path] = set()
    for model in models:
        key = (
            str(model["dataset"]), int(model["seed"]), str(model["source"])
        )
        if key in model_map:
            raise RuntimeError(f"duplicate frozen model {key}")
        model_map[key] = model
        dataset, seed, arm = key
        family = dataset_family[dataset]
        winner = winners[family]
        expected_variant = (
            "raw_fmt_residual" if arm == "fmt" else "raw_pca_residual"
        )
        expected_identity = {
            "physical_family": family,
            "source_search": winner["source_name"],
            "source_experiment": winner["source"]["experiment"],
            "source_selection_sha256": winner["source"]["hashes"]["selection"],
            "candidate_id": str(winner["row"]["optimization_id"]),
            "variant": expected_variant,
        }
        for field, expected in expected_identity.items():
            if str(model.get(field)) != str(expected):
                raise RuntimeError(f"{key}: frozen model {field} differs")
        for field in ("result", "checkpoint"):
            path = Path(model[field]).resolve()
            if not path.is_file() or not _is_under(path, frozen_root):
                raise RuntimeError(f"{key}: invalid frozen {field} path")
            expected_hash = str(model[f"{field}_sha256"]).lower()
            if _sha256(path) != expected_hash:
                raise RuntimeError(f"{key}: frozen {field} hash differs")
            if path in recorded_files:
                raise RuntimeError(f"duplicate frozen artifact path: {path}")
            recorded_files.add(path)
    if set(model_map) != expected_model_keys:
        raise RuntimeError("portfolio frozen model keys differ")
    actual_files = {path.resolve() for path in frozen_root.rglob("*") if path.is_file()}
    if actual_files != recorded_files:
        raise RuntimeError("frozen artifact tree differs from model records")
    for dataset in datasets:
        for seed in seeds:
            paired = [model_map[(dataset, seed, arm)] for arm in ARMS]
            for field in (
                "fmt_dim", "parameter_count",
                "trainable_residual_parameter_count",
            ):
                if int(paired[0][field]) != int(paired[1][field]):
                    raise RuntimeError(
                        f"{dataset}/seed{seed}: paired {field} differs"
                    )
    if int(portfolio.get("frozen_model_count", -1)) != len(expected_model_keys):
        raise RuntimeError("frozen_model_count differs")
    if int(portfolio.get("frozen_artifact_file_count", -1)) != len(recorded_files):
        raise RuntimeError("frozen_artifact_file_count differs")

    result = {
        "status": "passed" if maximum_difference <= 1e-12 else "failed",
        "independent_of_portfolio_selector_implementation": True,
        "counts": {
            "datasets": len(datasets),
            "families": len(reference),
            "frozen_artifact_files": len(recorded_files),
            "frozen_models": len(models),
            "sources": len(sources),
        },
        "selected_source_by_family": {
            family: winner["source_name"]
            for family, winner in winners.items()
        },
        "maximum_absolute_difference_vs_portfolio": maximum_difference,
        "all_frozen_hashes_verified": True,
        "all_paired_parameter_counts_equal": True,
        "input_sha256": {
            "config": _sha256(config_path),
            "portfolio_selection": _sha256(portfolio_path),
        },
    }
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
    config_path = arguments.config.resolve()
    artifact_dir = arguments.artifact_dir.resolve()
    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else artifact_dir / "independent_audit.json"
    )
    result = audit(config_path, artifact_dir, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
