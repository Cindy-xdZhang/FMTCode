"""Freeze the guarded Task3 winner between portfolios 54.1 and 55.1.

This selector trains nothing and reads development artifacts only. It was
preregistered before 55.1 produced a performance metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import yaml

from Search_Task3_FMTResidual_3D import _read_csv


EXPERIMENT = "Verify_Task3_AuxiliaryLearningRatePortfolio_56.1"
SOURCE_NAMES = {"current_portfolio", "auxiliary_learning_rate"}
ARMS = ("fmt", "raw_pca")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "task", "output_root", "confirmation_opened",
        "sources", "datasets", "source_paired_seeds",
        "frozen_confirmation_seeds", "selection",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing Task3 56.1 config keys: {missing}")
    if spec["experiment"] != EXPERIMENT:
        raise ValueError("Task3 56.1 experiment identity changed")
    if spec["task"] != "Task3" or bool(spec["confirmation_opened"]):
        raise ValueError("Task3 56.1 must remain development-only")
    if set(spec["sources"]) != SOURCE_NAMES:
        raise ValueError("Task3 56.1 source set changed")
    datasets = [str(value) for value in spec["datasets"]]
    if len(datasets) != 10 or len(set(datasets)) != 10:
        raise ValueError("Task3 56.1 requires ten unique datasets")
    if [int(value) for value in spec["source_paired_seeds"]] != [40, 41, 42]:
        raise ValueError("Task3 56.1 source seeds changed")
    if [int(value) for value in spec["frozen_confirmation_seeds"]] != [40, 41]:
        raise ValueError("Task3 56.1 frozen seeds changed")
    selection = dict(spec["selection"])
    metrics = [
        str(selection["primary_metric"]),
        *[str(value) for value in selection["tie_breakers"]],
    ]
    if len(metrics) != len(set(metrics)):
        raise ValueError("Task3 56.1 selection metrics are duplicated")
    if not bool(selection["require_source_absolute_fmt_guard"]):
        raise ValueError("Task3 56.1 requires source FMT guards")
    spec["datasets"] = datasets
    spec["source_paired_seeds"] = [40, 41, 42]
    spec["frozen_confirmation_seeds"] = [40, 41]
    spec["selection_metrics"] = metrics
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    return spec


def static_preflight(config_path: str | Path) -> dict:
    spec = _load_spec(config_path)
    payload = {
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "source_count": 2,
        "source_names": sorted(spec["sources"]),
        "dataset_count": len(spec["datasets"]),
        "selection_metrics": spec["selection_metrics"],
        "training_runs": 0,
        "confirmation_opened": False,
        "performance_artifacts_read": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def source_identity_preflight(config_path: str | Path) -> Path:
    """Validate deployed source configs without opening performance files."""
    spec = _load_spec(config_path)
    sources = {}
    for name, section in sorted(spec["sources"].items()):
        root = Path(section["repo_root"])
        config = _under(root, section["paths"]["config"])
        if not config.is_file():
            raise FileNotFoundError(config)
        actual = _canonical_sha256(config)
        expected = str(section["expected_config_canonical_sha256"])
        if actual != expected:
            raise RuntimeError(f"{name}: deployed config hash changed")
        overlay = yaml.safe_load(config.read_text(encoding="utf-8"))
        if str(overlay.get("experiment")) != str(section["expected_experiment"]):
            raise RuntimeError(f"{name}: deployed experiment changed")
        sources[name] = {
            "repo_root": str(root),
            "config": str(config),
            "experiment": overlay["experiment"],
            "canonical_sha256": actual,
        }
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "confirmation_opened": False,
        "performance_artifacts_read": False,
        "sources": sources,
    }
    target = Path(spec["output_root"]) / "source_identity_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _load_json(target) != payload:
        raise RuntimeError("Task3 56.1 source identity preflight changed")
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(target.read_text(encoding="utf-8"), end="")
    return target


def _family_map(selection: dict) -> dict[str, list[str]]:
    result = {}
    for family, row in dict(selection["primary_by_group"]).items():
        details = json.loads(str(row["datasets_json"]))
        if not isinstance(details, dict) or not details:
            raise RuntimeError(f"{family}: invalid datasets_json")
        result[str(family)] = [str(value) for value in details]
    return result


def _validate_source_config(name: str, section: dict) -> tuple[Path, Path, dict]:
    root = Path(section["repo_root"])
    config = _under(root, section["paths"]["config"])
    if not config.is_file():
        raise FileNotFoundError(config)
    if _canonical_sha256(config) != str(section["expected_config_canonical_sha256"]):
        raise RuntimeError(f"{name}: source config hash changed")
    overlay = yaml.safe_load(config.read_text(encoding="utf-8"))
    if str(overlay.get("experiment")) != str(section["expected_experiment"]):
        raise RuntimeError(f"{name}: source experiment changed")
    return root, config, overlay


def _load_current_source(section: dict) -> dict:
    root, config, overlay = _validate_source_config("current_portfolio", section)
    paths = {key: _under(root, value) for key, value in section["paths"].items()}
    for key in ("selection", "audit"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    selection = _load_json(paths["selection"])
    audit = _load_json(paths["audit"])
    if selection.get("experiment") != overlay["experiment"]:
        raise RuntimeError("54.1 portfolio experiment changed")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("54.1 portfolio opened confirmation")
    if audit.get("status") != "passed" or not audit.get("all_frozen_hashes_verified"):
        raise RuntimeError("54.1 independent audit did not pass")
    if str(audit["input_sha256"]["config"]) != _sha256(config):
        raise RuntimeError("54.1 audit/config hash mismatch")
    if str(audit["input_sha256"]["portfolio_selection"]) != _sha256(paths["selection"]):
        raise RuntimeError("54.1 audit/selection hash mismatch")
    frozen_root = paths["frozen_root"].resolve()
    models = {}
    for item in list(selection.get("models", [])):
        key = (str(item["dataset"]), int(item["seed"]), str(item["source"]))
        if key in models:
            raise RuntimeError(f"duplicate 54.1 model {key}")
        for field in ("result", "checkpoint"):
            path = Path(item[field]).resolve()
            try:
                path.relative_to(frozen_root)
            except ValueError as error:
                raise RuntimeError(f"54.1 {field} escaped frozen root") from error
            if not path.is_file() or _sha256(path) != str(item[f"{field}_sha256"]):
                raise RuntimeError(f"54.1 frozen {field} changed: {key}")
        models[key] = dict(item)
    if len(models) != 40:
        raise RuntimeError("54.1 frozen model set is incomplete")
    return {
        "name": "current_portfolio", "root": root, "config": config,
        "overlay": overlay, "selection": selection, "audit": audit,
        "models": models,
        "hashes": {
            "config": _sha256(config), "selection": _sha256(paths["selection"]),
            "audit": _sha256(paths["audit"]),
        },
    }


def _load_auxiliary_source(section: dict) -> dict:
    root, config, overlay = _validate_source_config(
        "auxiliary_learning_rate", section
    )
    paths = {key: _under(root, value) for key, value in section["paths"].items()}
    for key in ("preflight", "selection", "audit", "evidence", "archive"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    preflight = _load_json(paths["preflight"])
    selection = _load_json(paths["selection"])
    audit = _load_json(paths["audit"])
    evidence = _load_json(paths["evidence"])
    expected = str(overlay["experiment"])
    if any(str(value.get("experiment")) != expected for value in (preflight, selection)):
        raise RuntimeError("55.1 source experiment changed")
    if bool(preflight.get("confirmation_opened", True)) or bool(
        selection.get("confirmation_opened", True)
    ):
        raise RuntimeError("55.1 source opened confirmation")
    if audit.get("status") != "passed" or not audit.get(
        "all_source_hashes_consistent"
    ):
        raise RuntimeError("55.1 independent audit did not pass")
    expected_inputs = {
        "optimization_selection": _sha256(paths["selection"]),
        "preflight_manifest": _sha256(paths["preflight"]),
        "per_run_csv_archive": _sha256(paths["archive"]),
    }
    if any(str(audit["input_sha256"].get(key)) != value
           for key, value in expected_inputs.items()):
        raise RuntimeError("55.1 audit input hash mismatch")
    if evidence.get("status") != "passed" or int(
        evidence.get("archived_per_run_csv", -1)
    ) != 540:
        raise RuntimeError("55.1 evidence archive is incomplete")
    if str(evidence.get("stable_archive_sha256")) != _sha256(paths["archive"]):
        raise RuntimeError("55.1 evidence archive hash changed")
    return {
        "name": "auxiliary_learning_rate", "root": root, "config": config,
        "overlay": overlay, "selection": selection, "audit": audit,
        "candidate_root": paths["candidate_root"],
        "hashes": {
            "config": _sha256(config), "preflight": _sha256(paths["preflight"]),
            "selection": _sha256(paths["selection"]), "audit": _sha256(paths["audit"]),
            "evidence": _sha256(paths["evidence"]), "archive": _sha256(paths["archive"]),
        },
    }


def _score(row: dict, metrics: list[str], source: str, family: str) -> tuple:
    if not bool(row.get("eligible", True)):
        raise RuntimeError(f"{source}/{family}: source winner is ineligible")
    if row.get("absolute_fmt_guard_passed") is False:
        raise RuntimeError(f"{source}/{family}: source FMT guard failed")
    return tuple(_finite(row[metric], f"{source}/{family}/{metric}") for metric in metrics)


def _auxiliary_model(source: dict, row: dict, dataset: str,
                     seed: int, arm: str, family: str) -> dict:
    candidate = str(row["optimization_id"])
    result_path = (
        source["candidate_root"] / candidate / dataset / f"seed{seed}"
        / arm / "per_run.csv"
    )
    rows = _read_csv(result_path)
    if len(rows) != 1:
        raise RuntimeError(f"55.1 missing result {result_path}")
    result = rows[0]
    expected_variant = "raw_fmt_residual" if arm == "fmt" else "raw_pca_residual"
    expected = {
        "dataset": dataset, "seed": seed, "variant": expected_variant,
        "auxiliary_source": arm, "optimization_id": candidate,
    }
    for key, value in expected.items():
        if str(result.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"55.1 result changed: {key}")
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = source["root"] / checkpoint
    if not checkpoint.is_file() or checkpoint.resolve().parent != (
        result_path.parent / "checkpoints"
    ).resolve():
        raise RuntimeError(f"55.1 invalid checkpoint {checkpoint}")
    return {
        "dataset": dataset, "physical_family": family,
        "source_search": source["name"],
        "source_experiment": source["overlay"]["experiment"],
        "source_selection_sha256": source["hashes"]["selection"],
        "seed": seed, "source": arm, "variant": expected_variant,
        "candidate_id": candidate,
        "fmt_feature": str(result["fmt_feature"]),
        "fmt_dim": int(result["fmt_dim"]),
        "parameter_count": int(result["parameter_count"]),
        "trainable_residual_parameter_count": int(
            result["trainable_residual_parameter_count"]
        ),
        "result": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
    }


def _copy_verified(source: Path, target: Path, expected: str) -> None:
    if not source.is_file() or _sha256(source) != expected:
        raise RuntimeError(f"source artifact changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _sha256(target) != expected:
        raise RuntimeError(f"frozen artifact collision: {target}")
    if not target.exists():
        shutil.copy2(source, target)
    if _sha256(target) != expected:
        raise RuntimeError(f"frozen copy failed: {target}")


def _freeze_model(item: dict, frozen_root: Path, portfolio_source: str) -> dict:
    relative = Path(item["dataset"]) / f"seed{item['seed']}" / item["source"]
    source_result = Path(item["result"])
    source_checkpoint = Path(item["checkpoint"])
    target_result = frozen_root / relative / "per_run.csv"
    target_checkpoint = frozen_root / relative / source_checkpoint.name
    _copy_verified(source_result, target_result, item["result_sha256"])
    _copy_verified(source_checkpoint, target_checkpoint, item["checkpoint_sha256"])
    result = dict(item)
    result.update({
        "portfolio_source": portfolio_source,
        "source_result": str(source_result),
        "source_checkpoint": str(source_checkpoint),
        "result": str(target_result.resolve()),
        "checkpoint": str(target_checkpoint.resolve()),
    })
    return result


def select(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    source_id = Path(spec["output_root"]) / "source_identity_preflight.json"
    if not source_id.is_file():
        raise FileNotFoundError(source_id)
    identity = _load_json(source_id)
    if identity.get("config_sha256") != spec["config_sha256"]:
        raise RuntimeError("56.1 source identity/config hash mismatch")

    current = _load_current_source(spec["sources"]["current_portfolio"])
    auxiliary = _load_auxiliary_source(
        spec["sources"]["auxiliary_learning_rate"]
    )
    sources = {current["name"]: current, auxiliary["name"]: auxiliary}
    family_maps = {
        name: _family_map(source["selection"])
        for name, source in sources.items()
    }
    reference = family_maps["current_portfolio"]
    if family_maps["auxiliary_learning_rate"] != reference:
        raise RuntimeError("56.1 source family maps differ")
    flattened = [dataset for values in reference.values() for dataset in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(spec["datasets"]):
        raise RuntimeError("56.1 family map does not partition datasets")

    winners = {}
    public_primary = {}
    for family in reference:
        candidates = []
        for name, source in sources.items():
            row = dict(source["selection"]["primary_by_group"][family])
            candidates.append((_score(row, spec["selection_metrics"], name, family), name, row))
        _, name, row = max(candidates, key=lambda value: value[0])
        winners[family] = (name, row)
        public_primary[family] = {
            **row,
            "portfolio_source": name,
            "source_experiment": sources[name]["overlay"]["experiment"],
            "source_selection_sha256": sources[name]["hashes"]["selection"],
        }

    frozen_root = Path(spec["output_root"]) / "frozen_artifacts"
    models = []
    for family, datasets in reference.items():
        source_name, row = winners[family]
        source = sources[source_name]
        for dataset in datasets:
            for seed in spec["frozen_confirmation_seeds"]:
                paired = []
                for arm in ARMS:
                    if source_name == "current_portfolio":
                        item = dict(source["models"][(dataset, seed, arm)])
                        item["physical_family"] = family
                    else:
                        item = _auxiliary_model(
                            source, row, dataset, seed, arm, family
                        )
                    frozen = _freeze_model(item, frozen_root, source_name)
                    models.append(frozen)
                    paired.append(frozen)
                for field in (
                    "fmt_dim", "parameter_count", "trainable_residual_parameter_count"
                ):
                    if paired[0][field] != paired[1][field]:
                        raise RuntimeError(f"{dataset}/seed{seed}: paired {field} differs")
    if len(models) != 40:
        raise RuntimeError("56.1 must freeze exactly 40 models")

    dataset_details = []
    for family, row in public_primary.items():
        details = json.loads(str(row["datasets_json"]))
        for dataset, metrics in details.items():
            dataset_details.append({
                "physical_family": family, "dataset": dataset,
                "portfolio_source": row["portfolio_source"],
                "optimization_id": row["optimization_id"], **metrics,
            })
    mean = lambda values: float(sum(values) / len(values))
    f1_gain = mean([float(row["f1_gain"]) for row in dataset_details])
    ap_gain = mean([float(row["average_precision_gain"]) for row in dataset_details])
    fmt_f1 = mean([float(row["fmt"]["f1"]) for row in dataset_details])
    raw_f1 = mean([float(row["raw_pca"]["f1"]) for row in dataset_details])
    fmt_ap = mean([float(row["fmt"]["average_precision"]) for row in dataset_details])
    raw_ap = mean([float(row["raw_pca"]["average_precision"]) for row in dataset_details])
    target_gain = float(spec["selection"]["target_dataset_macro_f1_gain"])
    target_fmt = float(spec["selection"]["target_absolute_fmt_f1"])
    payload = {
        "schema": 1, "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "selection_rule": (
            "per physical family, maximize guarded development paired F1 gain "
            "between audited 54.1 and preregistered 55.1 winners; "
            f"tie breakers={spec['selection_metrics'][1:]}"
        ),
        "training_runs": 0, "confirmation_opened": False,
        "source_paired_seeds": spec["source_paired_seeds"],
        "frozen_confirmation_seeds": spec["frozen_confirmation_seeds"],
        "source_artifact_sha256": {
            name: source["hashes"] for name, source in sources.items()
        },
        "family_datasets": reference, "primary_by_group": public_primary,
        "models": models, "dataset_details": dataset_details,
        "frozen_model_count": len(models),
        "frozen_artifact_file_count": 2 * len(models),
        "frozen_artifact_root": str(frozen_root.resolve()),
        "development_dataset_macro_f1_gain_vs_raw_pca": f1_gain,
        "development_dataset_macro_ap_gain_vs_raw_pca": ap_gain,
        "development_dataset_macro_fmt_f1": fmt_f1,
        "development_dataset_macro_raw_pca_f1": raw_f1,
        "development_dataset_macro_fmt_average_precision": fmt_ap,
        "development_dataset_macro_raw_pca_average_precision": raw_ap,
        "target_dataset_macro_f1_gain": target_gain,
        "target_reached": f1_gain >= target_gain,
        "target_absolute_fmt_f1": target_fmt,
        "absolute_fmt_target_reached": fmt_f1 >= target_fmt,
        "joint_target_reached": f1_gain >= target_gain and fmt_f1 >= target_fmt,
    }
    target = Path(spec["output_root"]) / "portfolio_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"), end="")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=("static-preflight", "source-identity-preflight", "select"),
        required=True,
    )
    arguments = parser.parse_args()
    if arguments.mode == "static-preflight":
        static_preflight(arguments.config)
    elif arguments.mode == "source-identity-preflight":
        source_identity_preflight(arguments.config)
    else:
        select(arguments.config)


if __name__ == "__main__":
    main()
