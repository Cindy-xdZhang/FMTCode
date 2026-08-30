"""Select and freeze the strongest guarded Task3 family recipes.

No model is trained here.  The selector compares the completed development
selectors 44.1, 45.1, and 48.1 under their common metrics, then records all
selected seed-40/41 result and checkpoint hashes before confirmation data may
be generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from Search_Task3_FMTResidual_3D import _read_csv


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_text_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _under(root: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else root / value


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
        raise ValueError(f"missing Task3 49.1 config keys: {missing}")
    if spec["experiment"] != "Verify_Task3_FinalPortfolio_49.1":
        raise ValueError("Task3 49.1 experiment identity changed")
    if spec["task"] != "Task3" or bool(spec["confirmation_opened"]):
        raise ValueError("Task3 49.1 must remain development-only")
    datasets = [str(value) for value in spec["datasets"]]
    if len(datasets) != 10 or len(set(datasets)) != 10:
        raise ValueError("Task3 49.1 requires ten unique datasets")
    if [int(value) for value in spec["source_paired_seeds"]] != [40, 41, 42]:
        raise ValueError("Task3 49.1 source seeds changed")
    if [int(value) for value in spec["frozen_confirmation_seeds"]] != [40, 41]:
        raise ValueError("Task3 49.1 frozen confirmation seeds changed")
    if set(spec["sources"]) != {"safe_factor", "head_alpha_clip", "full_stack"}:
        raise ValueError("Task3 49.1 source portfolio changed")
    selection = dict(spec["selection"])
    required_metrics = [
        str(selection["primary_metric"]),
        *[str(value) for value in selection["tie_breakers"]],
    ]
    if len(required_metrics) != len(set(required_metrics)):
        raise ValueError("Task3 49.1 selection metrics are duplicated")
    if not bool(selection["require_source_absolute_fmt_guard"]):
        raise ValueError("Task3 49.1 requires guarded source winners")
    spec["datasets"] = datasets
    spec["source_paired_seeds"] = [40, 41, 42]
    spec["frozen_confirmation_seeds"] = [40, 41]
    spec["selection_metrics"] = required_metrics
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    return spec


def static_preflight(config_path: str | Path) -> dict:
    spec = _load_spec(config_path)
    payload = {
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "source_count": len(spec["sources"]),
        "source_names": sorted(spec["sources"]),
        "dataset_count": len(spec["datasets"]),
        "source_paired_seeds": spec["source_paired_seeds"],
        "frozen_confirmation_seeds": spec["frozen_confirmation_seeds"],
        "selection_metrics": spec["selection_metrics"],
        "training_runs": 0,
        "confirmation_opened": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _source_state(name: str, section: dict, spec: dict) -> dict:
    root = Path(section["repo_root"])
    paths = {
        key: _under(root, value) for key, value in section["paths"].items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if _canonical_text_sha256(paths["config"]) != str(
        section["expected_config_canonical_sha256"]
    ):
        raise RuntimeError(f"{name}: source config changed")
    overlay = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    expected = str(section["expected_experiment"])
    for artifact_name, payload in (
        ("config", overlay), ("preflight", preflight),
        ("selection", selection),
    ):
        if str(payload.get("experiment")) != expected:
            raise RuntimeError(f"{name}: {artifact_name} experiment changed")
    if bool(preflight.get("confirmation_opened", True)):
        raise RuntimeError(f"{name}: preflight opened confirmation")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError(f"{name}: selection opened confirmation")
    config_hash = _sha256(paths["config"])
    preflight_hash = _sha256(paths["preflight"])
    if str(preflight.get("optimization_config_sha256", "")).lower() != config_hash:
        raise RuntimeError(f"{name}: preflight/config hash mismatch")
    if str(selection.get("optimization_config_sha256", "")).lower() != config_hash:
        raise RuntimeError(f"{name}: selection/config hash mismatch")
    if str(selection.get("preflight_manifest_sha256", "")).lower() != preflight_hash:
        raise RuntimeError(f"{name}: selection/preflight hash mismatch")
    if [int(value) for value in selection.get("paired_seeds", [])] != (
        spec["source_paired_seeds"]
    ):
        raise RuntimeError(f"{name}: paired seeds changed")
    guard = selection.get("absolute_fmt_guard")
    if spec["selection"]["require_source_absolute_fmt_guard"] and guard is None:
        raise RuntimeError(f"{name}: source lacks absolute FMT guard")
    return {
        "name": name,
        "root": root,
        "paths": paths,
        "overlay": overlay,
        "preflight": preflight,
        "selection": selection,
        "artifact_sha256": {
            key: _sha256(path) for key, path in paths.items()
        },
    }


def _family_datasets(selection: dict) -> dict[str, list[str]]:
    result = {}
    for family, row in selection["primary_by_group"].items():
        datasets = json.loads(str(row["datasets_json"]))
        result[str(family)] = list(datasets)
    return result


def _result_path(source: dict, candidate_id: str, dataset: str,
                 seed: int, arm: str) -> Path:
    return (
        _under(source["root"], source["overlay"]["output_root"])
        / "candidates" / candidate_id / dataset / f"seed{int(seed)}" / arm
        / "per_run.csv"
    )


def _freeze_models(spec: dict, selected: dict,
                   family_datasets: dict[str, list[str]]) -> list[dict]:
    models = []
    for family, choice in selected.items():
        source = choice["_source_state"]
        row = choice["_source_row"]
        candidate_id = str(row["optimization_id"])
        recipe = json.loads(str(row["optimization_recipe_json"]))
        feature = str(recipe["fmt_feature"])
        for dataset in family_datasets[family]:
            for seed in spec["frozen_confirmation_seeds"]:
                paired = []
                for arm, variant in (
                    ("fmt", "raw_fmt_residual"),
                    ("raw_pca", "raw_pca_residual"),
                ):
                    result_path = _result_path(
                        source, candidate_id, dataset, seed, arm
                    )
                    rows = _read_csv(result_path)
                    if len(rows) != 1:
                        raise RuntimeError(
                            f"{source['name']}: missing result {result_path}"
                        )
                    result = rows[0]
                    expected = {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "auxiliary_source": arm,
                        "optimization_id": candidate_id,
                        "fmt_feature": feature,
                        "optimization_config_sha256": source[
                            "artifact_sha256"
                        ]["config"],
                        "preflight_manifest_sha256": source[
                            "artifact_sha256"
                        ]["preflight"],
                    }
                    for key, value in expected.items():
                        if str(result.get(key, "")).lower() != str(value).lower():
                            raise RuntimeError(
                                f"{source['name']}: result changed: {key}"
                            )
                    if json.loads(str(result["optimization_recipe_json"])) != recipe:
                        raise RuntimeError(
                            f"{source['name']}: selected recipe changed"
                        )
                    checkpoint = Path(result["checkpoint"])
                    if not checkpoint.is_absolute():
                        checkpoint = source["root"] / checkpoint
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    if checkpoint.resolve().parent != (
                        result_path.parent / "checkpoints"
                    ).resolve():
                        raise RuntimeError("checkpoint escaped result directory")
                    item = {
                        "dataset": dataset,
                        "physical_family": family,
                        "source_search": source["name"],
                        "source_experiment": source["overlay"]["experiment"],
                        "source_selection_sha256": source[
                            "artifact_sha256"
                        ]["selection"],
                        "seed": int(seed),
                        "source": arm,
                        "variant": variant,
                        "candidate_id": candidate_id,
                        "fmt_feature": feature,
                        "fmt_dim": int(result["fmt_dim"]),
                        "parameter_count": int(result["parameter_count"]),
                        "trainable_residual_parameter_count": int(
                            result["trainable_residual_parameter_count"]
                        ),
                        "result": str(result_path),
                        "result_sha256": _sha256(result_path),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": _sha256(checkpoint),
                    }
                    models.append(item)
                    paired.append(item)
                for key in (
                    "fmt_dim", "parameter_count",
                    "trainable_residual_parameter_count",
                ):
                    if paired[0][key] != paired[1][key]:
                        raise RuntimeError(
                            f"portfolio paired {key} mismatch: "
                            f"{dataset}/seed{seed}"
                        )
    if len(models) != 40:
        raise RuntimeError("Task3 49.1 must freeze exactly 40 models")
    return models


def select(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    sources = {
        name: _source_state(name, dict(section), spec)
        for name, section in spec["sources"].items()
    }
    family_maps = {
        name: _family_datasets(source["selection"])
        for name, source in sources.items()
    }
    reference = next(iter(family_maps.values()))
    if any(value != reference for value in family_maps.values()):
        raise RuntimeError("portfolio source family/dataset maps differ")
    flattened = [dataset for datasets in reference.values() for dataset in datasets]
    if set(flattened) != set(spec["datasets"]) or len(flattened) != len(set(flattened)):
        raise RuntimeError("portfolio family map does not partition datasets")

    selected = {}
    public_primary = {}
    for family in reference:
        candidates = []
        for source_name, source in sources.items():
            row = dict(source["selection"]["primary_by_group"][family])
            if not bool(row.get("eligible", True)):
                raise RuntimeError(f"{source_name}/{family}: selected row ineligible")
            if row.get("absolute_fmt_guard_passed") is False:
                raise RuntimeError(
                    f"{source_name}/{family}: selected row failed FMT guard"
                )
            for metric in spec["selection_metrics"]:
                if metric not in row or not np.isfinite(float(row[metric])):
                    raise RuntimeError(
                        f"{source_name}/{family}: missing metric {metric}"
                    )
            candidates.append((source_name, source, row))
        candidates.sort(
            key=lambda item: tuple(
                float(item[2][metric]) for metric in spec["selection_metrics"]
            ),
            reverse=True,
        )
        source_name, source, row = candidates[0]
        selected[family] = {
            "_source_state": source,
            "_source_row": row,
        }
        public_primary[family] = {
            **row,
            "portfolio_source": source_name,
            "source_experiment": source["overlay"]["experiment"],
            "source_selection_sha256": source["artifact_sha256"]["selection"],
        }

    models = _freeze_models(spec, selected, reference)
    dataset_details = []
    for family, row in public_primary.items():
        for dataset, metrics in json.loads(str(row["datasets_json"])).items():
            dataset_details.append({
                "physical_family": family,
                "dataset": dataset,
                "portfolio_source": row["portfolio_source"],
                "optimization_id": row["optimization_id"],
                **metrics,
            })
    f1_gain = float(np.mean([row["f1_gain"] for row in dataset_details]))
    ap_gain = float(np.mean([
        row["average_precision_gain"] for row in dataset_details
    ]))
    fmt_f1 = float(np.mean([row["fmt"]["f1"] for row in dataset_details]))
    raw_f1 = float(np.mean([
        row["raw_pca"]["f1"] for row in dataset_details
    ]))
    fmt_ap = float(np.mean([
        row["fmt"]["average_precision"] for row in dataset_details
    ]))
    raw_ap = float(np.mean([
        row["raw_pca"]["average_precision"] for row in dataset_details
    ]))
    target_gain = float(spec["selection"]["target_dataset_macro_f1_gain"])
    target_fmt = float(spec["selection"]["target_absolute_fmt_f1"])
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "selection_rule": (
            "per physical family, maximize guarded development paired F1 "
            "gain across the preregistered 44.1, 45.1, and 48.1 winners; "
            f"tie breakers={spec['selection_metrics'][1:]}"
        ),
        "training_runs": 0,
        "confirmation_opened": False,
        "source_paired_seeds": spec["source_paired_seeds"],
        "frozen_confirmation_seeds": spec["frozen_confirmation_seeds"],
        "source_artifact_sha256": {
            name: source["artifact_sha256"]
            for name, source in sources.items()
        },
        "family_datasets": reference,
        "primary_by_group": public_primary,
        "models": models,
        "dataset_details": dataset_details,
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
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("static-preflight", "select"), required=True)
    args = parser.parse_args()
    if args.mode == "static-preflight":
        static_preflight(args.config)
    else:
        select(args.config)


if __name__ == "__main__":
    main()
