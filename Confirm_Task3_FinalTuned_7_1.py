"""Sealed confirmation of the selected Task3 training stack.

The development-only 49.1 portfolio selection and its paired checkpoints are
frozen by hash before the fifth spatial primitive population is generated.
This file performs no training, feature selection, threshold selection, or
residual-scale selection on confirmation data.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import Build_Task3_FinalTuned_Confirmation_7_1 as spatial
import Confirm_Task3_AnchoredFeature_6_1 as _base
from Search_Task3_FMTResidual_3D import _read_csv, _write_csv


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_text_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _under(root: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else root / value


def _source_root(section: dict) -> Path:
    environment = str(section.get("environment", "TASK71_SOURCE_MODEL_ROOT"))
    return Path(os.environ.get(environment, section["repo_root"]))


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "task", "status", "output_root", "recipe_manifest",
        "source_model", "source_staging", "datasets", "paired_seeds",
        "confirmation_count", "expected_ivd_percentile",
        "require_confirmation_reference_match", "batch_size",
        "phase_key", "phase_key_sha256", "halton_index",
        "confirmation_seed_grid_phase", "confirmation_roots",
        "target_dataset_macro_f1_gain",
        "aspirational_dataset_macro_f1_gain",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing Task3 7.1 config keys: {missing}")
    if spec["experiment"] != spatial.EXPERIMENT or spec["task"] != "Task3":
        raise ValueError("Task3 7.1 experiment identity changed")
    if spec["status"] != "fresh_spatial_confirmation":
        raise ValueError("Task3 7.1 must remain a fresh confirmation")
    datasets = [str(value) for value in spec["datasets"]]
    seeds = [int(value) for value in spec["paired_seeds"]]
    if len(datasets) != 10 or len(set(datasets)) != 10:
        raise ValueError("Task3 7.1 requires ten unique datasets")
    if seeds != [40, 41]:
        raise ValueError("Task3 7.1 freezes paired seeds 40 and 41")
    source_seeds = [
        int(value) for value in spec["source_model"]["source_paired_seeds"]
    ]
    if source_seeds != [40, 41, 42] or not set(seeds).issubset(source_seeds):
        raise ValueError("Task3 7.1 source paired seeds changed")
    if int(spec["confirmation_count"]) != 4:
        raise ValueError("Task3 7.1 requires four spatial slices")
    if not np.isclose(float(spec["expected_ivd_percentile"]), 95.0):
        raise ValueError("Task3 7.1 requires whole-field IVD-p95")
    if not bool(spec["require_confirmation_reference_match"]):
        raise ValueError("Task3 7.1 requires source/reference identity")
    if str(spec["phase_key"]) != spatial.PHASE_KEY:
        raise ValueError("Task3 7.1 phase key changed")
    if str(spec["phase_key_sha256"]) != spatial.PHASE_KEY_SHA256:
        raise ValueError("Task3 7.1 phase-key SHA-256 changed")
    if int(spec["halton_index"]) != spatial.HALTON_INDEX:
        raise ValueError("Task3 7.1 Halton index changed")
    if [float(value) for value in spec["confirmation_seed_grid_phase"]] != list(
        spatial.SEED_GRID_PHASE
    ):
        raise ValueError("Task3 7.1 spatial phase changed")
    if float(spec["target_dataset_macro_f1_gain"]) != 0.15:
        raise ValueError("Task3 7.1 primary target changed")
    if float(spec["aspirational_dataset_macro_f1_gain"]) != 0.20:
        raise ValueError("Task3 7.1 aspirational target changed")
    if not bool(spec.get("new_spatial_primitive_population", False)):
        raise ValueError("Task3 7.1 must declare a new primitive population")
    if bool(spec.get("confirmation_opened_before_freeze", True)):
        raise ValueError("Task3 7.1 cannot be open before freeze")

    roots = dict(spec["confirmation_roots"])
    if set(roots) != set(spatial.SETTINGS):
        raise ValueError("Task3 7.1 root groups changed")
    grouped = []
    for name, group in roots.items():
        expected = list(spatial.SETTINGS[name]["indices"])
        observed = [str(value) for value in group.get("datasets", [])]
        if observed != expected:
            raise ValueError(f"Task3 7.1 {name} dataset order changed")
        if Path(group["source_root"]) != Path(spatial.SETTINGS[name]["cache_dir"]):
            raise ValueError(f"Task3 7.1 {name} source root changed")
        label_path = Path(spatial.SETTINGS[name]["label_config"])
        label_spec = yaml.safe_load(label_path.read_text(encoding="utf-8"))
        if Path(group["label_root"]) != Path(label_spec["output_dir"]) / "labels":
            raise ValueError(f"Task3 7.1 {name} label root changed")
        grouped.extend(observed)
    if set(grouped) != set(datasets) or len(grouped) != len(set(grouped)):
        raise ValueError("Task3 7.1 roots do not partition datasets")
    if len(str(spec["source_staging"]["parent_manifest_sha256"])) != 64:
        raise ValueError("source-staging parent SHA-256 is incomplete")
    expected_config_hash = str(
        spec["source_model"]["expected_config_canonical_sha256"]
    )
    if len(expected_config_hash) != 64:
        raise ValueError("source-model config hash is incomplete")

    # The selection hash cannot exist when this protocol is committed.  It is
    # injected only after the dependency-guarded selector has completed, then
    # persisted in static_preflight.json and frozen_recipe_manifest.json.
    selection_path = _under(
        _source_root(spec["source_model"]),
        spec["source_model"]["paths"]["selection"],
    )
    selection_hash = _sha256(selection_path) if selection_path.is_file() else "0" * 64
    spec["source_model"] = dict(spec["source_model"])
    spec["source_model"]["sha256"] = {"selection": selection_hash}
    spec["datasets"] = datasets
    spec["paired_seeds"] = seeds
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    return spec


def _selected_candidate(source: dict, selection: dict,
                        dataset: str) -> tuple[str, dict]:
    matches = [
        name for name, group in source["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Task3 7.1 family lookup failed: {dataset}")
    family = matches[0]
    row = dict(selection["primary_by_group"][family])
    recipe = json.loads(str(row["optimization_recipe_json"]))
    candidate_id = str(row["optimization_id"])
    if str(recipe.get("id")) != candidate_id:
        raise RuntimeError(f"49.1 selected recipe id changed for {family}")
    feature = str(recipe.get("fmt_feature", ""))
    if not feature:
        raise RuntimeError(f"49.1 selected feature missing for {family}")
    return family, {
        "id": candidate_id,
        "optimization_id": candidate_id,
        "fmt_feature": feature,
        "optimization_recipe": recipe,
    }


def _source_state(spec: dict) -> tuple:
    section = spec["source_model"]
    root = _source_root(section)
    paths = {
        name: _under(root, value)
        for name, value in section["paths"].items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if _canonical_text_sha256(paths["config"]) != str(
        section["expected_config_canonical_sha256"]
    ):
        raise RuntimeError("49.1 source config changed")
    overlay = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    expected = str(section["expected_experiment"])
    for name, payload in (("config", overlay), ("selection", selection)):
        if str(payload.get("experiment")) != expected:
            raise RuntimeError(f"49.1 {name} experiment changed")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("49.1 selection opened confirmation")
    config_hash = _sha256(paths["config"])
    if str(selection.get("config_sha256", "")).lower() != config_hash:
        raise RuntimeError("49.1 selection/config hash mismatch")
    if [int(value) for value in selection.get("source_paired_seeds", [])] != [
        40, 41, 42
    ]:
        raise RuntimeError("49.1 source paired seeds changed")
    if [int(value) for value in selection.get(
        "frozen_confirmation_seeds", []
    )] != spec["paired_seeds"]:
        raise RuntimeError("49.1 frozen confirmation seeds changed")
    groups = {
        str(family): {
            "datasets": [str(dataset) for dataset in datasets]
        }
        for family, datasets in selection.get("family_datasets", {}).items()
    }
    flattened = [
        dataset for group in groups.values() for dataset in group["datasets"]
    ]
    if set(flattened) != set(spec["datasets"]) or len(flattened) != len(
        set(flattened)
    ):
        raise RuntimeError("49.1 family/dataset map changed")
    if set(selection.get("primary_by_group", {})) != set(groups):
        raise RuntimeError("49.1 selected physical families changed")
    source = {
        "experiment": expected,
        "datasets": list(spec["datasets"]),
        "groups": groups,
    }
    for dataset in spec["datasets"]:
        _selected_candidate(source, selection, dataset)
    spec["source_model"]["sha256"]["selection"] = _sha256(paths["selection"])
    return root, paths, source, {}, selection


def _collect_models(spec: dict, source_root: Path, source: dict,
                    selection: dict) -> list[dict]:
    models = [dict(row) for row in selection.get("models", [])]
    if len(models) != 40:
        raise RuntimeError("Task3 7.1 must freeze exactly 40 models")
    expected_keys = {
        (dataset, seed, arm)
        for dataset in spec["datasets"]
        for seed in spec["paired_seeds"]
        for arm in ("fmt", "raw_pca")
    }
    observed_keys = set()
    for item in models:
        key = (
            str(item["dataset"]), int(item["seed"]), str(item["source"])
        )
        if key not in expected_keys or key in observed_keys:
            raise RuntimeError(f"Task3 7.1 unexpected portfolio model: {key}")
        observed_keys.add(key)
        family, candidate = _selected_candidate(
            source, selection, str(item["dataset"])
        )
        expected = {
            "physical_family": family,
            "candidate_id": candidate["id"],
            "fmt_feature": candidate["fmt_feature"],
        }
        for name, value in expected.items():
            if str(item.get(name, "")) != str(value):
                raise RuntimeError(f"Task3 7.1 portfolio model changed: {name}")
        for path_key, hash_key in (
            ("result", "result_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            path = Path(item[path_key])
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != str(item[hash_key]):
                raise RuntimeError(
                    f"Task3 7.1 portfolio {path_key} changed: {path}"
                )
    if observed_keys != expected_keys:
        raise RuntimeError("Task3 7.1 portfolio model set is incomplete")
    for dataset in spec["datasets"]:
        for seed in spec["paired_seeds"]:
            paired = [
                row for row in models
                if row["dataset"] == dataset and int(row["seed"]) == seed
            ]
            for name in (
                "fmt_dim", "parameter_count",
                "trainable_residual_parameter_count",
            ):
                if len({int(row[name]) for row in paired}) != 1:
                    raise RuntimeError(
                        f"Task3 7.1 paired {name} mismatch: {dataset}/seed{seed}"
                    )
    return models


_replay_adapter = SimpleNamespace(_selected_candidate=_selected_candidate)


@contextmanager
def _configured_base():
    replacements = {
        "spatial": spatial,
        "replay": _replay_adapter,
        "_load_spec": _load_spec,
        "_source_state": _source_state,
        "_collect_models": _collect_models,
    }
    previous = {name: getattr(_base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def static_preflight(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.static_preflight(config_path)


def freeze(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.freeze(config_path)


def _frozen_state(spec: dict):
    with _configured_base():
        return _base._frozen_state(spec)


def source_preflight(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.source_preflight(config_path)


def build_cache(config_path: str | Path, job_index: int,
                overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.build_cache(config_path, job_index, overwrite)


def build_labels(config_path: str | Path, group_index: int,
                 overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.build_labels(config_path, group_index, overwrite)


def evaluation_preflight(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.evaluation_preflight(config_path)


def run_dataset(config_path: str | Path, dataset: str) -> Path:
    with _configured_base():
        return _base.run_dataset(config_path, dataset)


def summarize(config_path: str | Path) -> Path:
    """Aggregate only after all ten sealed 49.1-portfolio evaluations."""
    with _configured_base():
        spec = _load_spec(config_path)
        manifest_path, manifest, _, _, selection = _base._frozen_state(spec)
        eval_path, _ = _base._evaluation_state(spec)
        manifest_hash = _sha256(manifest_path)
        eval_hash = _sha256(eval_path)
        rows = []
        for dataset in spec["datasets"]:
            shard = _read_csv(_base._shard_path(spec, dataset))
            _base._validate_rows(
                shard, spec, manifest, manifest_hash, eval_hash, dataset, True
            )
            rows.extend(shard)
        output = Path(spec["output_root"])
        _write_csv(output / "per_run.csv", rows)
        aggregate = _base._aggregate(rows, spec["datasets"])
        datasets = aggregate["datasets"]
        raw_f1 = float(np.mean([
            value["raw_pca_residual"]["f1"] for value in datasets.values()
        ]))
        fmt_f1 = float(np.mean([
            value["fmt_residual"]["f1"] for value in datasets.values()
        ]))
        raw_ap = float(np.mean([
            value["raw_pca_residual"]["average_precision"]
            for value in datasets.values()
        ]))
        fmt_ap = float(np.mean([
            value["fmt_residual"]["average_precision"]
            for value in datasets.values()
        ]))
        target = float(spec["target_dataset_macro_f1_gain"])
        aspirational = float(spec["aspirational_dataset_macro_f1_gain"])
        result = {
            "experiment": spec["experiment"],
            "status": spec["status"],
            "comparison": (
                "frozen 49.1 portfolio FMT residual minus its same-recipe, "
                "same-capacity train-only Raw-PCA residual"
            ),
            "source_search_experiment": selection["experiment"],
            "fresh_confirmation": True,
            "confirmation_data_was_not_used_for_selection": True,
            "recipe_manifest_sha256": manifest_hash,
            "evaluation_preflight_sha256": eval_hash,
            "source_model_selection_sha256": manifest[
                "source_model_selection_sha256"
            ],
            "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
            "phase_key_sha256": spatial.PHASE_KEY_SHA256,
            "halton_index": spatial.HALTON_INDEX,
            "paired_seeds": spec["paired_seeds"],
            "dataset_macro_raw_pca_f1": raw_f1,
            "dataset_macro_fmt_f1": fmt_f1,
            "dataset_macro_raw_pca_ap": raw_ap,
            "dataset_macro_fmt_ap": fmt_ap,
            "source_development_f1_gain": float(selection[
                "development_dataset_macro_f1_gain_vs_raw_pca"
            ]),
            **aggregate,
            "target_dataset_macro_f1_gain": target,
            "target_reached": aggregate[
                "dataset_macro_f1_gain_vs_raw_pca"
            ] >= target,
            "aspirational_dataset_macro_f1_gain": aspirational,
            "aspirational_target_reached": aggregate[
                "dataset_macro_f1_gain_vs_raw_pca"
            ] >= aspirational,
        }
        target_path = output / "summary.json"
        target_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(target_path.read_text(encoding="utf-8"))
        return target_path


def _decode_dataset(spec: dict, index: int) -> str:
    if not 0 <= int(index) < len(spec["datasets"]):
        raise IndexError("Task3 7.1 dataset job outside [0,10)")
    return spec["datasets"][int(index)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "static-preflight", "freeze", "source-preflight", "cache",
            "labels", "evaluation-preflight", "dataset", "summary",
        ),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "static-preflight":
        static_preflight(args.config)
    elif args.mode == "freeze":
        freeze(args.config)
    elif args.mode == "source-preflight":
        source_preflight(args.config)
    elif args.mode == "cache":
        if args.job_index is None:
            parser.error("cache mode requires --job-index")
        build_cache(args.config, args.job_index, args.overwrite)
    elif args.mode == "labels":
        if args.job_index is None:
            parser.error("labels mode requires --job-index")
        build_labels(args.config, args.job_index, args.overwrite)
    elif args.mode == "evaluation-preflight":
        evaluation_preflight(args.config)
    elif args.mode == "dataset":
        spec = _load_spec(args.config)
        dataset = args.dataset
        if dataset is None:
            if args.job_index is None:
                parser.error("dataset mode requires --dataset or --job-index")
            dataset = _decode_dataset(spec, args.job_index)
        run_dataset(args.config, dataset)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
