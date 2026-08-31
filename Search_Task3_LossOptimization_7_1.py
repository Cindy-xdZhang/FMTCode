"""Paired loss and optimization search for supervised 3D Task3.

The experiment waits for the development-only 5.2 selector, freezes its
family-specific feature/network recipes in a preflight manifest, and then
changes the same training recipe in both the FMT and train-only Raw-PCA arms.
No confirmation population is opened by this script.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.PathlineClassifier_3D import (
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from FMT_Utils.Task12Data_3D import feature_block_dims
from Search_Task3_FMTResidual_3D import (
    _candidate_spec,
    _frozen_raw_normalization,
    _group_for_dataset,
    _load_search_splits,
    _load_spec,
    _read_csv,
    _write_csv,
)
from Search_Task3_FMTResidual_Stage2_3D import _combined_candidate
from Verify_Task3_FMTClassifier import _append_csv, _normalize_train_only
from Verify_Task3_FMTResidual import (
    _auxiliary_learning_rate_multiplier,
    _auxiliary_weight_decay_multiplier,
    _build_training_loss,
    _gradient_clip_norm,
    _load_raw_model,
    _optimizer_betas,
    _train_one,
    _warmup_parameters,
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_text_sha256(path: str | Path) -> str:
    """Hash text with LF newlines so Git content is OS-independent."""
    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_optimization_spec(path: str | Path) -> dict:
    path = Path(path)
    overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "base_search_config",
        "base_search_config_sha256", "upstream_selection", "paired_seeds",
        "model_override", "optimization_candidates", "selection",
    }
    missing = sorted(required.difference(overlay))
    if missing:
        raise ValueError(f"missing optimization config keys: {missing}")
    base_path = Path(overlay["base_search_config"])
    base_hash = _canonical_text_sha256(base_path)
    if base_hash != str(overlay["base_search_config_sha256"]).lower():
        raise RuntimeError("base Task3 development config changed")
    base = _load_spec(base_path)
    seeds = [int(value) for value in overlay["paired_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("paired_seeds must be non-empty and unique")
    if not set(seeds).issubset({int(value) for value in base["stage2_screen_seeds"]}):
        raise ValueError("paired seeds require unavailable frozen Raw checkpoints")
    candidates = [dict(row) for row in overlay["optimization_candidates"]]
    candidate_ids = [str(row["id"]) for row in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("optimization candidate ids must be non-empty and unique")
    selection = dict(overlay["selection"])
    if bool(selection.get("confirmation_opened", False)):
        raise RuntimeError("optimization search must not open confirmation")
    absolute_guard = selection.get("absolute_fmt_guard")
    if absolute_guard is not None:
        absolute_guard = dict(absolute_guard)
        required_guard_keys = {
            "control_optimization_id", "f1_tolerance",
            "average_precision_tolerance",
        }
        missing_guard_keys = sorted(
            required_guard_keys.difference(absolute_guard)
        )
        if missing_guard_keys:
            raise ValueError(
                "absolute_fmt_guard misses keys: "
                f"{missing_guard_keys}"
            )
        control_id = str(absolute_guard["control_optimization_id"])
        if control_id not in candidate_ids:
            raise ValueError(
                "absolute_fmt_guard control is not an optimization candidate"
            )
        for key in ("f1_tolerance", "average_precision_tolerance"):
            value = float(absolute_guard[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"absolute_fmt_guard.{key} must be finite and non-negative"
                )
            absolute_guard[key] = value
        absolute_guard["control_optimization_id"] = control_id
        selection["absolute_fmt_guard"] = absolute_guard
    combination_sources = {
        str(name): dict(row)
        for name, row in dict(overlay.get("combination_sources", {})).items()
    }
    for name, row in combination_sources.items():
        if not row.get("selection") or not row.get("expected_experiment"):
            raise ValueError(
                f"combination source {name!r} requires selection and "
                "expected_experiment"
            )
        kind = str(row.get("kind", "optimization"))
        if kind not in {"optimization", "stage1_feature"}:
            raise ValueError(
                f"combination source {name!r} has unsupported kind {kind!r}"
            )
        row["kind"] = kind
    if combination_sources:
        for candidate in candidates:
            sources = [str(value) for value in candidate.get("sources", [])]
            unknown = sorted(set(sources) - set(combination_sources))
            if unknown:
                raise ValueError(
                    f"unknown combination sources for {candidate['id']}: "
                    f"{unknown}"
                )
            if len(sources) != len(set(sources)):
                raise ValueError(
                    f"duplicate combination source for {candidate['id']}"
                )
    spec = dict(base)
    spec.update({
        "experiment": str(overlay["experiment"]),
        "output_root": str(overlay["output_root"]),
        "optimization_config": str(path),
        "optimization_config_sha256": _sha256(path),
        "base_search_config": str(base_path),
        "base_search_config_sha256": base_hash,
        "upstream_selection": str(overlay["upstream_selection"]),
        "upstream_selector_job_id": str(
            overlay.get("upstream_selector_job_id", "")
        ),
        "paired_seeds": seeds,
        "stage2_screen_seeds": seeds,
        "model_override": dict(overlay["model_override"]),
        "optimization_candidates": candidates,
        "optimization_selection": selection,
        "combination_sources": combination_sources,
    })
    return spec


def _merge_combination_recipe(candidate_id: str, source_names: list[str],
                              source_rows: dict[str, dict]) -> dict:
    """Merge frozen source winners, rejecting hidden hyperparameter conflicts."""
    merged = {
        "id": str(candidate_id),
        "sources": [str(value) for value in source_names],
        "source_optimization_ids": {},
    }
    for source_name in source_names:
        row = source_rows[str(source_name)]
        recipe = json.loads(str(row["optimization_recipe_json"]))
        merged["source_optimization_ids"][str(source_name)] = str(
            row["optimization_id"]
        )
        # A completed combination selector stores these two fields only as
        # frozen provenance.  Allow that selector to become one source of a
        # later, explicitly preregistered combination without treating its
        # provenance as trainable hyperparameters.  Its exact recipe remains
        # bound by the source selection SHA-256.
        unsupported = sorted(
            set(recipe) - {
                "id", "training", "model", "fmt_feature", "sources",
                "source_optimization_ids",
            }
        )
        if unsupported:
            raise ValueError(
                f"unsupported keys from {source_name}: {unsupported}"
            )
        for section in ("training", "model"):
            values = dict(recipe.get(section, {}))
            target = merged.setdefault(section, {})
            for key, value in values.items():
                if key in target and target[key] != value:
                    raise ValueError(
                        f"conflicting {section}.{key} while combining "
                        f"{source_names}: {target[key]!r} vs {value!r}"
                    )
                target[key] = value
        if "fmt_feature" in recipe:
            value = str(recipe["fmt_feature"])
            if "fmt_feature" in merged and merged["fmt_feature"] != value:
                raise ValueError(
                    "conflicting fmt_feature while combining "
                    f"{source_names}: {merged['fmt_feature']!r} vs {value!r}"
                )
            merged["fmt_feature"] = value
    for section in ("training", "model"):
        if not merged.get(section):
            merged.pop(section, None)
    return merged


def _merge_candidate_overrides(merged: dict, candidate: dict) -> dict:
    """Add locally declared knobs without overwriting frozen source values."""
    unsupported = sorted(
        set(candidate) - {"id", "sources", "training", "model", "fmt_feature"}
    )
    if unsupported:
        raise ValueError(
            f"unsupported local candidate keys for {candidate['id']}: "
            f"{unsupported}"
        )
    result = dict(merged)
    for section in ("training", "model"):
        source_values = dict(result.get(section, {}))
        for key, value in dict(candidate.get(section, {})).items():
            if key in source_values and source_values[key] != value:
                raise ValueError(
                    f"local candidate conflicts with frozen {section}.{key}: "
                    f"{source_values[key]!r} vs {value!r}"
                )
            source_values[key] = value
        if source_values:
            result[section] = source_values
    if "fmt_feature" in candidate:
        value = str(candidate["fmt_feature"])
        if "fmt_feature" in result and result["fmt_feature"] != value:
            raise ValueError(
                "local candidate conflicts with frozen fmt_feature: "
                f"{result['fmt_feature']!r} vs {value!r}"
            )
        result["fmt_feature"] = value
    return result


def _resolve_combination_candidates(spec: dict) -> tuple[dict, dict]:
    """Resolve family-specific recipes only from completed selector files."""
    if not spec.get("combination_sources"):
        return {}, {}
    selections, hashes = {}, {}
    for name, source in spec["combination_sources"].items():
        path = Path(source["selection"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("experiment")) != str(source["expected_experiment"]):
            raise RuntimeError(f"combination source experiment changed: {name}")
        if bool(payload.get("confirmation_opened", True)):
            raise RuntimeError(f"combination source opened confirmation: {name}")
        if set(payload.get("primary_by_group", {})) != set(spec["groups"]):
            raise RuntimeError(f"combination source families changed: {name}")
        selections[name] = payload
        hashes[name] = _sha256(path)
    resolved = {}
    for group_name in spec["groups"]:
        source_rows = {}
        for name, payload in selections.items():
            row = dict(payload["primary_by_group"][group_name])
            kind = str(spec["combination_sources"][name]["kind"])
            if kind == "optimization":
                required = {"optimization_id", "optimization_recipe_json"}
                missing = sorted(required.difference(row))
                if missing:
                    raise RuntimeError(
                        f"optimization source {name!r}/{group_name} misses "
                        f"{missing}"
                    )
                source_rows[name] = row
            elif kind == "stage1_feature":
                required = {"candidate_id", "fmt_feature"}
                missing = sorted(required.difference(row))
                if missing:
                    raise RuntimeError(
                        f"stage1 feature source {name!r}/{group_name} misses "
                        f"{missing}"
                    )
                source_rows[name] = {
                    "optimization_id": str(row["candidate_id"]),
                    "optimization_recipe_json": json.dumps({
                        "id": str(row["candidate_id"]),
                        "fmt_feature": str(row["fmt_feature"]),
                    }, sort_keys=True),
                }
            else:  # guarded by _load_optimization_spec
                raise AssertionError(f"unsupported combination kind {kind!r}")
        resolved[group_name] = []
        for candidate in spec["optimization_candidates"]:
            merged = _merge_combination_recipe(
                candidate["id"],
                [str(value) for value in candidate.get("sources", [])],
                source_rows,
            )
            resolved[group_name].append(
                _merge_candidate_overrides(merged, candidate)
            )
    return resolved, hashes


def _manifest_path(spec: dict) -> Path:
    return Path(spec["output_root"]) / "preflight_manifest.json"


def _upstream_base_candidates(spec: dict, selection: dict) -> dict[str, dict]:
    if bool(selection.get("confirmation_opened", False)):
        raise RuntimeError("upstream selection unexpectedly opened confirmation")
    if int(selection.get("stage", -1)) != 2:
        raise RuntimeError("Task3 7.1 requires a completed stage-2 selection")
    primary = selection["primary_by_group"]
    if set(primary) != set(spec["groups"]):
        raise RuntimeError("upstream selection physical families changed")
    feature_lookup = {str(row["id"]): dict(row) for row in spec["candidates"]}
    network_lookup = {
        str(row["id"]): dict(row) for row in spec["stage2_networks"]
    }
    result = {}
    for group, row in primary.items():
        feature_id = str(row["feature_candidate_id"])
        network_id = str(row["network_id"])
        if feature_id not in feature_lookup or network_id not in network_lookup:
            raise RuntimeError(f"unknown upstream recipe for {group}")
        candidate = _combined_candidate(
            feature_lookup[feature_id], network_lookup[network_id]
        )
        if str(candidate["id"]) != str(row["candidate_id"]):
            raise RuntimeError(f"upstream candidate identity changed for {group}")
        candidate.update(spec["model_override"])
        candidate["upstream_candidate_id"] = str(row["candidate_id"])
        result[group] = candidate
    return result


def _load_manifest(spec: dict) -> dict:
    path = _manifest_path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"preflight manifest is required before training: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
    }
    for key, value in expected.items():
        if str(manifest.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"preflight manifest changed: {key}")
    selection_path = Path(spec["upstream_selection"])
    if _sha256(selection_path) != str(
        manifest["upstream_selection_sha256"]
    ).lower():
        raise RuntimeError("upstream 5.2 selection changed after 7.1 preflight")
    if bool(manifest.get("confirmation_opened", True)):
        raise RuntimeError("invalid preflight confirmation state")
    for name, source in spec.get("combination_sources", {}).items():
        observed = _sha256(Path(source["selection"]))
        expected = str(
            manifest.get("combination_source_selection_sha256", {}).get(
                name, ""
            )
        ).lower()
        if observed != expected:
            raise RuntimeError(
                f"combination source changed after preflight: {name}"
            )
    return manifest


def _optimization_candidate(spec: dict, manifest: dict, dataset: str,
                            candidate_index: int) -> dict:
    group_name, _ = _group_for_dataset(spec, dataset)
    index = int(candidate_index)
    if not 0 <= index < len(spec["optimization_candidates"]):
        raise IndexError("optimization candidate index outside configured grid")
    base = dict(manifest["base_candidate_by_group"][group_name])
    resolved = manifest.get("optimization_candidates_by_group", {})
    recipe = dict(
        resolved[group_name][index]
        if group_name in resolved else spec["optimization_candidates"][index]
    )
    training = dict(base.get("training", {}))
    training.update(recipe.get("training", {}))
    model = dict(recipe.get("model", {}))
    base.update(model)
    if "fmt_feature" in recipe:
        base["fmt_feature"] = str(recipe["fmt_feature"])
    if str(base.get("auxiliary_projection", "")).startswith("blockwise_"):
        inferred = list(feature_block_dims(base["fmt_feature"]))
        declared = base.get("auxiliary_block_dims")
        if declared is not None and [int(value) for value in declared] != inferred:
            raise ValueError(
                f"{recipe['id']}: auxiliary_block_dims disagree with "
                f"{base['fmt_feature']}: {declared} vs {inferred}"
            )
        base["auxiliary_block_dims"] = inferred
        recipe_model = dict(recipe.get("model", {}))
        recipe_model["auxiliary_block_dims"] = inferred
        recipe["model"] = recipe_model
    base["training"] = training
    base["id"] = str(recipe["id"])
    base["optimization_id"] = str(recipe["id"])
    base["optimization_recipe"] = recipe
    return base


def _parameter_budget(spec: dict, group: dict, candidate: dict,
                      dataset: str, fmt_dim: int) -> dict:
    totals, trainables = [], []
    for seed in spec["paired_seeds"]:
        raw_model, _ = _load_raw_model(
            Path(group["raw_checkpoint_dir"])
            / f"{dataset}_raw_seed{int(seed)}.pt",
            int(fmt_dim), torch.device("cpu"),
        )
        model = PathlineFMTResidualClassifier3D(
            raw_model, fmt_dim=int(fmt_dim),
            **residual_model_kwargs(candidate),
        )
        totals.append(sum(parameter.numel() for parameter in model.parameters()))
        trainables.append(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ))
    if len(set(totals)) != 1 or len(set(trainables)) != 1:
        raise RuntimeError("paired seeds changed parameter counts")
    limit = int(spec["raw_wide_parameter_count"])
    return {
        "eligible": int(totals[0]) < limit,
        "total_parameter_count": int(totals[0]),
        "trainable_residual_parameter_count": int(trainables[0]),
        "raw_wide_parameter_count": limit,
    }


def static_preflight(config_path: str) -> None:
    spec = _load_optimization_spec(config_path)
    payload = {
        "experiment": spec["experiment"],
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "upstream_selection": spec["upstream_selection"],
        "upstream_selection_exists": Path(spec["upstream_selection"]).exists(),
        "dataset_count": len(spec["datasets"]),
        "physical_family_count": len(spec["groups"]),
        "optimization_candidate_count": len(spec["optimization_candidates"]),
        "paired_seed_count": len(spec["paired_seeds"]),
        "paired_arm_count": 2,
        "expected_training_runs": (
            len(spec["datasets"]) * len(spec["optimization_candidates"])
            * len(spec["paired_seeds"]) * 2
        ),
        "confirmation_opened": False,
        "combination_source_count": len(spec.get("combination_sources", {})),
    }
    print(json.dumps(payload, indent=2))


def _array_fingerprint(values: np.ndarray) -> str:
    """Hash shape, dtype, and bytes without retaining a second full array."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def preflight(config_path: str) -> Path:
    spec = _load_optimization_spec(config_path)
    selection_path = Path(spec["upstream_selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    base_candidates = _upstream_base_candidates(spec, selection)
    resolved_candidates, source_hashes = _resolve_combination_candidates(spec)
    manifest_stub = {
        "base_candidate_by_group": base_candidates,
        "optimization_candidates_by_group": resolved_candidates,
    }
    datasets = []
    for dataset in spec["datasets"]:
        group_name, group = _group_for_dataset(spec, dataset)
        first = None
        reference_signature = None
        dataset_summary = None
        split_by_feature = {}
        recipes = []
        for index in range(len(spec["optimization_candidates"])):
            candidate = _optimization_candidate(
                spec, manifest_stub, dataset, index
            )
            feature = str(candidate["fmt_feature"])
            if feature not in split_by_feature:
                train, validation = _load_search_splits(
                    spec, dataset, candidate, torch.device("cpu")
                )
                fmt_dim = int(train[1].shape[1])
                raw_dim = int(np.prod(train[0].shape[1:], dtype=np.int64))
                if fmt_dim > raw_dim:
                    raise RuntimeError(
                        f"{dataset}/{candidate['id']}: FMT width exceeds Raw width"
                    )
                if not (0 < int(train[2].sum()) < len(train[2])):
                    raise RuntimeError(
                        f"{dataset}/{candidate['id']}: training labels are "
                        "single-class"
                    )
                if not (0 < int(validation[2].sum()) < len(validation[2])):
                    raise RuntimeError(
                        f"{dataset}/{candidate['id']}: validation labels are "
                        "single-class"
                    )
                signature = {
                    "raw_dim": raw_dim,
                    "training_samples": int(len(train[2])),
                    "training_positive_count": int(train[2].sum()),
                    "validation_samples": int(len(validation[2])),
                    "validation_positive_count": int(validation[2].sum()),
                    "training_raw_sha256": _array_fingerprint(train[0]),
                    "training_labels_sha256": _array_fingerprint(train[2]),
                    "validation_raw_sha256": _array_fingerprint(validation[0]),
                    "validation_labels_sha256": _array_fingerprint(
                        validation[2]
                    ),
                }
                split_by_feature[feature] = (
                    train, validation, fmt_dim, raw_dim, signature
                )
                if reference_signature is None:
                    reference_signature = signature
                    first = candidate
                    dataset_summary = {
                        key: signature[key] for key in (
                            "raw_dim", "training_samples",
                            "training_positive_count", "validation_samples",
                            "validation_positive_count",
                        )
                    }
                elif signature != reference_signature:
                    changed = sorted(
                        key for key in signature
                        if signature[key] != reference_signature[key]
                    )
                    raise RuntimeError(
                        f"{dataset}/{candidate['id']}: candidate-specific FMT "
                        "feature changed paired Raw/label population: "
                        f"{changed}"
                    )
            train, validation, fmt_dim, raw_dim, _ = split_by_feature[feature]
            budget = _parameter_budget(
                spec, group, candidate, dataset, fmt_dim
            )
            if not budget["eligible"]:
                raise RuntimeError(
                    f"{dataset}/{candidate['id']} exceeds Raw-wide capacity"
                )
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, spec["paired_seeds"][0],
                "fmt", Path(spec["output_root"]) / "preflight", fmt_dim,
            )
            _gradient_clip_norm(run_spec["training"])
            _auxiliary_learning_rate_multiplier(run_spec["training"])
            _auxiliary_weight_decay_multiplier(run_spec["training"])
            _optimizer_betas(run_spec["training"])
            _warmup_parameters(run_spec["training"])
            _, loss_metadata = _build_training_loss(
                run_spec["training"], float(train[2].sum()),
                float(len(train[2]) - train[2].sum()), torch.device("cpu"),
            )
            supervision_enabled = (
                float(loss_metadata["auxiliary_supervision_loss_weight"]) > 0.0
            )
            classifier_enabled = str(run_spec["model"].get(
                "auxiliary_classifier_architecture", "none"
            )).lower() != "none"
            if supervision_enabled != classifier_enabled:
                raise ValueError(
                    f"{dataset}/{candidate['id']}: auxiliary supervision and "
                    "auxiliary classifier must be enabled together"
                )
            recipes.append({
                "optimization_id": candidate["id"],
                "fmt_feature": candidate["fmt_feature"],
                "fmt_dim": fmt_dim,
                **budget,
            })
        if first is None or dataset_summary is None:
            raise RuntimeError(f"{dataset}: no optimization candidate was checked")
        datasets.append({
            "dataset": dataset,
            "physical_family": group_name,
            "fmt_feature": first["fmt_feature"],
            "upstream_candidate_id": first["upstream_candidate_id"],
            "fmt_dim": recipes[0]["fmt_dim"],
            **dataset_summary,
            "recipes": recipes,
        })
    payload = {
        "experiment": spec["experiment"],
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "upstream_selection_sha256": _sha256(selection_path),
        "upstream_selector_job_id": spec["upstream_selector_job_id"],
        "base_candidate_by_group": base_candidates,
        "optimization_candidates_by_group": resolved_candidates,
        "combination_source_selection_sha256": source_hashes,
        "confirmation_opened": False,
        "dataset_count": len(spec["datasets"]),
        "optimization_candidate_count": len(spec["optimization_candidates"]),
        "paired_seed_count": len(spec["paired_seeds"]),
        "paired_arm_count": 2,
        "expected_training_runs": (
            len(spec["datasets"]) * len(spec["optimization_candidates"])
            * len(spec["paired_seeds"]) * 2
        ),
        "datasets": datasets,
    }
    target = _manifest_path(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target)
    return target


def _result_path(spec: dict, candidate: dict, dataset: str,
                 seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "candidates" / str(candidate["id"])
        / dataset / f"seed{int(seed)}" / source / "per_run.csv"
    )


def _numerical_instability_path(spec: dict, candidate: dict,
                                dataset: str) -> Path:
    return (
        Path(spec["output_root"]) / "candidates" / str(candidate["id"])
        / dataset / "invalid_numerical_instability.json"
    )


def _is_numerical_instability(error: BaseException) -> bool:
    if isinstance(error, FloatingPointError):
        return True
    text = str(error).lower()
    return isinstance(error, ValueError) and any(
        token in text for token in (
            "input contains nan", "non-finite", "not finite", "contains inf",
            "infinity",
        )
    )


def _load_numerical_instability(spec: dict, manifest: dict, candidate: dict,
                                dataset: str) -> dict | None:
    path = _numerical_instability_path(spec, candidate, dataset)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": 1,
        "status": "invalid_numerical_instability",
        "dataset": str(dataset),
        "optimization_id": str(candidate["id"]),
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "preflight_manifest_sha256": _sha256(_manifest_path(spec)),
        "upstream_selection_sha256": manifest["upstream_selection_sha256"],
    }
    for key, value in expected.items():
        if str(payload.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(
                f"numerical-instability marker changed for {path}: {key}"
            )
    payload["path"] = str(path)
    return payload


def _write_numerical_instability(spec: dict, manifest: dict, candidate: dict,
                                 dataset: str, seed: int, source: str,
                                 error: BaseException) -> Path:
    path = _numerical_instability_path(spec, candidate, dataset)
    payload = {
        "schema": 1,
        "status": "invalid_numerical_instability",
        "dataset": str(dataset),
        "optimization_id": str(candidate["id"]),
        "failed_seed": int(seed),
        "failed_source": str(source),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "preflight_manifest_sha256": _sha256(_manifest_path(spec)),
        "upstream_selection_sha256": manifest["upstream_selection_sha256"],
    }
    if path.exists():
        _load_numerical_instability(spec, manifest, candidate, dataset)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def run_candidate(config_path: str, dataset: str,
                  candidate_index: int) -> Path:
    spec = _load_optimization_spec(config_path)
    manifest = _load_manifest(spec)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    candidate = _optimization_candidate(
        spec, manifest, dataset, candidate_index
    )
    invalid = _load_numerical_instability(
        spec, manifest, candidate, dataset
    )
    if invalid is not None:
        print(
            f"INELIGIBLE {candidate['id']} {dataset}: "
            "invalid_numerical_instability",
            flush=True,
        )
        return Path(invalid["path"])
    _, group = _group_for_dataset(spec, dataset)
    device_name = str(spec["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    train, validation = _load_search_splits(spec, dataset, candidate, device)
    frozen_stats = _frozen_raw_normalization(
        group, dataset, spec["paired_seeds"][0]
    )
    train, validation, _, stats = _normalize_train_only(
        train, validation, raw_stats=frozen_stats
    )
    fmt_dim = int(train[1].shape[1])
    budget = _parameter_budget(spec, group, candidate, dataset, fmt_dim)
    if not budget["eligible"]:
        raise RuntimeError(f"{dataset}/{candidate['id']} exceeds parameter cap")
    last = None
    manifest_hash = _sha256(_manifest_path(spec))
    for seed in spec["paired_seeds"]:
        for source in ("fmt", "raw_pca"):
            path = _result_path(spec, candidate, dataset, seed, source)
            rows = _read_csv(path)
            if len(rows) > 1:
                raise RuntimeError(f"duplicate optimization result: {path}")
            if rows:
                expected = {
                    "optimization_config_sha256": spec[
                        "optimization_config_sha256"
                    ],
                    "preflight_manifest_sha256": manifest_hash,
                    "upstream_selection_sha256": manifest[
                        "upstream_selection_sha256"
                    ],
                }
                for key, value in expected.items():
                    if str(rows[0].get(key, "")).lower() != str(value).lower():
                        raise RuntimeError(f"stale cached result {path}: {key}")
                print(f"cached {candidate['id']} {dataset} seed={seed} {source}")
                last = path
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, seed, source,
                path.parent, fmt_dim,
            )
            (path.parent / "config_snapshot.yaml").write_text(
                yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8"
            )
            try:
                row = _train_one(
                    run_spec, dataset, seed, (train, validation, None), stats,
                    device, path.parent,
                )
            except (FloatingPointError, ValueError) as error:
                if not _is_numerical_instability(error):
                    raise
                marker = _write_numerical_instability(
                    spec, manifest, candidate, dataset, seed, source, error
                )
                print(
                    f"INELIGIBLE {candidate['id']} {dataset} seed={seed} "
                    f"{source}: invalid_numerical_instability: {error}",
                    flush=True,
                )
                return marker
            row.update({
                "optimization_id": candidate["optimization_id"],
                "optimization_recipe_json": json.dumps(
                    candidate["optimization_recipe"], sort_keys=True
                ),
                "upstream_candidate_id": candidate["upstream_candidate_id"],
                "fmt_feature": candidate["fmt_feature"],
                "fmt_dim": fmt_dim,
                "optimization_config_sha256": spec[
                    "optimization_config_sha256"
                ],
                "preflight_manifest_sha256": manifest_hash,
                "upstream_selection_sha256": manifest[
                    "upstream_selection_sha256"
                ],
            })
            _append_csv(path, row)
            last = path
            print(
                f"DONE {candidate['id']} {dataset} seed={seed} {source}: "
                f"F1={row['validation_f1']:.5f} "
                f"AP={row['validation_average_precision']:.5f}",
                flush=True,
            )
    return last


def _decode_job(spec: dict, job_index: int) -> tuple[str, int]:
    per_dataset = len(spec["optimization_candidates"])
    total = len(spec["datasets"]) * per_dataset
    index = int(job_index)
    if not 0 <= index < total:
        raise IndexError(f"optimization job index outside [0,{total})")
    dataset_index, candidate_index = divmod(index, per_dataset)
    return spec["datasets"][dataset_index], candidate_index


def run_job(config_path: str, job_index: int) -> Path:
    spec = _load_optimization_spec(config_path)
    dataset, candidate_index = _decode_job(spec, job_index)
    return run_candidate(config_path, dataset, candidate_index)


def _candidate_summary(spec: dict, manifest: dict, group_name: str,
                       recipe: dict) -> dict:
    datasets = spec["groups"][group_name]["datasets"]
    recipe_index = next(
        index for index, row in enumerate(spec["optimization_candidates"])
        if str(row["id"]) == str(recipe["id"])
    )
    representative = _optimization_candidate(
        spec, manifest, datasets[0], recipe_index
    )
    resolved_recipe = representative["optimization_recipe"]
    per_dataset = {}
    seed_gains = {int(seed): [] for seed in spec["paired_seeds"]}
    parameter_counts = set()
    instabilities = []
    for dataset in datasets:
        candidate = _optimization_candidate(
            spec, manifest, dataset, recipe_index
        )
        marker = _load_numerical_instability(
            spec, manifest, candidate, dataset
        )
        if marker is not None:
            instabilities.append(marker)
    if instabilities:
        return {
            "physical_family": group_name,
            "optimization_id": str(recipe["id"]),
            "optimization_recipe_json": json.dumps(
                resolved_recipe, sort_keys=True
            ),
            "eligible": False,
            "status": "invalid_numerical_instability",
            "ineligible_datasets_json": json.dumps(sorted({
                row["dataset"] for row in instabilities
            })),
            "ineligible_reasons_json": json.dumps(sorted({
                row["error_message"] for row in instabilities
            })),
            "instability_markers_json": json.dumps(
                instabilities, sort_keys=True
            ),
        }
    for dataset in datasets:
        candidate = _optimization_candidate(
            spec, manifest, dataset, recipe_index
        )
        per_seed = {}
        for seed in spec["paired_seeds"]:
            rows = {}
            for source in ("fmt", "raw_pca"):
                values = _read_csv(_result_path(
                    spec, candidate, dataset, seed, source
                ))
                if len(values) != 1:
                    raise RuntimeError(
                        f"incomplete optimization result {candidate['id']}/"
                        f"{dataset}/seed={seed}/{source}"
                    )
                rows[source] = values[0]
            paired_counts = {
                int(rows[source]["trainable_residual_parameter_count"])
                for source in rows
            }
            if len(paired_counts) != 1:
                raise RuntimeError("FMT and Raw-PCA parameter counts differ")
            parameter_counts.add(int(rows["fmt"]["parameter_count"]))
            per_seed[int(seed)] = {
                source: {
                    "f1": float(rows[source]["validation_f1"]),
                    "average_precision": float(
                        rows[source]["validation_average_precision"]
                    ),
                } for source in rows
            }
            seed_gains[int(seed)].append(
                per_seed[int(seed)]["fmt"]["f1"]
                - per_seed[int(seed)]["raw_pca"]["f1"]
            )
        means = {
            source: {
                metric: float(np.mean([
                    per_seed[int(seed)][source][metric]
                    for seed in spec["paired_seeds"]
                ]))
                for metric in ("f1", "average_precision")
            }
            for source in ("fmt", "raw_pca")
        }
        per_dataset[dataset] = {
            **means,
            "f1_gain": means["fmt"]["f1"] - means["raw_pca"]["f1"],
            "average_precision_gain": (
                means["fmt"]["average_precision"]
                - means["raw_pca"]["average_precision"]
            ),
        }
    f1_gains = [row["f1_gain"] for row in per_dataset.values()]
    ap_gains = [row["average_precision_gain"] for row in per_dataset.values()]
    return {
        "physical_family": group_name,
        "optimization_id": str(recipe["id"]),
        "optimization_recipe_json": json.dumps(
            resolved_recipe, sort_keys=True
        ),
        "eligible": True,
        "status": "",
        "ineligible_datasets_json": "[]",
        "ineligible_reasons_json": "[]",
        "instability_markers_json": "[]",
        "dataset_macro_fmt_f1": float(np.mean([
            row["fmt"]["f1"] for row in per_dataset.values()
        ])),
        "dataset_macro_raw_pca_f1": float(np.mean([
            row["raw_pca"]["f1"] for row in per_dataset.values()
        ])),
        "dataset_macro_fmt_average_precision": float(np.mean([
            row["fmt"]["average_precision"]
            for row in per_dataset.values()
        ])),
        "dataset_macro_raw_pca_average_precision": float(np.mean([
            row["raw_pca"]["average_precision"]
            for row in per_dataset.values()
        ])),
        "dataset_macro_f1_gain_vs_raw_pca": float(np.mean(f1_gains)),
        "dataset_macro_average_precision_gain_vs_raw_pca": float(
            np.mean(ap_gains)
        ),
        "positive_dataset_count": int(np.count_nonzero(np.asarray(f1_gains) > 0)),
        "worst_dataset_f1_gain": float(min(f1_gains)),
        "worst_seed_f1_gain": float(min(
            np.mean(values) for values in seed_gains.values()
        )),
        "minimum_total_parameter_count": min(parameter_counts),
        "maximum_total_parameter_count": max(parameter_counts),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
        "seed_gains_json": json.dumps({
            str(seed): float(np.mean(values))
            for seed, values in seed_gains.items()
        }, sort_keys=True),
    }


def _apply_absolute_fmt_guard(rows: list[dict], selection: dict) -> list[dict]:
    """Reject gap-only winners that lower absolute FMT quality.

    The comparison is performed separately inside each physical family because
    ``select`` calls this helper once per family.  The exact declared control
    therefore provides the same-family FMT F1 and Average Precision reference.
    """
    guard = selection.get("absolute_fmt_guard")
    if guard is None:
        return rows
    control_id = str(guard["control_optimization_id"])
    controls = [
        row for row in rows if str(row["optimization_id"]) == control_id
    ]
    if len(controls) != 1:
        raise RuntimeError(
            f"absolute FMT guard requires exactly one control {control_id!r}"
        )
    control = controls[0]
    if not bool(control.get("eligible", False)):
        raise RuntimeError(
            f"absolute FMT guard control {control_id!r} is ineligible"
        )
    control_f1 = float(control["dataset_macro_fmt_f1"])
    control_ap = float(control["dataset_macro_fmt_average_precision"])
    f1_tolerance = float(guard["f1_tolerance"])
    ap_tolerance = float(guard["average_precision_tolerance"])
    for row in rows:
        row["absolute_fmt_control_optimization_id"] = control_id
        row["absolute_fmt_control_f1"] = control_f1
        row["absolute_fmt_control_average_precision"] = control_ap
        if not bool(row.get("eligible", False)):
            row["absolute_fmt_guard_passed"] = False
            row["absolute_fmt_f1_delta_vs_control"] = ""
            row["absolute_fmt_average_precision_delta_vs_control"] = ""
            continue
        f1_delta = float(row["dataset_macro_fmt_f1"]) - control_f1
        ap_delta = (
            float(row["dataset_macro_fmt_average_precision"]) - control_ap
        )
        passed = (
            f1_delta >= -f1_tolerance and ap_delta >= -ap_tolerance
        )
        row["absolute_fmt_f1_delta_vs_control"] = f1_delta
        row["absolute_fmt_average_precision_delta_vs_control"] = ap_delta
        row["absolute_fmt_guard_passed"] = bool(passed)
        if not passed:
            row["eligible"] = False
            row["status"] = "absolute_fmt_guard_failed"
            reasons = [
                f"FMT F1 delta {f1_delta:+.9f} < {-f1_tolerance:+.9f}"
                if f1_delta < -f1_tolerance else "",
                f"FMT AP delta {ap_delta:+.9f} < {-ap_tolerance:+.9f}"
                if ap_delta < -ap_tolerance else "",
            ]
            row["ineligible_reasons_json"] = json.dumps(
                [reason for reason in reasons if reason]
            )
    return rows


def select(config_path: str) -> Path:
    spec = _load_optimization_spec(config_path)
    manifest = _load_manifest(spec)
    primary, leaderboard = {}, []
    selection = spec["optimization_selection"]
    required = (
        str(selection["primary_metric"]),
        *tuple(selection["tie_breakers"]),
    )
    for group_name in spec["groups"]:
        rows = [
            _candidate_summary(spec, manifest, group_name, recipe)
            for recipe in spec["optimization_candidates"]
        ]
        rows = _apply_absolute_fmt_guard(rows, selection)
        eligible = [row for row in rows if bool(row["eligible"])]
        if not eligible:
            raise RuntimeError(
                f"all Task3 optimization candidates are ineligible for "
                f"{group_name}"
            )
        for key in required:
            if any(key not in row for row in eligible):
                raise KeyError(f"unknown optimization selection key {key!r}")
        ranked = sorted(
            eligible,
            key=lambda row: tuple(float(row[key]) for key in required),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["rank_within_family"] = rank
            leaderboard.append(row)
        for row in rows:
            if not bool(row["eligible"]):
                row["rank_within_family"] = ""
                leaderboard.append(row)
        primary[group_name] = ranked[0]
    output_root = Path(spec["output_root"])
    _write_csv(output_root / "optimization_leaderboard.csv", leaderboard)
    dataset_details = []
    for group_name, row in primary.items():
        for dataset, metrics in json.loads(row["datasets_json"]).items():
            dataset_details.append({
                "physical_family": group_name,
                "dataset": dataset,
                "optimization_id": row["optimization_id"],
                **metrics,
            })
    f1_gain = float(np.mean([
        row["f1_gain"] for row in dataset_details
    ]))
    ap_gain = float(np.mean([
        row["average_precision_gain"] for row in dataset_details
    ]))
    fmt_f1 = float(np.mean([
        row["fmt"]["f1"] for row in dataset_details
    ]))
    raw_pca_f1 = float(np.mean([
        row["raw_pca"]["f1"] for row in dataset_details
    ]))
    fmt_ap = float(np.mean([
        row["fmt"]["average_precision"] for row in dataset_details
    ]))
    raw_pca_ap = float(np.mean([
        row["raw_pca"]["average_precision"] for row in dataset_details
    ]))
    target_gain = float(selection["target_dataset_macro_f1_gain"])
    target_absolute_fmt_f1 = selection.get("target_absolute_fmt_f1")
    absolute_target_reached = (
        None if target_absolute_fmt_f1 is None
        else fmt_f1 >= float(target_absolute_fmt_f1)
    )
    guard_text = (
        " Candidates must first pass the preregistered same-family absolute "
        "FMT F1 and Average Precision guard against the exact control."
        if selection.get("absolute_fmt_guard") is not None else ""
    )
    payload = {
        "experiment": spec["experiment"],
        "selection_rule": (
            "family-specific paired optimization: maximize development "
            "dataset-macro F1 gain of FMT over the same-width train-only "
            "Raw-PCA arm; ordered tie-break metrics are "
            f"{list(required[1:])}." + guard_text
        ),
        "optimization_config_sha256": spec["optimization_config_sha256"],
        "base_search_config_sha256": spec["base_search_config_sha256"],
        "upstream_selection_sha256": manifest[
            "upstream_selection_sha256"
        ],
        "preflight_manifest_sha256": _sha256(_manifest_path(spec)),
        "opened_only_exposed_development_populations": True,
        "confirmation_opened": False,
        "paired_seeds": spec["paired_seeds"],
        "ineligible_candidates": [
            {
                "physical_family": row["physical_family"],
                "optimization_id": row["optimization_id"],
                "status": row["status"],
                "ineligible_datasets_json": row[
                    "ineligible_datasets_json"
                ],
                "ineligible_reasons_json": row[
                    "ineligible_reasons_json"
                ],
                "instability_markers_json": row[
                    "instability_markers_json"
                ],
            }
            for row in leaderboard if not bool(row["eligible"])
        ],
        "primary_by_group": primary,
        "development_dataset_macro_f1_gain_vs_raw_pca": f1_gain,
        "development_dataset_macro_ap_gain_vs_raw_pca": ap_gain,
        "development_dataset_macro_fmt_f1": fmt_f1,
        "development_dataset_macro_raw_pca_f1": raw_pca_f1,
        "development_dataset_macro_fmt_average_precision": fmt_ap,
        "development_dataset_macro_raw_pca_average_precision": raw_pca_ap,
        "target_dataset_macro_f1_gain": target_gain,
        "target_reached": f1_gain >= target_gain,
        "target_absolute_fmt_f1": target_absolute_fmt_f1,
        "absolute_fmt_target_reached": absolute_target_reached,
        "joint_target_reached": (
            f1_gain >= target_gain and absolute_target_reached is not False
        ),
        "absolute_fmt_guard": selection.get("absolute_fmt_guard"),
        "dataset_details": dataset_details,
    }
    target = output_root / "optimization_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=("static-preflight", "preflight", "candidate", "select"),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    arguments = parser.parse_args()
    if arguments.mode == "static-preflight":
        static_preflight(arguments.config)
    elif arguments.mode == "preflight":
        preflight(arguments.config)
    elif arguments.mode == "select":
        select(arguments.config)
    elif arguments.job_index is None:
        parser.error("candidate mode requires --job-index")
    else:
        run_job(arguments.config, arguments.job_index)


if __name__ == "__main__":
    main()
