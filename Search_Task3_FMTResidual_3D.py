"""Development-only family search for supervised 3D Task3 FMT residuals.

Every FMT candidate is paired with a train-only Raw-PCA residual of exactly
the same auxiliary width and trainable architecture.  Candidate selection
opens only the registered development train/validation ordinals.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.Task12Data_3D import feature_matrix, load_cache_records
from Verify_Task3_FMTClassifier import (
    _append_csv,
    _normalize_train_only,
    _portable_basename,
    _stack_split,
)
from Verify_Task3_FMTResidual import _train_one


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "groups", "datasets", "candidates",
        "screen_seeds", "screen_split", "training",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    datasets = list(spec["datasets"])
    memberships = [
        dataset for group in spec["groups"].values()
        for dataset in group["datasets"]
    ]
    if len(datasets) != len(set(datasets)) or sorted(datasets) != sorted(memberships):
        raise ValueError("groups must partition unique datasets exactly once")
    candidate_ids = [str(row["id"]) for row in spec["candidates"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ids must be unique")
    train = {int(value) for value in spec["screen_split"]["train_ordinals"]}
    validation = {
        int(value) for value in spec["screen_split"]["validation_ordinals"]
    }
    outer = {int(value) for value in spec.get("outer_ordinals", [])}
    if train & validation or (train | validation) & outer:
        raise ValueError("Task3 train, validation, and outer ordinals must be disjoint")
    for group in spec["groups"].values():
        for key in ("source_cache_root", "label_cache_root"):
            root = str(group[key]).lower()
            if "confirmation" in root or "test" in root:
                raise ValueError(
                    f"development search cannot read held-out {key}: {root}"
                )
    exposed_populations = (
        (
            "exposed_training",
            "exposed_training_source_cache_root",
            "exposed_training_label_cache_root",
        ),
        (
            "robust_validation",
            "exposed_spatial_source_cache_root",
            "exposed_spatial_label_cache_root",
        ),
    )
    for population_name, source_key, label_key in exposed_populations:
        population = spec.get(population_name)
        if population is None:
            continue
        if str(population.get("status", "")) != "exposed_development":
            raise ValueError(
                f"{population_name} must explicitly declare "
                "status: exposed_development"
            )
        ordinals = [int(value) for value in population.get("ordinals", [])]
        if not ordinals or len(ordinals) != len(set(ordinals)):
            raise ValueError(
                f"{population_name} ordinals must be unique and non-empty"
            )
        expected = int(population.get("expected_slices", 0))
        if expected <= 0 or any(not 0 <= value < expected for value in ordinals):
            raise ValueError(f"{population_name} ordinals exceed expected_slices")
        manifest_path = population.get("source_manifest")
        manifest_hash = str(population.get("source_manifest_sha256", ""))
        if manifest_path is not None or manifest_hash:
            if manifest_path is None or len(manifest_hash) != 64:
                raise ValueError(
                    f"{population_name} must provide a manifest path and full SHA-256"
                )
            manifest = Path(manifest_path)
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            actual_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            if actual_hash != manifest_hash.lower():
                raise RuntimeError(f"{population_name} source manifest changed")
        for group_name, group in spec["groups"].items():
            for key in (source_key, label_key):
                if key not in group:
                    raise ValueError(
                        f"{population_name} requires {key} for group {group_name}"
                    )
    training_population = spec.get("exposed_training")
    validation_population = spec.get("robust_validation")
    if training_population is not None and validation_population is not None:
        training_phase = tuple(training_population.get("seed_grid_phase", ()))
        validation_phase = tuple(validation_population.get("seed_grid_phase", ()))
        if training_phase and training_phase == validation_phase:
            raise ValueError(
                "exposed training and robust validation must use different "
                "spatial populations"
            )
    absolute_guard = dict(
        spec.get("selection", {}).get("absolute_fmt_guard", {})
    )
    if absolute_guard:
        tolerance = float(absolute_guard.get("tolerance", -1.0))
        controls = dict(absolute_guard.get("by_group", {}))
        source_selection = absolute_guard.get("source_selection")
        source_selection_sha = str(
            absolute_guard.get("source_selection_sha256", "")
        )
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("absolute FMT guard tolerance must be in [0,1)")
        if len(source_selection_sha) != 64:
            raise ValueError(
                "absolute FMT guard requires a full source selection SHA-256"
            )
        if source_selection is None:
            raise ValueError(
                "absolute FMT guard requires its source selection path"
            )
        source_selection_path = Path(source_selection)
        if not source_selection_path.exists():
            raise FileNotFoundError(source_selection_path)
        if hashlib.sha256(source_selection_path.read_bytes()).hexdigest() != (
            source_selection_sha.lower()
        ):
            raise RuntimeError("absolute FMT guard source selection changed")
        if set(controls) != set(spec["groups"]):
            raise ValueError(
                "absolute FMT guard must provide one control per physical family"
            )
        for group_name, control in controls.items():
            required_control = {"feature", "fmt_f1", "fmt_average_precision"}
            if not required_control.issubset(control):
                raise ValueError(
                    f"absolute FMT guard for {group_name} misses "
                    f"{sorted(required_control.difference(control))}"
                )
            for metric in ("fmt_f1", "fmt_average_precision"):
                value = float(control[metric])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"absolute FMT guard {group_name}.{metric} is invalid"
                    )
            feature = str(control["feature"])
            if [
                str(candidate["fmt_feature"])
                for candidate in spec["candidates"]
            ].count(feature) != 1:
                raise ValueError(
                    f"absolute FMT guard control {feature!r} for {group_name} "
                    "must appear exactly once in candidates"
                )
    target_absolute_fmt_f1 = spec.get("selection", {}).get(
        "target_absolute_fmt_f1"
    )
    if target_absolute_fmt_f1 is not None and not 0.0 <= float(
        target_absolute_fmt_f1
    ) <= 1.0:
        raise ValueError("target_absolute_fmt_f1 must be in [0,1]")
    return spec


def _group_for_dataset(spec: dict, dataset: str) -> tuple[str, dict]:
    matches = [
        (name, group) for name, group in spec["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} matched {len(matches)} groups")
    return matches[0]


def _frozen_raw_normalization(group: dict, dataset: str, seed: int) -> dict:
    """Load Raw coordinate statistics without changing the frozen backbone."""
    path = Path(group["raw_checkpoint_dir"]) / f"{dataset}_raw_seed{int(seed)}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "raw":
        raise ValueError(f"expected frozen Raw checkpoint at {path}")
    normalization = checkpoint.get("normalization", {})
    return {
        key: np.asarray(normalization[key], dtype=np.float32)
        for key in ("raw_mean", "raw_std")
    }


def _candidate(spec: dict, index: int) -> dict:
    index = int(index)
    if not 0 <= index < len(spec["candidates"]):
        raise IndexError(
            f"candidate index {index} outside [0,{len(spec['candidates'])})"
        )
    row = dict(spec["candidates"][index])
    row["index"] = index
    return row


def _load_records_from_roots(
    spec: dict,
    dataset: str,
    candidate: dict,
    device,
    *,
    source_cache_root,
    label_cache_root,
    expected_slices: int,
    ordinals,
) -> list[tuple]:
    required = sorted({int(value) for value in ordinals})
    source_dir = Path(source_cache_root) / dataset
    records = load_cache_records(
        source_dir,
        expected_count=int(expected_slices),
        ordinals=required,
    )
    result = []
    for record in records:
        label_path = Path(label_cache_root) / dataset / record["path"].name
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        with np.load(label_path) as label_file:
            labels = np.asarray(label_file["labels"], dtype=np.float32)
            metadata = json.loads(str(label_file["metadata_json"]))
        if _portable_basename(metadata["source_cache"]) != record["path"].name:
            raise ValueError(f"label/source mismatch for {record['path']}")
        expected_percentile = spec.get("expected_ivd_percentile")
        if expected_percentile is not None:
            actual_percentile = metadata.get(
                "label_value", metadata.get("ivd_percentile")
            )
            if actual_percentile is None or not np.isclose(
                float(actual_percentile), float(expected_percentile)
            ):
                raise RuntimeError(
                    f"Task3 label percentile mismatch in {label_path}: "
                    f"expected {expected_percentile}, found {actual_percentile}"
                )
        require_reference_match = bool(
            spec.get("require_source_reference_match", True)
        )
        if require_reference_match and not np.array_equal(
            labels.astype(bool), record["reference"]
        ):
            raise RuntimeError(
                f"Task3 global-IVD labels differ from source reference: {label_path}"
            )
        fmt = feature_matrix(record, str(candidate["fmt_feature"]), device)
        sampled_steps = record["raw"].shape[1] // (7 * 3)
        raw = record["raw"].reshape(-1, 7, sampled_steps, 3)
        if len(raw) != len(fmt) or len(raw) != len(labels):
            raise ValueError(f"feature/label length mismatch in {record['path']}")
        result.append((
            raw, fmt, labels, int(record["ordinal"]), metadata,
        ))
    return result


def _load_records(spec: dict, dataset: str, candidate: dict, device,
                  ordinals=None) -> list[tuple]:
    _, group = _group_for_dataset(spec, dataset)
    required = sorted(
        {int(value) for value in spec["screen_split"]["train_ordinals"]}
        | {int(value) for value in spec["screen_split"]["validation_ordinals"]}
    ) if ordinals is None else sorted({int(value) for value in ordinals})
    return _load_records_from_roots(
        spec,
        dataset,
        candidate,
        device,
        source_cache_root=group["source_cache_root"],
        label_cache_root=group["label_cache_root"],
        expected_slices=int(spec.get("expected_slices", 10)),
        ordinals=required,
    )


def _concatenate_splits(*splits):
    """Concatenate compatible ``(raw, auxiliary, labels)`` populations."""
    if not splits:
        raise ValueError("at least one split is required")
    return tuple(
        np.concatenate([split[index] for split in splits], axis=0)
        for index in range(3)
    )


def _load_search_splits(spec: dict, dataset: str, candidate: dict, device):
    """Load train and validation without opening current outer/final data.

    Optional spatial populations may point to confirmation sets from earlier
    experiments only when the config explicitly reclassifies them as exposed
    development data.  ``exposed_training`` is appended to both residual
    arms' common training population; ``robust_validation`` is appended to
    their common validation population.  The frozen Raw backbone itself is
    unchanged, so FMT and Raw-PCA retain the same trainable architecture.
    """
    base_records = _load_records(spec, dataset, candidate, device)
    train = _stack_split(
        base_records, spec["screen_split"]["train_ordinals"]
    )
    validation = _stack_split(
        base_records, spec["screen_split"]["validation_ordinals"]
    )
    _, group = _group_for_dataset(spec, dataset)
    exposed_training = spec.get("exposed_training")
    if exposed_training is not None:
        training_ordinals = [
            int(value) for value in exposed_training["ordinals"]
        ]
        training_records = _load_records_from_roots(
            spec,
            dataset,
            candidate,
            device,
            source_cache_root=group["exposed_training_source_cache_root"],
            label_cache_root=group["exposed_training_label_cache_root"],
            expected_slices=int(exposed_training["expected_slices"]),
            ordinals=training_ordinals,
        )
        spatial_training = _stack_split(training_records, training_ordinals)
        train = _concatenate_splits(train, spatial_training)
    robust = spec.get("robust_validation")
    if robust is not None:
        robust_ordinals = [int(value) for value in robust["ordinals"]]
        robust_records = _load_records_from_roots(
            spec,
            dataset,
            candidate,
            device,
            source_cache_root=group["exposed_spatial_source_cache_root"],
            label_cache_root=group["exposed_spatial_label_cache_root"],
            expected_slices=int(robust["expected_slices"]),
            ordinals=robust_ordinals,
        )
        spatial_validation = _stack_split(robust_records, robust_ordinals)
        validation = _concatenate_splits(validation, spatial_validation)
    return train, validation


def _fusion(candidate: dict) -> dict:
    if "fixed_alpha" in candidate:
        return {
            "fixed_alpha": float(candidate["fixed_alpha"]),
            "selection_metric": str(
                candidate.get("selection_metric", "average_precision")
            ),
        }
    return {
        "alpha_min": float(candidate.get("alpha_min", 0.0)),
        "alpha_max": float(candidate.get("alpha_max", 3.0)),
        "alpha_steps": int(candidate.get("alpha_steps", 61)),
        "selection_metric": str(candidate.get("selection_metric", "minimum_gain")),
        "minimum_f1_gain": float(candidate.get("minimum_f1_gain", 0.0)),
    }


def _training(spec: dict, candidate: dict, seed: int) -> dict:
    settings = dict(spec["training"])
    settings.update(candidate.get("training", {}))
    settings["seeds"] = [int(seed)]
    return settings


def _candidate_spec(spec: dict, group: dict, candidate: dict, dataset: str,
                    seed: int, source: str, output_dir: Path,
                    fmt_dim: int) -> dict:
    return {
        "experiment": f"{spec['experiment']}_{candidate['id']}_{source}",
        "source_cache_root": group["source_cache_root"],
        "label_cache_root": group["label_cache_root"],
        "raw_checkpoint_dir": group["raw_checkpoint_dir"],
        "output_dir": str(output_dir),
        "datasets": [dataset],
        "expected_slices": int(spec.get("expected_slices", 10)),
        "sampled_steps": int(spec.get("sampled_steps", 32)),
        "fmt_subset": candidate["fmt_feature"],
        "auxiliary_source": source,
        "raw_pca_components": int(fmt_dim),
        "raw_pca_random_state": int(spec.get("raw_pca_random_state", 7068)),
        "raw_wide_parameter_count": int(spec["raw_wide_parameter_count"]),
        "split": dict(spec["screen_split"]),
        "evaluation": {"test_enabled": False},
        "model": {
            "embedding_dim": int(candidate.get("embedding_dim", 128)),
            "auxiliary_dim": int(candidate.get("auxiliary_dim", 64)),
            "residual_input": str(candidate.get("residual_input", "geometry_fmt")),
            "head_architecture": str(candidate.get(
                "head_architecture", "mlp"
            )),
            "head_hidden_dim": int(candidate.get(
                "head_hidden_dim", candidate.get("embedding_dim", 128)
            )),
            "head_depth": int(candidate.get("head_depth", 2)),
            "bilinear_rank": int(candidate.get("bilinear_rank", 32)),
            "attention_heads": int(candidate.get("attention_heads", 4)),
            "head_dropout": float(candidate.get("head_dropout", 0.0)),
            "auxiliary_projection": str(candidate.get(
                "auxiliary_projection", "linear_layernorm_gelu"
            )),
            "auxiliary_hidden_dim": int(candidate.get(
                "auxiliary_hidden_dim", 64
            )),
        },
        "fusion": _fusion(candidate),
        "training": _training(spec, candidate, seed),
        "search_candidate": candidate,
    }


def _result_path(spec: dict, candidate: dict, dataset: str,
                 seed: int, source: str) -> Path:
    return (
        Path(spec["output_root"]) / "stage1" / "candidates"
        / str(candidate["id"]) / dataset / f"seed{int(seed)}"
        / source / "per_run.csv"
    )


def run_candidate(config_path: str, dataset: str, candidate_index: int) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    candidate = _candidate(spec, candidate_index)
    _, group = _group_for_dataset(spec, dataset)
    device_name = str(spec["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    train, validation = _load_search_splits(
        spec, dataset, candidate, device
    )
    raw_stats = _frozen_raw_normalization(
        group, dataset, int(spec["screen_seeds"][0])
    )
    train, validation, _, stats = _normalize_train_only(
        train, validation, raw_stats=raw_stats
    )
    fmt_dim = int(train[1].shape[1])
    if fmt_dim > train[0].reshape(len(train[0]), -1).shape[1]:
        raise ValueError(
            f"matched Raw-PCA cannot provide {fmt_dim} components for "
            f"{candidate['id']}"
        )
    last_path = None
    for seed_value in spec["screen_seeds"]:
        seed = int(seed_value)
        for source in ("fmt", "raw_pca"):
            result_path = _result_path(spec, candidate, dataset, seed, source)
            existing = [
                row for row in _read_csv(result_path)
                if row["dataset"] == dataset and int(row["seed"]) == seed
            ]
            if len(existing) > 1:
                raise RuntimeError(f"duplicate candidate result: {result_path}")
            if existing:
                print(f"cached {candidate['id']} {dataset} seed={seed} {source}")
                last_path = result_path
                continue
            output_dir = result_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, seed, source, output_dir, fmt_dim
            )
            (output_dir / "config_snapshot.yaml").write_text(
                yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8"
            )
            row = _train_one(
                run_spec, dataset, seed, (train, validation, None), stats,
                device, output_dir,
            )
            row.update({
                "candidate_id": candidate["id"],
                "candidate_index": int(candidate_index),
                "fmt_feature": candidate["fmt_feature"],
                "fmt_dim": fmt_dim,
            })
            _append_csv(result_path, row)
            last_path = result_path
            print(
                f"DONE {candidate['id']} {dataset} seed={seed} {source}: "
                f"F1={row['validation_f1']:.5f} "
                f"AP={row['validation_average_precision']:.5f}",
                flush=True,
            )
    return last_path


def _decode_job(spec: dict, job_index: int) -> tuple[str, int]:
    count = len(spec["datasets"]) * len(spec["candidates"])
    index = int(job_index)
    if not 0 <= index < count:
        raise IndexError(f"Task3 job index {index} outside [0,{count})")
    dataset_index, candidate_index = divmod(index, len(spec["candidates"]))
    return spec["datasets"][dataset_index], candidate_index


def run_job(config_path: str, job_index: int) -> Path:
    spec = _load_spec(config_path)
    dataset, candidate_index = _decode_job(spec, job_index)
    return run_candidate(
        config_path, dataset, candidate_index
    )


def _baseline_from_row(row: dict) -> dict:
    value = row.get("validation_selection_baseline", "")
    if isinstance(value, dict):
        parsed = value
    else:
        parsed = ast.literal_eval(str(value))
    return {
        "f1": float(parsed["f1"]),
        "average_precision": float(parsed["average_precision"]),
    }


def _candidate_summary(spec: dict, group_name: str, candidate: dict) -> dict:
    datasets = spec["groups"][group_name]["datasets"]
    seeds = [int(value) for value in spec["screen_seeds"]]
    metrics = ("f1", "average_precision")
    per_dataset = {}
    seed_f1_gains = {seed: [] for seed in seeds}
    for dataset in datasets:
        per_seed = {}
        for seed in seeds:
            rows = {}
            for source in ("fmt", "raw_pca"):
                path = _result_path(spec, candidate, dataset, seed, source)
                values = _read_csv(path)
                if len(values) != 1:
                    raise RuntimeError(
                        f"incomplete Task3 result {candidate['id']}/{dataset}/"
                        f"seed={seed}/{source}: {len(values)}"
                    )
                rows[source] = values[0]
            if {
                int(rows[source]["trainable_residual_parameter_count"])
                for source in ("fmt", "raw_pca")
            } != {int(rows["fmt"]["trainable_residual_parameter_count"])}:
                raise RuntimeError(
                    f"FMT/Raw-PCA trainable parameter mismatch for "
                    f"{candidate['id']}/{dataset}/seed={seed}"
                )
            strong = _baseline_from_row(rows["fmt"])
            fmt = {metric: float(rows["fmt"][f"validation_{metric}"])
                   for metric in metrics}
            raw_pca = {
                metric: float(rows["raw_pca"][f"validation_{metric}"])
                for metric in metrics
            }
            per_seed[seed] = {"fmt": fmt, "raw_pca": raw_pca, "strong_raw": strong}
            seed_f1_gains[seed].append(fmt["f1"] - raw_pca["f1"])
        per_dataset[dataset] = {}
        for source in ("fmt", "raw_pca", "strong_raw"):
            per_dataset[dataset][source] = {
                metric: float(np.mean([
                    per_seed[seed][source][metric] for seed in seeds
                ])) for metric in metrics
            }
        per_dataset[dataset]["gains"] = {
            f"{metric}_vs_raw_pca": (
                per_dataset[dataset]["fmt"][metric]
                - per_dataset[dataset]["raw_pca"][metric]
            ) for metric in metrics
        }
        per_dataset[dataset]["gains"].update({
            f"{metric}_vs_strong_raw": (
                per_dataset[dataset]["fmt"][metric]
                - per_dataset[dataset]["strong_raw"][metric]
            ) for metric in metrics
        })
    macro = {}
    for source in ("fmt", "raw_pca", "strong_raw"):
        macro[source] = {
            metric: float(np.mean([
                value[source][metric] for value in per_dataset.values()
            ])) for metric in metrics
        }
    gains = {
        f"{metric}_vs_raw_pca": macro["fmt"][metric] - macro["raw_pca"][metric]
        for metric in metrics
    }
    gains.update({
        f"{metric}_vs_strong_raw": macro["fmt"][metric] - macro["strong_raw"][metric]
        for metric in metrics
    })
    tolerance = float(
        spec.get("selection", {}).get("allowed_fmt_below_strong_raw", 0.005)
    )
    guard = min(
        gains["f1_vs_strong_raw"], gains["average_precision_vs_strong_raw"]
    ) >= -tolerance
    seed_gains = {
        str(seed): float(np.mean(values)) for seed, values in seed_f1_gains.items()
    }
    return {
        "group": group_name,
        "candidate_id": candidate["id"],
        "fmt_feature": candidate["fmt_feature"],
        "dataset_count": len(datasets),
        "fmt_f1_macro": macro["fmt"]["f1"],
        "raw_pca_f1_macro": macro["raw_pca"]["f1"],
        "strong_raw_f1_macro": macro["strong_raw"]["f1"],
        "fmt_minus_raw_pca_f1_macro": gains["f1_vs_raw_pca"],
        "fmt_minus_strong_raw_f1_macro": gains["f1_vs_strong_raw"],
        "fmt_ap_macro": macro["fmt"]["average_precision"],
        "raw_pca_ap_macro": macro["raw_pca"]["average_precision"],
        "strong_raw_ap_macro": macro["strong_raw"]["average_precision"],
        "fmt_minus_raw_pca_ap_macro": gains["average_precision_vs_raw_pca"],
        "fmt_minus_strong_raw_ap_macro": gains["average_precision_vs_strong_raw"],
        "worst_seed_f1_gain": min(seed_gains.values()),
        "all_seed_f1_gains_positive": min(seed_gains.values()) > 0.0,
        "strong_raw_guard_passed": guard,
        "seed_gains_json": json.dumps(seed_gains, sort_keys=True),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
    }


def _selection_key(row: dict) -> tuple[float, ...]:
    """Rank by the registered Task3 FMT versus Raw-PCA residual comparison.

    Strong Raw and Raw-wide remain important reported baselines, but they are
    different model routes and do not replace the same-structure Raw-PCA arm
    used to isolate the contribution of FMT.
    """
    default = (
        float(row["fmt_minus_raw_pca_f1_macro"]),
        float(row["fmt_minus_raw_pca_ap_macro"]),
        float(row["worst_seed_f1_gain"]),
        float(row["fmt_f1_macro"]),
    )
    if "absolute_fmt_guard_passed" not in row:
        return default
    return (float(bool(row["absolute_fmt_guard_passed"])), *default)


def _apply_absolute_fmt_guard(spec: dict, group_name: str,
                              row: dict) -> dict:
    """Reject gain-only winners that materially lower absolute FMT quality."""
    guard = dict(spec.get("selection", {}).get("absolute_fmt_guard", {}))
    if not guard:
        return row
    control = dict(guard["by_group"][group_name])
    tolerance = float(guard["tolerance"])
    result = dict(row)
    result.update({
        "absolute_fmt_control_feature": str(control["feature"]),
        "absolute_fmt_control_f1": float(control["fmt_f1"]),
        "absolute_fmt_control_ap": float(control["fmt_average_precision"]),
        "fmt_f1_delta_vs_absolute_control": (
            float(row["fmt_f1_macro"]) - float(control["fmt_f1"])
        ),
        "fmt_ap_delta_vs_absolute_control": (
            float(row["fmt_ap_macro"])
            - float(control["fmt_average_precision"])
        ),
    })
    result["absolute_fmt_guard_passed"] = bool(
        result["fmt_f1_delta_vs_absolute_control"] >= -tolerance
        and result["fmt_ap_delta_vs_absolute_control"] >= -tolerance
    )
    return result


def select(config_path: str) -> Path:
    spec = _load_spec(config_path)
    top_k = int(spec.get("selection", {}).get("top_k", 3))
    leaderboard = []
    selected = {}
    for group_name in spec["groups"]:
        rows = [
            _apply_absolute_fmt_guard(
                spec, group_name,
                _candidate_summary(spec, group_name, candidate),
            )
            for candidate in spec["candidates"]
        ]
        if (
            spec.get("selection", {}).get("absolute_fmt_guard")
            and not any(row["absolute_fmt_guard_passed"] for row in rows)
        ):
            raise RuntimeError(
                f"no {group_name} candidate preserves the frozen absolute "
                "FMT F1 and Average Precision control"
            )
        ranked = sorted(
            rows,
            key=_selection_key,
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["rank_within_group"] = rank
            leaderboard.append(row)
        selected[group_name] = ranked[:top_k]
    output = Path(spec["output_root"])
    _write_csv(output / "stage1_leaderboard.csv", leaderboard)
    primary = {group: rows[0] for group, rows in selected.items()}
    dataset_rows = []
    for group, row in primary.items():
        details = json.loads(row["datasets_json"])
        for dataset, metrics in details.items():
            dataset_rows.append({"group": group, "dataset": dataset, **metrics})
    f1_gain = float(np.mean([
        row["gains"]["f1_vs_raw_pca"] for row in dataset_rows
    ]))
    ap_gain = float(np.mean([
        row["gains"]["average_precision_vs_raw_pca"] for row in dataset_rows
    ]))
    fmt_f1 = float(np.mean([
        row["fmt"]["f1"] for row in dataset_rows
    ]))
    raw_pca_f1 = float(np.mean([
        row["raw_pca"]["f1"] for row in dataset_rows
    ]))
    fmt_ap = float(np.mean([
        row["fmt"]["average_precision"] for row in dataset_rows
    ]))
    raw_pca_ap = float(np.mean([
        row["raw_pca"]["average_precision"] for row in dataset_rows
    ]))
    target_gain = float(
        spec.get("selection", {}).get("target_dataset_macro_f1_gain", 0.15)
    )
    payload = {
        "experiment": spec["experiment"],
        "selection_rule": (
            "family-specific: first require absolute FMT F1 and Average "
            "Precision to remain within the preregistered tolerance of the "
            "frozen family control, then maximize validation F1 gain over "
            "the same-width Raw-PCA residual; tie-break by AP gain, worst "
            "seed, and absolute FMT F1"
            if spec.get("selection", {}).get("absolute_fmt_guard") else
            "family-specific: maximize validation F1 gain over the same-width "
            "Raw-PCA residual; tie-break by AP gain, worst seed, and absolute "
            "FMT F1. Strong Raw is reported but does not select the recipe"
        ),
        "opened_ordinals": sorted(
            set(spec["screen_split"]["train_ordinals"])
            | set(spec["screen_split"]["validation_ordinals"])
        ),
        "exposed_spatial_training": spec.get("exposed_training"),
        "exposed_spatial_validation": spec.get("robust_validation"),
        "outer_ordinals_opened": False,
        "confirmation_opened": False,
        "top_k_by_group": selected,
        "primary_by_group": primary,
        "absolute_fmt_guard": spec.get("selection", {}).get(
            "absolute_fmt_guard"
        ),
        "development_dataset_macro_fmt_f1": fmt_f1,
        "development_dataset_macro_raw_pca_f1": raw_pca_f1,
        "development_dataset_macro_f1_gain_vs_raw_pca": f1_gain,
        "development_dataset_macro_fmt_ap": fmt_ap,
        "development_dataset_macro_raw_pca_ap": raw_pca_ap,
        "development_dataset_macro_ap_gain_vs_raw_pca": ap_gain,
        "target_absolute_fmt_f1": spec.get("selection", {}).get(
            "target_absolute_fmt_f1"
        ),
        "absolute_fmt_f1_target_reached": (
            fmt_f1 >= float(spec["selection"]["target_absolute_fmt_f1"])
            if spec.get("selection", {}).get("target_absolute_fmt_f1")
            is not None else None
        ),
        "development_target_gain": target_gain,
        "development_target_reached": f1_gain >= target_gain,
        "dataset_details": dataset_rows,
    }
    target = output / "stage1_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("candidate", "select"), required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--candidate-index", type=int)
    args = parser.parse_args()
    if args.mode == "select":
        select(args.config)
    elif args.job_index is not None:
        run_job(args.config, args.job_index)
    elif args.dataset is not None and args.candidate_index is not None:
        run_candidate(args.config, args.dataset, args.candidate_index)
    else:
        parser.error("candidate mode requires --job-index or dataset/candidate-index")


if __name__ == "__main__":
    main()
