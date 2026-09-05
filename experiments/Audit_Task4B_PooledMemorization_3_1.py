"""Independent strict audit for ``Verify_Task4B_PooledMemorization_3.1``.

Versioned copy of Audit_Task4B_FullVolumeMemorization_1_1; contract literals
updated and source/voxel identity made volume-aware for the pooled cache.

The auditor deliberately does not import the training driver.  It reconstructs
the all-sample normalizer, epoch permutations, predictions, confusion matrices,
and four-class metrics directly from the frozen cache and saved artifacts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


AUDIT_VERSION = "Audit_Task4B_PooledMemorization_3.1"
EXPERIMENT = "Verify_Task4B_PooledMemorization_3.1"
VARIANTS = ("raw", "fmt_only", "raw_fmt")
SEEDS = (7068, 7069, 7070)
CLASS_NAMES = (
    "ordinary_streamwise",
    "ordinary_spanwise",
    "hairpin_head",
    "hairpin_limb",
)
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
NUMERIC_TOLERANCE = 1e-9
PROBABILITY_TOLERANCE = 2e-6


class AuditError(RuntimeError):
    """Raised when a memorization artifact violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_int64(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def epoch_permutation_sha256(sample_count: int, seed: int, epoch: int) -> str:
    """Recompute the trainer's documented ``default_rng`` permutation hash."""

    _require(sample_count > 0, "sample_count must be positive")
    _require(epoch > 0, "epoch must be positive")
    order = np.random.default_rng(int(seed) + int(epoch)).permutation(
        int(sample_count)
    )
    return _sha256_int64(order)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _assert_close(label: str, actual: Any, expected: Any, *, atol: float = NUMERIC_TOLERANCE) -> None:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric: {actual!r}") from exc
    _require(
        np.isfinite(actual_value) and np.isfinite(expected_value),
        f"{label} contains a non-finite value",
    )
    difference = abs(actual_value - expected_value)
    _require(
        difference <= float(atol),
        f"{label} differs by {difference:.17g}: "
        f"reported={actual_value:.17g}, recomputed={expected_value:.17g}",
    )


def _resolve_existing_file(path_text: str | Path, anchors: tuple[Path, ...]) -> Path:
    path = Path(path_text).expanduser()
    candidates = [path] if path.is_absolute() else [anchor / path for anchor in anchors]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise AuditError(f"cache file not found; checked: {checked}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"missing history artifact: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(rows, f"history artifact is empty: {path.name}")
    return rows


def _validate_config(spec: dict[str, Any]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    _require(spec.get("experiment") == EXPERIMENT, "unexpected experiment name")
    variants = tuple(str(value) for value in spec.get("variants", ()))
    seeds = tuple(int(value) for value in spec.get("training", {}).get("seeds", ()))
    _require(variants == VARIANTS, f"variant order must be {VARIANTS}, found {variants}")
    _require(seeds == SEEDS, f"seed order must be {SEEDS}, found {seeds}")
    contract = spec.get("diagnostic_contract", {})
    _require(contract.get("holdout") is False, "memorization config must disable holdout")
    _require(
        contract.get("evaluation_set") == "exactly_the_same_all_91711_rows",
        "evaluation set is not frozen as the same full cache",
    )
    training = spec.get("training", {})
    _require(
        training.get("sampling")
        == "one_without_replacement_permutation_of_every_row_per_epoch",
        "training sampler contract is not one complete permutation per epoch",
    )
    _require(training.get("early_stopping") is False, "early stopping must be disabled")
    _require(float(spec.get("model", {}).get("dropout", -1.0)) == 0.0, "dropout must be zero")
    _require(float(training.get("weight_decay", -1.0)) == 0.0, "weight decay must be zero")
    gate = spec.get("pass_gate", {})
    _require(
        bool(gate.get("experiment_requires_every_variant_seed_run_to_pass")),
        "pass gate does not require every run",
    )
    _require(
        bool(gate.get("require_positive_minimum_true_logit_margin")),
        "pass gate does not require a positive true-class margin",
    )
    _require(
        int(gate.get("required_consecutive_zero_error_epochs", 0)) >= 1,
        "required zero-error streak must be positive",
    )
    return variants, seeds


def _load_cache(
    cache_path: Path, spec: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    expected_sha = str(spec.get("cache_sha256", "")).lower()
    _require(
        len(expected_sha) == 64 and all(char in "0123456789abcdef" for char in expected_sha),
        "config cache_sha256 is not a 64-character hexadecimal digest",
    )
    actual_sha = _sha256(cache_path)
    _require(
        actual_sha == expected_sha,
        f"cache SHA-256 differs: expected={expected_sha}, actual={actual_sha}",
    )
    required = {
        "raw_features",
        "fmt_features",
        "labels",
        "split_codes",
        "vortex_ids",
        "source_candidate_indices",
        "voxel_indices_xyz",
        "seeds_xyz",
        "volume_codes",
        "metadata_json",
    }
    with np.load(cache_path, allow_pickle=False) as cache:
        _require(required.issubset(cache.files), f"cache lacks arrays: {sorted(required - set(cache.files))}")
        arrays = {
            "raw_features": np.asarray(cache["raw_features"], dtype=np.float32),
            "fmt_features": np.asarray(cache["fmt_features"], dtype=np.float32),
            "labels": np.asarray(cache["labels"], dtype=np.int64).reshape(-1),
            "split_codes": np.asarray(cache["split_codes"], dtype=np.int8).reshape(-1),
            "vortex_ids": np.asarray(cache["vortex_ids"], dtype=np.int32).reshape(-1),
            "source_candidate_indices": np.asarray(
                cache["source_candidate_indices"], dtype=np.int64
            ).reshape(-1),
            "voxel_indices_xyz": np.asarray(
                cache["voxel_indices_xyz"], dtype=np.int64
            ),
            "seeds_xyz": np.asarray(cache["seeds_xyz"], dtype=np.float64),
            "volume_codes": np.asarray(cache["volume_codes"], dtype=np.int64).reshape(-1),
        }
        metadata = json.loads(str(cache["metadata_json"]))
    count = len(arrays["labels"])
    gate = spec["pass_gate"]
    expected_count = int(gate["expected_sample_count"])
    expected_support = np.asarray(gate["expected_class_support"], dtype=np.int64)
    _require(count == expected_count, f"cache row count is {count}, expected {expected_count}")
    for name, array in arrays.items():
        _require(len(array) == count, f"cache {name} row count differs")
        _require(np.isfinite(array).all(), f"cache {name} contains non-finite values")
    _require(arrays["raw_features"].ndim == 2, "raw_features must be two-dimensional")
    _require(arrays["fmt_features"].ndim == 2, "fmt_features must be two-dimensional")
    _require(
        arrays["fmt_features"].shape[1] == int(spec["encoder"]["expected_feature_dim"]),
        "FMT feature width differs from config",
    )
    _require(set(np.unique(arrays["labels"])) == {0, 1, 2, 3}, "cache labels are not exactly 0..3")
    support = np.bincount(arrays["labels"], minlength=4)
    _require(np.array_equal(support, expected_support), f"class support differs: {support.tolist()}")
    _require(set(np.unique(arrays["split_codes"])) == set(SPLIT_NAMES), "original splits are not exactly 0,1,2")
    _require(
        np.array_equal(arrays["vortex_ids"] > 0, arrays["labels"] >= 2),
        "positive VortexId support disagrees with hairpin labels",
    )
    cache_encoder = metadata.get("encoder", {})
    for key in (
        "num_freq",
        "mode",
        "include_chirality",
        "neighbor_pool",
        "neighbor_scale",
        "neighbor_weight_after_train_standardization",
    ):
        _require(
            cache_encoder.get(key) == spec["encoder"].get(key),
            f"cache/config encoder mismatch for {key}",
        )
    return arrays, metadata, actual_sha


def exact_row_label_evidence(
    features: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Count bitwise-exact feature rows and their irreducible label errors."""

    matrix = np.asarray(features)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    _require(matrix.ndim == 2, "exact-row input must be a two-dimensional matrix")
    _require(len(matrix) == len(targets), "exact-row feature/label lengths differ")
    _require(len(matrix) > 0 and matrix.shape[1] > 0, "exact-row input is empty")
    _require(np.isin(targets, np.arange(4)).all(), "exact-row labels are not 0..3")
    contiguous = np.ascontiguousarray(matrix)
    row_byte_width = int(contiguous.dtype.itemsize * contiguous.shape[1])
    packed_rows = contiguous.view(np.dtype((np.void, row_byte_width))).reshape(-1)
    _, inverse, counts = np.unique(
        packed_rows,
        return_inverse=True,
        return_counts=True,
    )
    group_class_counts = np.zeros((len(counts), 4), dtype=np.int64)
    np.add.at(group_class_counts, (inverse, targets), 1)
    represented_classes = np.count_nonzero(group_class_counts, axis=1)
    conflicting_groups = represented_classes > 1
    minimum_errors_by_group = counts - group_class_counts.max(axis=1)
    return {
        "row_count": int(len(contiguous)),
        "column_count": int(contiguous.shape[1]),
        "dtype": contiguous.dtype.str,
        "row_byte_width": row_byte_width,
        "exact_unique_row_count": int(len(counts)),
        "exact_duplicate_row_count": int(np.sum(counts - 1)),
        "exact_duplicate_group_count": int(np.count_nonzero(counts > 1)),
        "maximum_exact_group_size": int(np.max(counts)),
        "cross_label_conflict_group_count": int(np.count_nonzero(conflicting_groups)),
        "cross_label_conflicting_row_count": int(np.sum(counts[conflicting_groups])),
        "irreducible_minimum_error_count": int(np.sum(minimum_errors_by_group)),
    }


def _exact_index_row_evidence(values: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    _require(matrix.ndim == 2 and len(matrix) > 0, "index identity array is invalid")
    contiguous = np.ascontiguousarray(matrix)
    row_byte_width = int(contiguous.dtype.itemsize * contiguous.shape[1])
    packed_rows = contiguous.view(np.dtype((np.void, row_byte_width))).reshape(-1)
    _, counts = np.unique(packed_rows, return_counts=True)
    return {
        "row_count": int(len(contiguous)),
        "exact_unique_row_count": int(len(counts)),
        "exact_duplicate_row_count": int(np.sum(counts - 1)),
        "exact_duplicate_group_count": int(np.count_nonzero(counts > 1)),
        "maximum_exact_group_size": int(np.max(counts)),
    }


def validate_cache_identity(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Reject exact feature-label conflicts and non-unique source mappings."""

    labels = arrays["labels"]
    representations = {
        "raw": exact_row_label_evidence(arrays["raw_features"], labels),
        "fmt": exact_row_label_evidence(arrays["fmt_features"], labels),
        "raw_fmt": exact_row_label_evidence(
            np.concatenate(
                (arrays["raw_features"], arrays["fmt_features"]), axis=1
            ),
            labels,
        ),
    }
    for name, evidence in representations.items():
        _require(
            evidence["cross_label_conflict_group_count"] == 0,
            f"{name} has {evidence['cross_label_conflict_group_count']} "
            "bitwise-exact cross-label feature groups",
        )
        _require(
            evidence["irreducible_minimum_error_count"] == 0,
            f"{name} has an irreducible minimum of "
            f"{evidence['irreducible_minimum_error_count']} classification errors",
        )
    # Pooled cache: identity is (volume, index) because channel and TBL index
    # spaces overlap.
    volume = np.asarray(arrays["volume_codes"], dtype=np.int64).reshape(-1, 1)
    source_evidence = _exact_index_row_evidence(
        np.column_stack((volume, arrays["source_candidate_indices"].reshape(-1, 1)))
    )
    voxel_evidence = _exact_index_row_evidence(
        np.column_stack((volume, arrays["voxel_indices_xyz"]))
    )
    _require(
        source_evidence["exact_duplicate_row_count"] == 0,
        "source_candidate_indices contain duplicate rows",
    )
    _require(
        voxel_evidence["exact_duplicate_row_count"] == 0,
        "voxel_indices_xyz contain duplicate rows",
    )
    return {
        "representations": representations,
        "source_candidate_indices": source_evidence,
        "voxel_indices_xyz": voxel_evidence,
        "strict_gate": {
            "all_representation_conflict_group_counts_zero": True,
            "all_representation_irreducible_minimum_errors_zero": True,
            "source_candidate_duplicate_rows_zero": True,
            "voxel_duplicate_rows_zero": True,
        },
    }


def recompute_all_sample_normalization(
    raw_features: np.ndarray,
    fmt_features: np.ndarray,
    *,
    sampled_steps: int,
    encoder_spec: dict[str, Any],
) -> dict[str, np.ndarray | int]:
    """Reconstruct normalization from every cache row without shared code."""

    raw = np.asarray(raw_features, dtype=np.float32).reshape(
        -1, 7, int(sampled_steps), 3
    )
    fmt = np.asarray(fmt_features, dtype=np.float32)
    raw_mean = raw.mean(axis=(0, 1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    raw_std = raw.std(axis=(0, 1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    raw_std = np.maximum(raw_std, np.float32(1e-6))
    fmt_mean = fmt.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    fmt_std = fmt.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    fmt_std = np.maximum(fmt_std, np.float32(1e-6))
    num_freq = int(encoder_spec["num_freq"])
    mode_width = 1 if str(encoder_spec["mode"]) == "magnitude" else 3
    base_width = num_freq * mode_width + (
        num_freq - 1 if bool(encoder_spec["include_chirality"]) else 0
    )
    normalized_raw = (raw - raw_mean) / raw_std
    normalized_fmt = (fmt - fmt_mean) / fmt_std
    normalized_fmt[:, base_width:] *= float(
        encoder_spec["neighbor_weight_after_train_standardization"]
    )
    _require(
        np.isfinite(normalized_raw).all() and np.isfinite(normalized_fmt).all(),
        "independent all-sample normalization produced non-finite values",
    )
    return {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "fmt_mean": fmt_mean,
        "fmt_std": fmt_std,
        "fmt_base_width": int(base_width),
    }


def _validate_normalization(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    _require(path.is_file(), f"missing required formal artifact: {path.name}")
    sampled_steps = int(metadata.get("streamlines", {}).get("sampled_steps", 0))
    _require(sampled_steps > 0, "cache metadata lacks sampled_steps")
    expected = recompute_all_sample_normalization(
        arrays["raw_features"],
        arrays["fmt_features"],
        sampled_steps=sampled_steps,
        encoder_spec=spec["encoder"],
    )
    expected_keys = set(expected)
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == expected_keys, f"normalization keys differ: {saved.files}")
        for key in ("raw_mean", "raw_std", "fmt_mean", "fmt_std"):
            actual = np.asarray(saved[key])
            wanted = np.asarray(expected[key])
            _require(actual.shape == wanted.shape, f"normalization {key} shape differs")
            _require(actual.dtype == wanted.dtype, f"normalization {key} dtype differs")
            _require(np.array_equal(actual, wanted), f"normalization {key} is not the all-row recomputation")
        actual_base = int(np.asarray(saved["fmt_base_width"]).item())
    _require(actual_base == int(expected["fmt_base_width"]), "normalization FMT base width differs")
    return {
        "sampled_steps": sampled_steps,
        "fit_row_count": int(len(arrays["labels"])),
        "raw_mean": np.asarray(expected["raw_mean"]).reshape(-1).tolist(),
        "raw_std": np.asarray(expected["raw_std"]).reshape(-1).tolist(),
        "fmt_base_width": actual_base,
        "artifact_sha256": _sha256(path),
    }


def metric_bundle(targets: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Compute four-class confusion, precision, recall, and F1 independently."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    _require(targets.shape == predicted.shape, "target/prediction lengths differ")
    _require(np.isin(targets, np.arange(4)).all(), "targets contain invalid class IDs")
    _require(np.isin(predicted, np.arange(4)).all(), "predictions contain invalid class IDs")
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(4, dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(4, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(4, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "sample_count": int(len(targets)),
        "accuracy": float(np.mean(targets == predicted)),
        "error_count": int(np.count_nonzero(targets != predicted)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_class_precision": {
            name: float(precision[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_recall": {
            name: float(recall[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_f1": {
            name: float(f1[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "support": {
            name: int(support[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _compare_named_metrics(reported: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ("sample_count", "error_count"):
        _require(int(reported.get(key, -1)) == int(expected[key]), f"{label} {key} differs")
    for key in ("accuracy", "macro_f1", "balanced_accuracy"):
        _assert_close(f"{label}.{key}", reported.get(key), expected[key])
    for key in ("per_class_precision", "per_class_recall", "per_class_f1"):
        values = reported.get(key, {})
        _require(set(values) == set(CLASS_NAMES), f"{label} {key} class keys differ")
        for class_name in CLASS_NAMES:
            _assert_close(
                f"{label}.{key}.{class_name}", values[class_name], expected[key][class_name]
            )
    support = reported.get("support", {})
    _require(set(support) == set(CLASS_NAMES), f"{label} support class keys differ")
    for class_name in CLASS_NAMES:
        _require(
            int(support[class_name]) == int(expected["support"][class_name]),
            f"{label} support differs for {class_name}",
        )
    _require(
        reported.get("confusion_matrix_true_rows_predicted_columns")
        == expected["confusion_matrix_true_rows_predicted_columns"],
        f"{label} confusion matrix differs",
    )


def _validate_prediction(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_keys = {
        "cache_row_indices",
        "source_candidate_indices",
        "voxel_indices_xyz",
        "seeds_xyz",
        "targets",
        "logits",
        "probabilities",
        "predicted_labels",
        "original_split_codes",
        "vortex_ids",
    }
    with np.load(path, allow_pickle=False) as artifact:
        _require(set(artifact.files) == required_keys, f"{path.name} prediction keys differ")
        cache_rows = np.asarray(artifact["cache_row_indices"], dtype=np.int64).reshape(-1)
        source_candidates = np.asarray(
            artifact["source_candidate_indices"], dtype=np.int64
        ).reshape(-1)
        voxel_indices = np.asarray(artifact["voxel_indices_xyz"], dtype=np.int64)
        seeds_xyz = np.asarray(artifact["seeds_xyz"], dtype=np.float64)
        targets = np.asarray(artifact["targets"], dtype=np.int64).reshape(-1)
        logits = np.asarray(artifact["logits"], dtype=np.float64)
        probabilities = np.asarray(artifact["probabilities"], dtype=np.float64)
        predicted_saved = np.asarray(artifact["predicted_labels"], dtype=np.int64).reshape(-1)
        split_codes = np.asarray(artifact["original_split_codes"], dtype=np.int8).reshape(-1)
        vortex_ids = np.asarray(artifact["vortex_ids"], dtype=np.int32).reshape(-1)
    count = len(arrays["labels"])
    expected_rows = np.arange(count, dtype=np.int64)
    _require(
        np.array_equal(cache_rows, expected_rows),
        f"{path.name} cache_row_indices are not every cache row in order",
    )
    _require(
        np.array_equal(source_candidates, arrays["source_candidate_indices"]),
        f"{path.name} source_candidate_indices differ from cache",
    )
    _require(
        np.array_equal(voxel_indices, arrays["voxel_indices_xyz"]),
        f"{path.name} voxel_indices_xyz differ from cache",
    )
    _require(
        np.array_equal(seeds_xyz, arrays["seeds_xyz"]),
        f"{path.name} seeds_xyz differ from cache",
    )
    _require(np.array_equal(targets, arrays["labels"]), f"{path.name} targets differ from cache labels")
    _require(np.array_equal(split_codes, arrays["split_codes"]), f"{path.name} original split codes differ")
    _require(np.array_equal(vortex_ids, arrays["vortex_ids"]), f"{path.name} VortexIds differ")
    _require(logits.shape == (count, 4), f"{path.name} logits shape differs")
    _require(probabilities.shape == (count, 4), f"{path.name} probability shape differs")
    _require(np.isfinite(logits).all(), f"{path.name} logits contain non-finite values")
    _require(np.isfinite(probabilities).all(), f"{path.name} probabilities contain non-finite values")
    _require(
        np.all((probabilities >= 0.0) & (probabilities <= 1.0)),
        f"{path.name} probabilities are outside [0,1]",
    )
    _require(
        np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE),
        f"{path.name} probability rows do not sum to one",
    )
    recomputed_probabilities = _stable_softmax(logits)
    maximum_probability_difference = float(
        np.max(np.abs(probabilities - recomputed_probabilities))
    )
    _require(
        maximum_probability_difference <= PROBABILITY_TOLERANCE,
        f"{path.name} probabilities differ from softmax(logits) by "
        f"{maximum_probability_difference:.9g}",
    )
    predicted = np.argmax(logits, axis=1).astype(np.int64)
    _require(np.array_equal(predicted_saved, predicted), f"{path.name} predicted_labels differ from logits")
    _require(
        np.array_equal(np.argmax(probabilities, axis=1), predicted),
        f"{path.name} probability argmax differs from logits",
    )
    metrics = metric_bundle(targets, predicted)
    true_logits = logits[np.arange(count), targets]
    alternatives = logits.copy()
    alternatives[np.arange(count), targets] = -np.inf
    margins = true_logits - alternatives.max(axis=1)
    minimum_margin = float(np.min(margins))
    mean_margin = float(np.mean(margins))
    _require(metrics["error_count"] == 0, f"{path.name} has {metrics['error_count']} memorization errors")
    _require(minimum_margin > 0.0, f"{path.name} minimum true-class logit margin is not positive")

    split_evidence: dict[str, Any] = {}
    for split_code, split_name in SPLIT_NAMES.items():
        member = split_codes == split_code
        _require(np.any(member), f"{path.name} original {split_name} split is empty")
        errors = int(np.count_nonzero(predicted[member] != targets[member]))
        _require(errors == 0, f"{path.name} has {errors} errors in original {split_name} split")
        split_evidence[split_name] = {"sample_count": int(member.sum()), "error_count": errors}

    vortex_evidence: dict[str, Any] = {}
    positive_ids = np.unique(vortex_ids[vortex_ids > 0])
    _require(len(positive_ids) > 0, f"{path.name} contains no positive VortexId")
    for vortex_id in positive_ids:
        member = vortex_ids == vortex_id
        errors = int(np.count_nonzero(predicted[member] != targets[member]))
        _require(errors == 0, f"{path.name} has {errors} errors for VortexId {int(vortex_id)}")
        vortex_evidence[str(int(vortex_id))] = {
            "sample_count": int(member.sum()),
            "error_count": errors,
        }
    evidence = {
        "cache_row_indices_equal_all_cache_rows": True,
        "source_candidate_indices_equal_cache": True,
        "voxel_indices_xyz_equal_cache": True,
        "seeds_xyz_equal_cache": True,
        "targets_equal_cache_labels": True,
        "probabilities_equal_softmax_logits": True,
        "maximum_probability_absolute_difference": maximum_probability_difference,
        "minimum_true_logit_margin": minimum_margin,
        "mean_true_logit_margin": mean_margin,
        "original_splits": split_evidence,
        "positive_vortex_ids": vortex_evidence,
        "prediction_sha256": _sha256(path),
    }
    return metrics, evidence


def _validate_history(
    path: Path,
    *,
    seed: int,
    sample_count: int,
    run: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    rows = _read_csv(path)
    executed = int(run.get("epochs_executed", -1))
    selected = int(run.get("selected_epoch", -1))
    _require(len(rows) == executed, f"{path.name} row count differs from epochs_executed")
    _require(executed >= 1, f"{path.name} has no executed epoch")
    _require(selected == executed, f"{path.name} selected epoch must be the final exact-fit epoch")
    required_streak = int(spec["pass_gate"]["required_consecutive_zero_error_epochs"])
    streak = 0
    hashes: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(rows, start=1):
        epoch = int(row.get("epoch", -1))
        _require(epoch == expected_epoch, f"{path.name} epochs are not contiguous from one")
        _require(int(row.get("draw_count", -1)) == sample_count, f"{path.name} epoch {epoch} draw_count differs")
        _require(int(row.get("unique_count", -1)) == sample_count, f"{path.name} epoch {epoch} is incomplete")
        _require(int(row.get("duplicate_count", -1)) == 0, f"{path.name} epoch {epoch} has duplicate draws")
        _require(int(row.get("missing_count", -1)) == 0, f"{path.name} epoch {epoch} has missing rows")
        expected_hash = epoch_permutation_sha256(sample_count, seed, epoch)
        actual_hash = str(row.get("order_sha256", ""))
        _require(
            actual_hash == expected_hash,
            f"{path.name} epoch {epoch} permutation hash differs from default_rng(seed+epoch)",
        )
        error_count = int(row.get("fit_error_count", -1))
        margin = float(row.get("minimum_true_logit_margin", "nan"))
        is_exact = error_count == 0 and np.isfinite(margin) and margin > 0.0
        streak = streak + 1 if is_exact else 0
        _require(
            int(row.get("zero_error_streak", -1)) == streak,
            f"{path.name} epoch {epoch} zero-error streak differs",
        )
        for field in (
            "train_cross_entropy",
            "fit_accuracy",
            "fit_macro_f1",
            "fit_balanced_accuracy",
            "mean_true_logit_margin",
            "learning_rate",
        ):
            _require(np.isfinite(float(row.get(field, "nan"))), f"{path.name} epoch {epoch} {field} is non-finite")
        hashes.append({"epoch": epoch, "order_sha256": expected_hash})
    final = rows[-1]
    _require(streak >= required_streak, f"{path.name} final exact-fit streak is {streak}, required {required_streak}")
    _require(int(final["fit_error_count"]) == 0, f"{path.name} final epoch is not zero-error")
    _assert_close(f"{path.name}.final_accuracy", final["fit_accuracy"], 1.0)
    _assert_close(f"{path.name}.final_macro_f1", final["fit_macro_f1"], 1.0)
    _assert_close(f"{path.name}.final_balanced_accuracy", final["fit_balanced_accuracy"], 1.0)
    _require(
        int(final["fit_error_count"]) == int(run["error_count"]),
        f"{path.name} final error count differs from run JSON",
    )
    for history_key, run_key in (
        ("fit_accuracy", "accuracy"),
        ("fit_macro_f1", "macro_f1"),
        ("fit_balanced_accuracy", "balanced_accuracy"),
        ("minimum_true_logit_margin", "minimum_true_logit_margin"),
    ):
        _assert_close(
            f"{path.name}.final.{history_key}",
            final[history_key],
            run[run_key],
            atol=1e-6,
        )
    history_sha256 = _sha256(path)
    _require(
        str(run.get("history_sha256", "")) == history_sha256,
        f"{path.name} SHA-256 differs from run JSON",
    )
    _require(
        int(run.get("terminal_zero_error_streak", -1)) == streak,
        f"{path.name} terminal zero-error streak differs from run JSON",
    )
    _require(
        int(run.get("required_zero_error_streak", -1)) == required_streak,
        f"{path.name} required zero-error streak differs from config",
    )
    return {
        "epoch_count": executed,
        "selected_epoch": selected,
        "final_zero_error_streak": streak,
        "all_epochs_complete_without_replacement": True,
        "all_epoch_hashes_recomputed": True,
        "epoch_permutation_hashes": hashes,
        "history_sha256": history_sha256,
    }


def _validate_run_report(
    run: dict[str, Any],
    metrics: dict[str, Any],
    prediction_evidence: dict[str, Any],
    *,
    variant: str,
    seed: int,
) -> None:
    label = f"run {variant}/seed{seed}"
    _require(run.get("variant") == variant, f"{label} variant differs")
    _require(int(run.get("seed", -1)) == seed, f"{label} seed differs")
    _require(bool(run.get("passed")), f"{label} is not reported as passed")
    _require(int(run.get("parameter_count", 0)) > 0, f"{label} parameter_count is not positive")
    _require(float(run.get("elapsed_seconds", 0.0)) > 0.0, f"{label} elapsed time is not positive")
    _require(
        run.get("checkpoint_policy") == "selected_state_in_memory_only_no_model_file",
        f"{label} checkpoint policy differs",
    )
    _compare_named_metrics(run, metrics, label)
    _assert_close(
        f"{label}.minimum_true_logit_margin",
        run.get("minimum_true_logit_margin"),
        prediction_evidence["minimum_true_logit_margin"],
        atol=1e-7,
    )
    _require(
        Path(str(run.get("prediction_path", ""))).name == f"{variant}_seed{seed}.npz",
        f"{label} prediction path basename differs",
    )
    _require(
        Path(str(run.get("history_path", ""))).name == f"{variant}_seed{seed}.csv",
        f"{label} history path basename differs",
    )


def _validate_summary(
    summary: dict[str, Any],
    spec: dict[str, Any],
    cache_sha256: str,
    runs: list[dict[str, Any]],
    *,
    config_path: Path,
    normalization_path: Path,
) -> None:
    expected_count = len(VARIANTS) * len(SEEDS)
    _require(summary.get("experiment") == EXPERIMENT, "summary experiment differs")
    _require(summary.get("cache_sha256") == cache_sha256, "summary cache SHA-256 differs")
    _require(
        summary.get("normalization_sha256")
        == _sha256(normalization_path),
        "summary normalization SHA-256 differs",
    )
    _require(summary.get("fit_equals_evaluation") is True, "summary does not state fit=evaluation")
    _require(summary.get("holdout") is False, "summary incorrectly claims a holdout")
    _require(summary.get("selected_subset_run") is False, "summary is a subset/smoke run")
    _require(int(summary.get("sample_count", -1)) == int(spec["pass_gate"]["expected_sample_count"]), "summary sample count differs")
    _require(
        [int(value) for value in summary.get("class_support", ())]
        == [int(value) for value in spec["pass_gate"]["expected_class_support"]],
        "summary class support differs",
    )
    aggregate = summary.get("aggregate", {})
    _require(int(aggregate.get("expected_run_count", -1)) == expected_count, "aggregate expected run count differs")
    _require(int(aggregate.get("completed_run_count", -1)) == expected_count, "aggregate is incomplete")
    _require(bool(aggregate.get("all_expected_runs_present")), "aggregate says expected runs are missing")
    _require(int(aggregate.get("passed_run_count", -1)) == expected_count, "aggregate is not 9/9 passed")
    _require(bool(aggregate.get("experiment_passed")), "aggregate experiment_passed is false")
    _require(int(aggregate.get("maximum_error_count", -1)) == 0, "aggregate maximum error count is not zero")
    reported_runs = aggregate.get("runs", [])
    _require(len(reported_runs) == expected_count, "summary aggregate does not contain nine runs")
    indexed_reported = {
        (str(row.get("variant")), int(row.get("seed"))): row for row in reported_runs
    }
    indexed_files = {(row["variant"], int(row["seed"])): row for row in runs}
    _require(set(indexed_reported) == set(indexed_files), "summary run set differs from run JSON files")
    summary_identity = summary.get("artifact_identity", {})
    identity_files = {
        "config_sha256": config_path,
        "trainer_sha256": config_path.parent.parent
        / "experiments"
        / "Verify_Task4B_PooledMemorization_3_1.py",
        "shared_training_helpers_sha256": config_path.parent.parent
        / "experiments"
        / "Train_Task4B_FourClassClassifier_1_1.py",
        "model_sha256": config_path.parent.parent
        / "FMT_Utils"
        / "Task4B_Classifier_3D.py",
        "raw_geometry_encoder_sha256": config_path.parent.parent
        / "FMT_Utils"
        / "PathlineClassifier_3D.py",
    }
    _require(
        set(summary_identity)
        == {"git_head", *identity_files},
        "summary artifact identity keys differ",
    )
    commit = str(summary_identity["git_head"]).lower()
    _require(
        len(commit) == 40 and all(char in "0123456789abcdef" for char in commit),
        "summary artifact identity lacks a hexadecimal full git commit",
    )
    for hash_name, source_path in identity_files.items():
        _require(source_path.is_file(), f"identity source file is missing: {source_path}")
        digest = str(summary_identity[hash_name]).lower()
        _require(
            len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
            f"summary artifact identity {hash_name} is not SHA-256",
        )
        _require(
            digest == _sha256(source_path),
            f"summary artifact identity {hash_name} differs from current source file",
        )
    for key in indexed_files:
        _require(
            _canonical_json(indexed_reported[key]) == _canonical_json(indexed_files[key]),
            f"summary run {key} differs from its run JSON",
        )
        _require(
            indexed_files[key].get("artifact_identity") == summary_identity,
            f"run {key} artifact identity differs from summary",
        )


def audit(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    write_json: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    _require(config_path.is_file(), f"config does not exist: {config_path}")
    _require(output_dir.is_dir(), f"output directory does not exist: {output_dir}")
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants, seeds = _validate_config(spec)

    summary_path = output_dir / "summary.json"
    snapshot_path = output_dir / "config_snapshot.yaml"
    normalization_path = output_dir / "normalization_all_samples.npz"
    for required in (summary_path, snapshot_path, normalization_path):
        _require(required.is_file(), f"missing required formal artifact: {required.name}")
    for directory_name in ("runs", "predictions", "histories"):
        _require((output_dir / directory_name).is_dir(), f"missing required artifact directory: {directory_name}")

    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    _require(_canonical_json(snapshot) == _canonical_json(spec), "config snapshot differs from supplied frozen config")
    cache_path = _resolve_existing_file(
        spec["cache"],
        (Path.cwd(), config_path.parent.parent, config_path.parent),
    )
    arrays, metadata, cache_sha256 = _load_cache(cache_path, spec)
    cache_identity_evidence = validate_cache_identity(arrays)
    normalization_evidence = _validate_normalization(
        normalization_path, arrays, metadata, spec
    )

    expected_pairs = {(variant, seed) for variant in variants for seed in seeds}
    expected_run_paths = {
        output_dir / "runs" / f"{variant}_seed{seed}.json"
        for variant, seed in expected_pairs
    }
    expected_prediction_paths = {
        output_dir / "predictions" / f"{variant}_seed{seed}.npz"
        for variant, seed in expected_pairs
    }
    expected_history_paths = {
        output_dir / "histories" / f"{variant}_seed{seed}.csv"
        for variant, seed in expected_pairs
    }
    for directory, expected, description in (
        (output_dir / "runs", expected_run_paths, "run JSON"),
        (output_dir / "predictions", expected_prediction_paths, "prediction"),
        (output_dir / "histories", expected_history_paths, "history"),
    ):
        actual = {path for path in directory.iterdir() if path.is_file()}
        missing = sorted(path.name for path in expected - actual)
        unexpected = sorted(path.name for path in actual - expected)
        _require(
            actual == expected,
            f"incomplete/invalid {description} artifacts: missing={missing}, unexpected={unexpected}",
        )

    model_files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]
    _require(not model_files, f"persistent model checkpoints found: {model_files}")

    audited_runs: list[dict[str, Any]] = []
    run_documents: list[dict[str, Any]] = []
    parameter_counts: dict[str, set[int]] = {variant: set() for variant in variants}
    for variant in variants:
        for seed in seeds:
            run_path = output_dir / "runs" / f"{variant}_seed{seed}.json"
            prediction_path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
            history_path = output_dir / "histories" / f"{variant}_seed{seed}.csv"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            metrics, prediction_evidence = _validate_prediction(prediction_path, arrays)
            _validate_run_report(
                run,
                metrics,
                prediction_evidence,
                variant=variant,
                seed=seed,
            )
            history_evidence = _validate_history(
                history_path,
                seed=seed,
                sample_count=len(arrays["labels"]),
                run=run,
                spec=spec,
            )
            parameter_counts[variant].add(int(run["parameter_count"]))
            run_documents.append(run)
            audited_runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "status": "PASS",
                    "metrics": metrics,
                    "prediction_checks": prediction_evidence,
                    "history_checks": history_evidence,
                    "run_json_sha256": _sha256(run_path),
                }
            )
    for variant, counts in parameter_counts.items():
        _require(len(counts) == 1, f"parameter count changes across seeds for {variant}: {counts}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_summary(
        summary,
        spec,
        cache_sha256,
        run_documents,
        config_path=config_path,
        normalization_path=normalization_path,
    )
    interpretation = str(summary.get("interpretation_boundary", ""))
    _require("no test or generalization evidence" in interpretation, "summary lacks the no-generalization boundary")

    payload = {
        "audit": AUDIT_VERSION,
        "status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "cache": str(cache_path),
            "cache_sha256": cache_sha256,
            "output_dir": str(output_dir),
        },
        "strict_gate": {
            "expected_run_count": 9,
            "audited_run_count": len(audited_runs),
            "passed_run_count": len(audited_runs),
            "all_runs_zero_error": True,
            "all_four_class_metrics_exact": True,
            "all_original_splits_zero_error": True,
            "all_positive_vortex_ids_zero_error": True,
            "all_epoch_permutations_complete_and_recomputed": True,
            "all_sample_normalization_recomputed": True,
            "all_representation_conflicts_and_minimum_errors_zero": True,
            "source_and_voxel_duplicate_rows_zero": True,
            "persistent_model_checkpoint_count": 0,
        },
        "cache_evidence": {
            "sample_count": int(len(arrays["labels"])),
            "class_support": np.bincount(arrays["labels"], minlength=4).tolist(),
            "original_split_support": {
                SPLIT_NAMES[code]: int(np.count_nonzero(arrays["split_codes"] == code))
                for code in SPLIT_NAMES
            },
            "positive_vortex_id_count": int(len(np.unique(arrays["vortex_ids"][arrays["vortex_ids"] > 0]))),
        },
        "normalization_evidence": normalization_evidence,
        "cache_identity_evidence": cache_identity_evidence,
        "parameter_counts": {
            variant: next(iter(parameter_counts[variant])) for variant in variants
        },
        "runs": audited_runs,
        "interpretation_boundary": (
            "All metrics use the same cache rows used for fitting. PASS verifies "
            "finite-sample memorization and artifact consistency only; it is not "
            "test or generalization evidence and does not compare representations."
        ),
    }
    if write_json is not None:
        destination = Path(write_json).expanduser().resolve()
        _require(destination.parent.is_dir(), f"audit output parent does not exist: {destination.parent}")
        _require(not destination.is_dir(), f"audit output is a directory: {destination}")
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen memorization YAML")
    parser.add_argument("--output-dir", required=True, help="Completed nine-run output directory")
    parser.add_argument("--write-json", help="Optional path for the independent audit JSON")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    try:
        result = audit(
            arguments.config,
            arguments.output_dir,
            write_json=arguments.write_json,
        )
    except (AuditError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"AUDIT FAILED: {error}") from error
    print(
        json.dumps(
            {
                "status": result["status"],
                "audit": result["audit"],
                "passed_run_count": result["strict_gate"]["passed_run_count"],
                "write_json": (
                    str(Path(arguments.write_json).resolve())
                    if arguments.write_json
                    else None
                ),
            },
            indent=2,
        )
    )
