"""Read-only integrity and metric audit for formal Task4-b 1.2 outputs.

The audit does not import the training driver and never modifies the experiment
directory.  It writes files only when explicit ``--write-json`` or
``--write-manifest`` paths are supplied.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import yaml

from FMT_Utils.Task4B_Classifier_3D import PathlineMulticlassClassifier3D
from FMT_Utils.Task4A_StreamlineClustering_3D import (
    SPANWISE_CLASS,
    STREAMWISE_CLASS,
    summarize_topology,
    topology_proxy_rows,
)


AUDIT_VERSION = "Audit_Task4B_Supervised_1.2"
ABSOLUTE_TOLERANCE = 1e-10
PROBABILITY_TOLERANCE = 1e-6
CANONICAL_VARIANTS = ("raw", "raw_wide", "fmt_only", "raw_fmt")
CANONICAL_CLASS_NAMES = (
    "ordinary_streamwise",
    "ordinary_spanwise",
    "hairpin_head",
    "hairpin_limb",
)
SPLIT_NAMES = ("train", "validation", "test")
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt"}


class AuditError(RuntimeError):
    """Raised when a formal-result contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _assert_close(label: str, actual: Any, expected: Any) -> None:
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric: {actual!r}") from exc
    _require(
        np.isfinite(actual_float) and np.isfinite(expected_float),
        f"{label} contains a non-finite value",
    )
    difference = abs(actual_float - expected_float)
    _require(
        difference <= ABSOLUTE_TOLERANCE,
        f"{label} differs by {difference:.17g}: "
        f"reported={actual_float:.17g}, recomputed={expected_float:.17g}",
    )


def _resolve_existing_path(path_text: str | Path, anchors: tuple[Path, ...]) -> Path:
    path = Path(path_text).expanduser()
    candidates = [path] if path.is_absolute() else [anchor / path for anchor in anchors]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise AuditError(f"file not found; checked: {rendered}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), f"CSV is empty: {path}")
    return rows


def _class_names(spec: dict[str, Any]) -> tuple[str, ...]:
    classes = spec.get("taxonomy", {}).get("classes", {})
    names = tuple(str(classes.get(index, classes.get(str(index), ""))) for index in range(4))
    _require(
        names == CANONICAL_CLASS_NAMES,
        f"unexpected Task4-b taxonomy: {names!r}",
    )
    return names


def _metric_bundle(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    _require(
        probabilities.shape == (len(targets), len(class_names)),
        f"probability shape {probabilities.shape} is incompatible with targets {targets.shape}",
    )
    labels = np.arange(len(class_names), dtype=np.int64)
    predicted = np.argmax(probabilities, axis=1)
    per_f1 = f1_score(
        targets, predicted, labels=labels, average=None, zero_division=0
    )
    per_precision = precision_score(
        targets, predicted, labels=labels, average=None, zero_division=0
    )
    per_recall = recall_score(
        targets, predicted, labels=labels, average=None, zero_division=0
    )
    one_hot = np.eye(len(class_names), dtype=np.float64)[targets]
    per_ap = average_precision_score(one_hot, probabilities, average=None)
    return {
        "sample_count": int(len(targets)),
        "macro_f1": float(np.mean(per_f1)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_average_precision_ovr": float(np.mean(per_ap)),
        "per_class_f1": {
            name: float(per_f1[index]) for index, name in enumerate(class_names)
        },
        "per_class_precision": {
            name: float(per_precision[index])
            for index, name in enumerate(class_names)
        },
        "per_class_recall": {
            name: float(per_recall[index])
            for index, name in enumerate(class_names)
        },
        "per_class_average_precision_ovr": {
            name: float(per_ap[index]) for index, name in enumerate(class_names)
        },
        "support": {
            name: int(np.count_nonzero(targets == index))
            for index, name in enumerate(class_names)
        },
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(
            targets, predicted, labels=labels
        ).tolist(),
    }


def _flatten_metrics(
    prefix: str, metrics: dict[str, Any], class_names: tuple[str, ...]
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name in (
        "sample_count",
        "macro_f1",
        "balanced_accuracy",
        "macro_average_precision_ovr",
    ):
        flat[f"{prefix}_{name}"] = metrics[name]
    for class_name in class_names:
        flat[f"{prefix}_f1_{class_name}"] = metrics["per_class_f1"][class_name]
        flat[f"{prefix}_precision_{class_name}"] = metrics[
            "per_class_precision"
        ][class_name]
        flat[f"{prefix}_recall_{class_name}"] = metrics["per_class_recall"][
            class_name
        ]
        flat[f"{prefix}_ap_{class_name}"] = metrics[
            "per_class_average_precision_ovr"
        ][class_name]
        flat[f"{prefix}_support_{class_name}"] = metrics["support"][class_name]
    flat[f"{prefix}_confusion_matrix_true_rows_predicted_columns"] = metrics[
        "confusion_matrix_true_rows_predicted_columns"
    ]
    return flat


def _compare_flat_metrics(
    location: str,
    reported: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for field, expected_value in expected.items():
        _require(field in reported, f"{location} lacks {field}")
        actual_value = reported[field]
        if field.endswith("confusion_matrix_true_rows_predicted_columns"):
            if isinstance(actual_value, str):
                try:
                    actual_value = ast.literal_eval(actual_value)
                except (SyntaxError, ValueError) as exc:
                    raise AuditError(
                        f"{location}.{field} is not a matrix literal"
                    ) from exc
            _require(
                actual_value == expected_value,
                f"{location}.{field} differs from independently recomputed confusion matrix",
            )
        elif field.endswith("sample_count") or "_support_" in field:
            try:
                actual_integer = int(actual_value)
            except (TypeError, ValueError) as exc:
                raise AuditError(f"{location}.{field} is not an integer") from exc
            _require(
                actual_integer == int(expected_value),
                f"{location}.{field}: reported={actual_integer}, expected={expected_value}",
            )
        else:
            _assert_close(f"{location}.{field}", actual_value, expected_value)


def _validate_formal_config(spec: dict[str, Any]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    _require(
        spec.get("experiment") == "mainExp_Task4B_1.2",
        f"not the formal Task4-b 1.2 experiment: {spec.get('experiment')!r}",
    )
    variants = tuple(str(value) for value in spec.get("variants", ()))
    seeds = tuple(int(value) for value in spec.get("training", {}).get("seeds", ()))
    _require(variants == CANONICAL_VARIANTS, f"unexpected variants or order: {variants}")
    _require(len(seeds) == 3 and len(set(seeds)) == 3, f"expected three unique seeds, got {seeds}")
    _require(
        len(variants) * len(seeds) == 12,
        "formal Task4-b output must contain exactly 12 runs",
    )
    _require(
        spec.get("training", {}).get("selection_metric") == "validation_macro_f1",
        "selection metric must be validation_macro_f1",
    )
    _require(
        "no_persistent_model_file"
        in str(spec.get("training", {}).get("checkpoint_policy", "")),
        "formal config does not prohibit persistent model checkpoints",
    )
    return variants, seeds


def _validate_cache(
    cache_path: Path,
    expected_sha256: str,
    summary: dict[str, Any],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    actual_sha256 = _sha256(cache_path)
    _require(
        actual_sha256 == str(expected_sha256).lower(),
        f"cache SHA-256 mismatch: config={expected_sha256}, actual={actual_sha256}",
    )
    _require(
        summary.get("cache_sha256") == actual_sha256,
        "summary cache SHA-256 differs from the frozen cache",
    )
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {
            "raw_features",
            "fmt_features",
            "labels",
            "split_codes",
            "source_candidate_indices",
            "vortex_ids",
            "voxel_indices_xyz",
            "resolution_xyz",
            "metadata_json",
        }
        _require(required.issubset(cache.files), f"cache lacks keys: {sorted(required - set(cache.files))}")
        labels = np.asarray(cache["labels"], dtype=np.int64).reshape(-1)
        split_codes = np.asarray(cache["split_codes"], dtype=np.int64).reshape(-1)
        source_candidates = np.asarray(cache["source_candidate_indices"], dtype=np.int64).reshape(-1)
        vortex_ids = np.asarray(cache["vortex_ids"], dtype=np.int64).reshape(-1)
        voxel_indices_xyz = np.asarray(
            cache["voxel_indices_xyz"], dtype=np.int64
        )
        resolution_xyz = np.asarray(cache["resolution_xyz"], dtype=np.int64).reshape(-1)
        fmt_dim = int(np.asarray(cache["fmt_features"]).shape[1])
        row_count = int(len(labels))
        _require(
            len(split_codes) == row_count
            and len(source_candidates) == row_count
            and len(vortex_ids) == row_count
            and voxel_indices_xyz.shape == (row_count, 3),
            "cache row arrays have inconsistent lengths",
        )
        _require(resolution_xyz.shape == (3,), "cache resolution_xyz must have length three")
        _require(
            np.all(voxel_indices_xyz >= 0)
            and np.all(voxel_indices_xyz < resolution_xyz[None, :]),
            "cache voxel indices lie outside resolution_xyz",
        )
        _require(set(np.unique(labels).tolist()) == {0, 1, 2, 3}, "cache labels are not exactly the four Task4-b classes")
        _require(set(np.unique(split_codes).tolist()) == {0, 1, 2}, "cache split codes are not exactly train/validation/test")
        _require(len(np.unique(source_candidates)) == row_count, "cache source_candidate_indices are not unique")
        metadata = json.loads(str(cache["metadata_json"]))

    certificate = metadata.get("split_nonoverlap_certificate")
    _require(isinstance(certificate, dict), "cache has no split non-overlap certificate")
    minimum_gap = float(certificate.get("minimum_cyclic_x_gap", np.nan))
    required_gap = float(certificate.get("required_minimum_gap", np.nan))
    _require(
        certificate.get("certified") is True
        and np.isfinite(minimum_gap)
        and np.isfinite(required_gap)
        and minimum_gap > required_gap,
        f"invalid split non-overlap certificate: {certificate}",
    )
    _require(
        _canonical_json(summary.get("cache_split_nonoverlap_certificate"))
        == _canonical_json(certificate),
        "summary split certificate differs from cache metadata",
    )

    split_counts = {
        split_name: {
            class_name: int(
                np.count_nonzero((split_codes == split_code) & (labels == class_id))
            )
            for class_id, class_name in enumerate(class_names)
        }
        for split_code, split_name in enumerate(SPLIT_NAMES)
    }
    _require(
        _canonical_json(summary.get("split_counts")) == _canonical_json(split_counts),
        "summary split counts differ from cache rows",
    )
    metadata_counts = {
        (str(item["split"]), str(item["class_name"])): int(item["cached_count"])
        for item in metadata.get("final_counts", ())
    }
    expected_metadata_counts = {
        (split_name, class_name): count
        for split_name, counts in split_counts.items()
        for class_name, count in counts.items()
    }
    _require(
        metadata_counts == expected_metadata_counts,
        "cache metadata final_counts differ from cache labels/splits",
    )
    return {
        "sha256": actual_sha256,
        "row_count": row_count,
        "fmt_feature_dim": fmt_dim,
        "labels": labels,
        "split_codes": split_codes,
        "vortex_ids": vortex_ids,
        "voxel_indices_xyz": voxel_indices_xyz,
        "resolution_xyz": resolution_xyz,
        "split_counts": split_counts,
        "split_nonoverlap_certificate": certificate,
    }


def _expected_parameter_counts(
    spec: dict[str, Any], variants: tuple[str, ...], fmt_dim: int
) -> dict[str, int]:
    model_spec = spec["model"]
    _require(int(model_spec.get("num_classes", -1)) == 4, "model must have four classes")
    counts: dict[str, int] = {}
    for variant in variants:
        model = PathlineMulticlassClassifier3D(
            variant=variant,
            fmt_dim=fmt_dim,
            num_classes=4,
            temporal_width=int(model_spec["temporal_width"]),
            embedding_dim=int(model_spec["embedding_dim"]),
            auxiliary_dim=int(model_spec["auxiliary_dim"]),
            dropout=float(model_spec["dropout"]),
        )
        counts[variant] = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
    return counts


def _validate_history(
    path: Path,
    csv_run: dict[str, Any],
    summary_run: dict[str, Any],
    validation_metrics: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    rows = _read_csv(path)
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["validation_macro_f1"]) for row in rows], dtype=np.float64)
    _require(np.isfinite(scores).all(), f"history contains non-finite validation scores: {path}")
    _require(np.array_equal(epochs, np.arange(1, len(rows) + 1)), f"history epochs are not consecutive: {path}")
    min_delta = float(spec["training"]["min_delta"])
    running_best = -np.inf
    expected_best_flags: list[int] = []
    expected_stale_epochs: list[int] = []
    selected_index = -1
    stale = 0
    for index, score in enumerate(scores):
        improved = bool(score > running_best + min_delta)
        expected_best_flags.append(int(improved))
        if improved:
            running_best = float(score)
            selected_index = index
            stale = 0
        else:
            stale += 1
        expected_stale_epochs.append(stale)
    _require(selected_index >= 0, f"history contains no selected epoch: {path}")
    selected_epoch = int(epochs[selected_index])
    selected_score = float(scores[selected_index])
    csv_best_epoch = int(csv_run["best_epoch"])
    summary_best_epoch = int(summary_run["best_epoch"])
    _require(
        csv_best_epoch == summary_best_epoch == selected_epoch,
        f"best epoch violates the sequential min_delta rule in {path}: "
        f"CSV={csv_best_epoch}, summary={summary_best_epoch}, selected={selected_epoch}",
    )
    _assert_close(f"{path}.best_validation_macro_f1", csv_run["best_validation_macro_f1"], selected_score)
    _assert_close(f"{path}.summary_best_validation_macro_f1", summary_run["best_validation_macro_f1"], selected_score)
    _assert_close(f"{path}.prediction_validation_macro_f1", validation_metrics["macro_f1"], selected_score)

    actual_best_flags = [int(row["is_best"]) for row in rows]
    _require(actual_best_flags == expected_best_flags, f"history is_best flags violate min_delta in {path}")
    actual_stale_epochs = [int(row["stale_epochs"]) for row in rows]
    _require(
        actual_stale_epochs == expected_stale_epochs,
        f"history stale_epochs violate the sequential min_delta rule in {path}",
    )
    flagged_epochs = [int(row["epoch"]) for row in rows if int(row["is_best"]) == 1]
    _require(flagged_epochs and flagged_epochs[-1] == selected_epoch, f"last is_best marker is not the selected epoch in {path}")

    max_epochs = int(spec["training"]["max_epochs"])
    patience = int(spec["training"]["patience"])
    _require(len(rows) <= max_epochs, f"history exceeds max_epochs in {path}")
    if len(rows) < max_epochs:
        _require(int(rows[-1]["stale_epochs"]) == patience, f"history stopped before max_epochs at the wrong patience count in {path}")
    return {
        "epoch_count": int(len(rows)),
        "best_epoch": selected_epoch,
        "best_validation_macro_f1": selected_score,
        "stopped_by_patience": bool(len(rows) < max_epochs),
    }


def _validate_prediction(
    path: Path,
    labels: np.ndarray,
    split_codes: np.ndarray,
    class_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as prediction:
        required = {
            f"{split}_{suffix}"
            for split in ("validation", "test")
            for suffix in ("source_indices", "targets", "probabilities")
        }
        _require(required == set(prediction.files), f"unexpected prediction NPZ keys in {path}: {prediction.files}")
        result: dict[str, dict[str, Any]] = {}
        test_arrays: dict[str, np.ndarray] = {}
        for split_code, split_name in ((1, "validation"), (2, "test")):
            source = np.asarray(prediction[f"{split_name}_source_indices"])
            targets = np.asarray(prediction[f"{split_name}_targets"])
            probabilities = np.asarray(prediction[f"{split_name}_probabilities"])
            _require(np.issubdtype(source.dtype, np.integer), f"source indices are not integers in {path}")
            _require(np.issubdtype(targets.dtype, np.integer), f"targets are not integers in {path}")
            source = source.astype(np.int64, copy=False).reshape(-1)
            targets = targets.astype(np.int64, copy=False).reshape(-1)
            expected_source = np.flatnonzero(split_codes == split_code)
            _require(np.array_equal(source, expected_source), f"{split_name} source indices are missing, duplicated or reordered in {path}")
            _require(np.array_equal(targets, labels[source]), f"{split_name} targets differ from cache labels in {path}")
            _require(probabilities.shape == (len(source), 4), f"invalid {split_name} probability shape in {path}: {probabilities.shape}")
            _require(np.isfinite(probabilities).all(), f"non-finite {split_name} probabilities in {path}")
            _require(float(np.min(probabilities)) >= -PROBABILITY_TOLERANCE, f"negative {split_name} probability in {path}")
            _require(float(np.max(probabilities)) <= 1.0 + PROBABILITY_TOLERANCE, f"{split_name} probability exceeds one in {path}")
            _require(
                np.allclose(
                    probabilities.sum(axis=1),
                    1.0,
                    rtol=0.0,
                    atol=PROBABILITY_TOLERANCE,
                ),
                f"{split_name} probability rows do not sum to one in {path}",
            )
            result[split_name] = _metric_bundle(targets, probabilities, class_names)
            if split_name == "test":
                test_arrays = {
                    "source_indices": source.copy(),
                    "targets": targets.copy(),
                    "predicted": np.argmax(probabilities, axis=1).astype(
                        np.int64, copy=False
                    ),
                }
    return result, test_arrays


def _topology_rows_from_four_classes(
    vortex_ids: np.ndarray,
    voxel_indices_xyz: np.ndarray,
    four_classes: np.ndarray,
) -> list[dict[str, Any]]:
    """Apply the frozen Task4-a 26-neighbour component topology per instance."""

    ids = np.asarray(vortex_ids, dtype=np.int64).reshape(-1)
    indices = np.asarray(voxel_indices_xyz, dtype=np.int64)
    classes = np.asarray(four_classes, dtype=np.int64).reshape(-1)
    _require(
        indices.shape == (len(ids), 3) and len(classes) == len(ids),
        "topology inputs have inconsistent shapes",
    )
    rows: list[dict[str, Any]] = []
    for vortex_id in np.unique(ids[ids > 0]):
        member = ids == vortex_id
        member_xyz = indices[member]
        local_xyz = member_xyz - member_xyz.min(axis=0)
        size_xyz = local_xyz.max(axis=0) + 1
        shape_zyx = tuple(int(value) for value in size_xyz[::-1])
        ix, iy, iz = local_xyz.T
        instances = np.zeros(shape_zyx, dtype=np.int32)
        semantic = np.zeros(shape_zyx, dtype=np.int8)
        instances[iz, iy, ix] = int(vortex_id)
        local_classes = classes[member]
        local_semantic = np.zeros(len(local_classes), dtype=np.int8)
        local_semantic[local_classes == 3] = STREAMWISE_CLASS
        local_semantic[local_classes == 2] = SPANWISE_CLASS
        semantic[iz, iy, ix] = local_semantic
        instance_rows = topology_proxy_rows(
            instances,
            semantic,
            dominant_min_voxels=3,
            dominant_min_fraction=0.05,
        )
        _require(
            len(instance_rows) == 1
            and int(instance_rows[0]["vortex_id"]) == int(vortex_id),
            f"frozen topology returned the wrong instance for VortexId {vortex_id}",
        )
        rows.append(instance_rows[0])
    return rows


def _hairpin_instance_metrics(
    source_indices: np.ndarray,
    targets: np.ndarray,
    predicted: np.ndarray,
    cache_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Recover held-out hairpins and score anatomy with IDs equally weighted."""

    source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    guess = np.asarray(predicted, dtype=np.int64).reshape(-1)
    _require(
        len(source) == len(truth) == len(guess),
        "test source, target and prediction lengths differ",
    )
    ids = cache_evidence["vortex_ids"][source]
    indices = cache_evidence["voxel_indices_xyz"][source]
    hairpin = ids > 0
    test_ids = tuple(int(value) for value in np.unique(ids[hairpin]))
    _require(len(test_ids) == 6, f"expected six held-out VortexIds, got {test_ids}")
    _require(
        np.all(np.isin(truth[hairpin], (2, 3))),
        "positive test VortexIds contain non-hairpin proxy classes",
    )

    per_vortex: list[dict[str, Any]] = []
    for vortex_id in test_ids:
        member = ids == vortex_id
        member_truth = truth[member]
        member_guess = guess[member]
        per_f1 = f1_score(
            member_truth,
            member_guess,
            labels=np.asarray((2, 3), dtype=np.int64),
            average=None,
            zero_division=0,
        )
        per_vortex.append(
            {
                "vortex_id": vortex_id,
                "voxel_count": int(np.count_nonzero(member)),
                "head_count": int(np.count_nonzero(member_truth == 2)),
                "limb_count": int(np.count_nonzero(member_truth == 3)),
                "head_f1": float(per_f1[0]),
                "limb_f1": float(per_f1[1]),
                "head_limb_macro_f1": float(np.mean(per_f1)),
                "predicted_as_ordinary_count": int(
                    np.count_nonzero(np.isin(member_guess, (0, 1)))
                ),
            }
        )
    instance_scores = np.asarray(
        [row["head_limb_macro_f1"] for row in per_vortex], dtype=np.float64
    )
    topology_rows = _topology_rows_from_four_classes(
        ids[hairpin], indices[hairpin], guess[hairpin]
    )
    topology_summary = summarize_topology(topology_rows)
    _require(
        int(topology_summary["vortex_id_count"]) == 6,
        "supervised topology did not return all six held-out VortexIds",
    )
    return {
        "vortex_ids": list(test_ids),
        "per_vortex": per_vortex,
        "equal_vortex_id_head_limb_macro_f1_mean": float(
            np.mean(instance_scores)
        ),
        "equal_vortex_id_head_limb_macro_f1_median": float(
            np.median(instance_scores)
        ),
        "topology_definition": {
            "source": "FMT_Utils.Task4A_StreamlineClustering_3D.topology_proxy_rows",
            "hairpin_limb_semantic": "streamwise",
            "hairpin_head_semantic": "spanwise",
            "ordinary_prediction_inside_hairpin": "unassigned",
            "connectivity": "26-neighbour",
            "dominant_min_voxels": 3,
            "dominant_min_fraction": 0.05,
        },
        "topology": topology_summary,
        "topology_per_vortex": topology_rows,
    }


def _distribution(values_by_seed: dict[int, float]) -> dict[str, Any]:
    seeds = sorted(values_by_seed)
    values = np.asarray([values_by_seed[seed] for seed in seeds], dtype=np.float64)
    sample_std = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    half_width = float(4.302652729911275 * sample_std / np.sqrt(len(values)))
    return {
        "mean": mean,
        "sample_std": sample_std,
        "per_seed": {str(seed): float(values_by_seed[seed]) for seed in seeds},
        "positive_seed_count": int(np.count_nonzero(values > 0.0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "exploratory_seed_level_t95_interval": [
            mean - half_width,
            mean + half_width,
        ],
    }


def _aggregate_recomputed(
    runs: dict[tuple[str, int], dict[str, Any]],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"variants": {}, "paired_raw_fmt_minus_raw": {}}
    global_metrics = ("macro_f1", "balanced_accuracy", "macro_average_precision_ovr")
    per_class_metrics = (
        ("f1", "per_class_f1"),
        ("precision", "per_class_precision"),
        ("recall", "per_class_recall"),
        ("average_precision_ovr", "per_class_average_precision_ovr"),
    )
    for variant in variants:
        item: dict[str, Any] = {
            "run_count": len(seeds),
            "parameter_count": int(runs[(variant, seeds[0])]["parameter_count"]),
            "test": {},
            "per_class_test": {},
            "test_hairpin_instances": {},
        }
        for metric in global_metrics:
            item["test"][metric] = _distribution(
                {seed: runs[(variant, seed)]["test"][metric] for seed in seeds}
            )
        for class_name in class_names:
            item["per_class_test"][class_name] = {}
            for short_name, metric_key in per_class_metrics:
                item["per_class_test"][class_name][short_name] = _distribution(
                    {
                        seed: runs[(variant, seed)]["test"][metric_key][class_name]
                        for seed in seeds
                    }
                )
        hairpin_extractors = {
            "equal_vortex_id_head_limb_macro_f1_mean": lambda item: item[
                "equal_vortex_id_head_limb_macro_f1_mean"
            ],
            "equal_vortex_id_head_limb_macro_f1_median": lambda item: item[
                "equal_vortex_id_head_limb_macro_f1_median"
            ],
            "hard_2plus1_success_count_of_6": lambda item: item["topology"][
                "hard_2plus1_success_count_all"
            ],
            "soft_topology_score_mean": lambda item: item["topology"][
                "mean_soft_topology_score_all"
            ],
            "soft_topology_score_median": lambda item: item["topology"][
                "median_soft_topology_score_all"
            ],
        }
        for metric_name, extract in hairpin_extractors.items():
            item["test_hairpin_instances"][metric_name] = _distribution(
                {
                    seed: float(extract(runs[(variant, seed)]["test_hairpin_instances"]))
                    for seed in seeds
                }
            )
        aggregate["variants"][variant] = item

    for metric in global_metrics:
        aggregate["paired_raw_fmt_minus_raw"][metric] = _distribution(
            {
                seed: runs[("raw_fmt", seed)]["test"][metric]
                - runs[("raw", seed)]["test"][metric]
                for seed in seeds
            }
        )
    aggregate["paired_raw_fmt_minus_raw"]["per_class"] = {}
    for class_name in class_names:
        aggregate["paired_raw_fmt_minus_raw"]["per_class"][class_name] = {}
        for short_name, metric_key in per_class_metrics:
            aggregate["paired_raw_fmt_minus_raw"]["per_class"][class_name][short_name] = _distribution(
                {
                    seed: runs[("raw_fmt", seed)]["test"][metric_key][class_name]
                    - runs[("raw", seed)]["test"][metric_key][class_name]
                    for seed in seeds
                }
            )
    aggregate["paired_raw_fmt_minus_raw"]["test_hairpin_instances"] = {}
    hairpin_extractors = {
        "equal_vortex_id_head_limb_macro_f1_mean": lambda item: item[
            "equal_vortex_id_head_limb_macro_f1_mean"
        ],
        "equal_vortex_id_head_limb_macro_f1_median": lambda item: item[
            "equal_vortex_id_head_limb_macro_f1_median"
        ],
        "hard_2plus1_success_count_of_6": lambda item: item["topology"][
            "hard_2plus1_success_count_all"
        ],
        "soft_topology_score_mean": lambda item: item["topology"][
            "mean_soft_topology_score_all"
        ],
        "soft_topology_score_median": lambda item: item["topology"][
            "median_soft_topology_score_all"
        ],
    }
    for metric_name, extract in hairpin_extractors.items():
        aggregate["paired_raw_fmt_minus_raw"]["test_hairpin_instances"][
            metric_name
        ] = _distribution(
            {
                seed: float(
                    extract(runs[("raw_fmt", seed)]["test_hairpin_instances"])
                    - extract(runs[("raw", seed)]["test_hairpin_instances"])
                )
                for seed in seeds
            }
        )
    aggregate["paired_additional_controls"] = {}
    for comparison_name, minuend, subtrahend in (
        ("raw_fmt_minus_raw_wide", "raw_fmt", "raw_wide"),
        ("raw_wide_minus_raw", "raw_wide", "raw"),
        ("fmt_only_minus_raw", "fmt_only", "raw"),
    ):
        comparison: dict[str, Any] = {}
        for metric in global_metrics:
            comparison[metric] = _distribution(
                {
                    seed: runs[(minuend, seed)]["test"][metric]
                    - runs[(subtrahend, seed)]["test"][metric]
                    for seed in seeds
                }
            )
        comparison["per_class_f1"] = {
            class_name: _distribution(
                {
                    seed: runs[(minuend, seed)]["test"]["per_class_f1"][class_name]
                    - runs[(subtrahend, seed)]["test"]["per_class_f1"][class_name]
                    for seed in seeds
                }
            )
            for class_name in class_names
        }
        aggregate["paired_additional_controls"][comparison_name] = comparison
    return aggregate


def _compare_reported_aggregate(
    reported: dict[str, Any],
    recomputed: dict[str, Any],
    variants: tuple[str, ...],
    class_names: tuple[str, ...],
) -> None:
    reported_variants = reported.get("variants", {})
    _require(set(reported_variants) == set(variants), "summary aggregate variant set is wrong")
    for variant in variants:
        actual = reported_variants[variant]
        expected = recomputed["variants"][variant]
        _require(int(actual["run_count"]) == 3, f"summary {variant} run_count is not three")
        _require(int(actual["parameter_count"]) == expected["parameter_count"], f"summary {variant} parameter_count differs")
        for metric in ("macro_f1", "balanced_accuracy", "macro_average_precision_ovr"):
            dist = expected["test"][metric]
            _assert_close(f"summary.aggregate.{variant}.test_{metric}_mean", actual[f"test_{metric}_mean"], dist["mean"])
            _assert_close(f"summary.aggregate.{variant}.test_{metric}_std", actual[f"test_{metric}_std"], dist["sample_std"])
        for class_name in class_names:
            _assert_close(
                f"summary.aggregate.{variant}.test_f1_{class_name}_mean",
                actual[f"test_f1_{class_name}_mean"],
                expected["per_class_test"][class_name]["f1"]["mean"],
            )

    paired = reported.get("paired_comparisons", {})
    for metric in ("macro_f1", "balanced_accuracy", "macro_average_precision_ovr"):
        key = f"raw_fmt_minus_raw_test_{metric}"
        _require(key in paired, f"summary lacks paired comparison {key}")
        actual = paired[key]
        expected = recomputed["paired_raw_fmt_minus_raw"][metric]
        _assert_close(f"summary.aggregate.{key}.mean", actual["mean"], expected["mean"])
        _assert_close(f"summary.aggregate.{key}.std", actual["std"], expected["sample_std"])
        _require(set(actual.get("per_seed", {})) == set(expected["per_seed"]), f"summary {key} seed set differs")
        for seed, value in expected["per_seed"].items():
            _assert_close(f"summary.aggregate.{key}.per_seed.{seed}", actual["per_seed"][seed], value)


def _artifact_manifest(
    config_path: Path,
    cache_path: Path,
    output_dir: Path,
    excluded_paths: set[Path],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    repository_root = config_path.parent.parent
    code_candidates = (
        Path(__file__).resolve(),
        repository_root / "experiments/Train_Task4B_FourClassClassifier_1_1.py",
        repository_root / "FMT_Utils" / "Task4B_Classifier_3D.py",
        repository_root / "FMT_Utils" / "Task4A_StreamlineClustering_3D.py",
    )
    sources = [("config", config_path), ("cache", cache_path)]
    sources.extend(("code", path) for path in code_candidates if path.is_file())
    sources.extend(
        ("output_artifact", path)
        for path in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.resolve() not in excluded_paths
    )
    seen: set[Path] = set()
    for role, path in sources:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            display = "output/" + resolved.relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            display = str(resolved)
        entries.append(
            {
                "role": role,
                "path": display,
                "bytes": int(resolved.stat().st_size),
                "sha256": _sha256(resolved),
            }
        )
    return {"algorithm": "SHA-256", "files": entries}


def audit(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    write_json: str | Path | None = None,
    write_manifest: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    _require(config_path.is_file(), f"config does not exist: {config_path}")
    _require(output_dir.is_dir(), f"output directory does not exist: {output_dir}")
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants, seeds = _validate_formal_config(spec)
    class_names = _class_names(spec)

    summary_path = output_dir / "summary.json"
    results_path = output_dir / "per_run_metrics.csv"
    snapshot_path = output_dir / "config_snapshot.yaml"
    normalization_path = output_dir / "normalization_train_only.npz"
    for required in (summary_path, results_path, snapshot_path, normalization_path):
        _require(required.is_file(), f"missing formal artifact: {required}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(summary.get("experiment") == spec["experiment"], "summary experiment differs from config")
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    _require(_canonical_json(snapshot) == _canonical_json(spec), "config snapshot differs from supplied config")

    device = summary.get("device", {})
    _require(device.get("type") == "cuda", f"formal result was not recorded on CUDA: {device}")
    _require("A100" in str(device.get("name", "")).upper(), f"formal result was not recorded on an A100: {device}")
    _require(bool(str(device.get("torch_version", "")).strip()), "summary lacks PyTorch version")
    _require("test never selects" in str(summary.get("selection", "")), "summary lacks the no-test-selection policy")
    _require("no .pt/.pth/.ckpt" in str(summary.get("checkpoint_policy", "")), "summary checkpoint policy is not explicit")

    cache_path = _resolve_existing_path(
        spec["cache"],
        (Path.cwd(), config_path.parent.parent, config_path.parent),
    )
    cache_evidence = _validate_cache(
        cache_path, spec.get("cache_sha256", ""), summary, class_names
    )
    parameter_counts = _expected_parameter_counts(
        spec, variants, int(cache_evidence["fmt_feature_dim"])
    )

    expected_pairs = {(variant, seed) for variant in variants for seed in seeds}
    expected_prediction_paths = {
        output_dir / "predictions" / f"{variant}_seed{seed}.npz"
        for variant, seed in expected_pairs
    }
    expected_history_paths = {
        output_dir / "histories" / f"{variant}_seed{seed}.csv"
        for variant, seed in expected_pairs
    }
    prediction_dir = output_dir / "predictions"
    history_dir = output_dir / "histories"
    _require(prediction_dir.is_dir() and history_dir.is_dir(), "predictions or histories directory is missing")
    actual_prediction_paths = {path for path in prediction_dir.iterdir() if path.is_file()}
    actual_history_paths = {path for path in history_dir.iterdir() if path.is_file()}
    _require(actual_prediction_paths == expected_prediction_paths, "prediction directory is not exactly the expected 12 NPZ files")
    _require(actual_history_paths == expected_history_paths, "history directory is not exactly the expected 12 CSV files")

    model_files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]
    _require(not model_files, f"persistent model checkpoints found: {model_files}")

    csv_rows = _read_csv(results_path)
    json_rows = summary.get("runs", [])
    _require(len(csv_rows) == len(json_rows) == 12, "CSV/summary must each contain exactly 12 runs")

    def keyed(rows: list[dict[str, Any]], location: str) -> dict[tuple[str, int], dict[str, Any]]:
        result: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("variant")), int(row.get("seed")))
            _require(key not in result, f"duplicate {location} run: {key}")
            result[key] = row
        _require(set(result) == expected_pairs, f"{location} run set differs from config: {set(result) ^ expected_pairs}")
        return result

    csv_by_run = keyed(csv_rows, "CSV")
    json_by_run = keyed(json_rows, "summary")
    expected_test_source = np.flatnonzero(cache_evidence["split_codes"] == 2)
    proxy_hairpin_reference = _hairpin_instance_metrics(
        expected_test_source,
        cache_evidence["labels"][expected_test_source],
        cache_evidence["labels"][expected_test_source],
        cache_evidence,
    )
    _require(
        proxy_hairpin_reference["topology"][
            "hard_2plus1_success_count_all"
        ]
        == 3,
        "frozen proxy reference no longer has the recorded 3/6 strict topology count",
    )
    audited_runs: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in variants:
        for seed in seeds:
            key = (variant, seed)
            csv_run = csv_by_run[key]
            json_run = json_by_run[key]
            expected_parameter_count = parameter_counts[variant]
            _require(int(csv_run["parameter_count"]) == expected_parameter_count, f"CSV parameter count differs for {key}")
            _require(int(json_run["parameter_count"]) == expected_parameter_count, f"summary parameter count differs for {key}")
            _require(csv_run.get("checkpoint_policy") == "best_state_in_memory_only_no_model_file", f"CSV checkpoint policy differs for {key}")
            _require(json_run.get("checkpoint_policy") == "best_state_in_memory_only_no_model_file", f"summary checkpoint policy differs for {key}")
            prediction_path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
            _require(Path(str(csv_run["prediction_path"])).name == prediction_path.name, f"CSV prediction path basename differs for {key}")
            _require(Path(str(json_run["prediction_path"])).name == prediction_path.name, f"summary prediction path basename differs for {key}")
            prediction_metrics, test_arrays = _validate_prediction(
                prediction_path,
                cache_evidence["labels"],
                cache_evidence["split_codes"],
                class_names,
            )
            for split_name in ("validation", "test"):
                flat = _flatten_metrics(split_name, prediction_metrics[split_name], class_names)
                _compare_flat_metrics(f"CSV.{variant}.{seed}", csv_run, flat)
                _compare_flat_metrics(f"summary.{variant}.{seed}", json_run, flat)
            history = _validate_history(
                output_dir / "histories" / f"{variant}_seed{seed}.csv",
                csv_run,
                json_run,
                prediction_metrics["validation"],
                spec,
            )
            _assert_close(f"CSV/summary elapsed_seconds {key}", csv_run["elapsed_seconds"], json_run["elapsed_seconds"])
            _require(float(csv_run["elapsed_seconds"]) > 0.0, f"non-positive elapsed time for {key}")
            test_hairpin_instances = _hairpin_instance_metrics(
                test_arrays["source_indices"],
                test_arrays["targets"],
                test_arrays["predicted"],
                cache_evidence,
            )
            audited_runs[key] = {
                "variant": variant,
                "seed": seed,
                "parameter_count": expected_parameter_count,
                "history": history,
                "validation": prediction_metrics["validation"],
                "test": prediction_metrics["test"],
                "test_hairpin_instances": test_hairpin_instances,
                "prediction_sha256": _sha256(prediction_path),
                "history_sha256": _sha256(output_dir / "histories" / f"{variant}_seed{seed}.csv"),
            }

    recomputed_aggregate = _aggregate_recomputed(
        audited_runs, variants, seeds, class_names
    )
    _compare_reported_aggregate(
        summary.get("aggregate", {}), recomputed_aggregate, variants, class_names
    )

    excluded_paths = {
        Path(path).expanduser().resolve()
        for path in (write_json, write_manifest)
        if path is not None
    }
    manifest = _artifact_manifest(
        config_path, cache_path, output_dir, excluded_paths
    )
    payload = {
        "audit": AUDIT_VERSION,
        "status": "PASS",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "numeric_absolute_tolerance": ABSOLUTE_TOLERANCE,
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "output_dir": str(output_dir),
            "cache": str(cache_path),
            "cache_sha256": cache_evidence["sha256"],
        },
        "evidence": {
            "formal_run_count": 12,
            "variants": list(variants),
            "seeds": list(seeds),
            "parameter_counts": parameter_counts,
            "prediction_file_count": 12,
            "history_file_count": 12,
            "cache_row_count": cache_evidence["row_count"],
            "cache_fmt_feature_dim": cache_evidence["fmt_feature_dim"],
            "split_counts": cache_evidence["split_counts"],
            "split_nonoverlap_certificate": cache_evidence[
                "split_nonoverlap_certificate"
            ],
            "device": device,
            "persistent_model_file_count": 0,
            "metric_source": "independently recomputed from prediction NPZ files",
            "selection_check": (
                "best_epoch follows the frozen sequential score > best + "
                "min_delta rule in every history"
            ),
            "test_hairpin_vortex_ids": proxy_hairpin_reference["vortex_ids"],
            "proxy_hairpin_topology_reference": {
                "role": (
                    "descriptive ceiling/reference only; proxy itself reaches "
                    "3/6 and is not a supervised hard pass threshold"
                ),
                "topology": proxy_hairpin_reference["topology"],
                "topology_per_vortex": proxy_hairpin_reference[
                    "topology_per_vortex"
                ],
            },
        },
        "formal_metrics": recomputed_aggregate,
        "runs": [
            audited_runs[(variant, seed)]
            for variant in variants
            for seed in seeds
        ],
        "interpretation_boundary": summary.get("interpretation_boundary"),
        "artifact_sha256_manifest": manifest,
    }

    for requested_path, document in (
        (write_json, payload),
        (write_manifest, manifest),
    ):
        if requested_path is None:
            continue
        destination = Path(requested_path).expanduser().resolve()
        _require(destination.parent.is_dir(), f"output parent does not exist: {destination.parent}")
        _require(not destination.is_dir(), f"requested output is a directory: {destination}")
        destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen mainExp_Task4B_1.2 YAML")
    parser.add_argument("--output-dir", required=True, help="Completed formal result directory")
    parser.add_argument(
        "--write-json",
        help="Optional explicit path for the audit JSON; otherwise no audit file is written",
    )
    parser.add_argument(
        "--write-manifest",
        help="Optional explicit path for a standalone SHA-256 manifest JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    try:
        result = audit(
            arguments.config,
            arguments.output_dir,
            write_json=arguments.write_json,
            write_manifest=arguments.write_manifest,
        )
    except (AuditError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"AUDIT FAILED: {error}") from error
    if arguments.write_json:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "audit": result["audit"],
                    "write_json": str(Path(arguments.write_json).resolve()),
                    "write_manifest": (
                        str(Path(arguments.write_manifest).resolve())
                        if arguments.write_manifest
                        else None
                    ),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(result, indent=2))
