"""Diagnose Task4-b hairpin errors in the frozen validation/test predictions.

This is a read-only analysis of ``mainExp_Task4B_1.2``.  It verifies that all
12 prediction artifacts refer to the exact frozen cache rows and targets, then
separates two different questions:

1. How often does the four-class model leak a true hairpin voxel into either
   ordinary-vortex class?
2. If the true hairpin mask were already known, can the model distinguish
   head from limb using only its class-2 and class-3 scores?

The second quantity is explicitly a conditional two-class diagnostic.  It is
not, and must not be reported as, four-class held-out performance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DIAGNOSIS_VERSION = "Diagnose_Task4B_HoldoutErrors_1.1"
EXPERIMENT = "mainExp_Task4B_1.2"
VARIANTS = ("raw", "raw_wide", "fmt_only", "raw_fmt")
SEEDS = (7068, 7069, 7070)
SPLITS = {"validation": 1, "test": 2}
CLASS_NAMES = (
    "ordinary_streamwise",
    "ordinary_spanwise",
    "hairpin_head",
    "hairpin_limb",
)
HEAD_CLASS = 2
LIMB_CLASS = 3
PROBABILITY_TOLERANCE = 2e-6

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    REPO_ROOT
    / "outputs"
    / "Other_Task4B_FMTFourClassClustering_1.2"
    / "channel_GTs"
    / "cache"
    / "task4b_four_class_cache.npz"
)
DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT / "outputs" / "mainExp_Task4B_1.2" / "channel_GTs"
)
DEFAULT_JSON = DEFAULT_EXPERIMENT_DIR / "holdout_error_diagnosis.json"

CORE_METRICS = (
    "hairpin_to_ordinary_rate",
    "hairpin_head_to_ordinary_rate",
    "hairpin_limb_to_ordinary_rate",
    "known_hairpin_conditional_head_limb_macro_f1",
    "known_hairpin_conditional_head_limb_accuracy",
    "known_hairpin_conditional_macro_f1_minus_majority_limb_baseline",
    "vortex_id_equal_mean_hairpin_to_ordinary_rate",
    "vortex_id_equal_mean_hairpin_head_to_ordinary_rate",
    "vortex_id_equal_mean_hairpin_limb_to_ordinary_rate",
    "vortex_id_equal_mean_known_hairpin_conditional_head_limb_macro_f1",
    "vortex_id_equal_mean_known_hairpin_conditional_head_limb_accuracy",
)


class DiagnosisError(RuntimeError):
    """Raised when a frozen prediction artifact violates the analysis contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    _require(float(denominator) > 0.0, "metric denominator must be positive")
    return float(numerator) / float(denominator)


def _two_class_scores(
    targets: np.ndarray,
    predicted: np.ndarray,
    *,
    labels: tuple[int, int] = (HEAD_CLASS, LIMB_CLASS),
) -> dict[str, Any]:
    """Return deterministic two-class metrics without sklearn dependencies."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    _require(len(targets) == len(predicted) and len(targets) > 0, "invalid targets")
    _require(set(np.unique(targets)).issubset(labels), "targets leave head/limb classes")
    _require(set(np.unique(predicted)).issubset(labels), "predictions leave head/limb classes")

    confusion = np.zeros((2, 2), dtype=np.int64)
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            confusion[true_index, predicted_index] = int(
                np.count_nonzero((targets == true_label) & (predicted == predicted_label))
            )

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for true_index, true_label in enumerate(labels):
        true_positive = int(confusion[true_index, true_index])
        false_negative = int(confusion[true_index].sum() - true_positive)
        false_positive = int(confusion[:, true_index].sum() - true_positive)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            float(true_positive / precision_denominator)
            if precision_denominator
            else 0.0
        )
        recall = float(true_positive / recall_denominator) if recall_denominator else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        class_name = CLASS_NAMES[true_label]
        per_class[class_name] = {
            "support": int(recall_denominator),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    return {
        "sample_count": int(len(targets)),
        "accuracy": float(np.mean(targets == predicted)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix_true_rows_predicted_columns_labels_2_3": confusion.tolist(),
    }


def majority_limb_baseline(targets: np.ndarray) -> dict[str, Any]:
    """Evaluate the all-limb predictor on a known-hairpin target set."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    _require(len(targets) > 0, "majority-limb baseline received no samples")
    _require(
        set(np.unique(targets)).issubset({HEAD_CLASS, LIMB_CLASS}),
        "majority-limb baseline targets must be head/limb only",
    )
    return _two_class_scores(targets, np.full_like(targets, LIMB_CLASS))


def _base_hairpin_diagnostics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute voxel-weighted leakage and the conditional head/limb diagnostic."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    _require(probabilities.shape == (len(targets), 4), "invalid probability shape")
    hairpin_mask = np.isin(targets, (HEAD_CLASS, LIMB_CLASS))
    _require(np.any(hairpin_mask), "split contains no hairpin targets")

    hairpin_targets = targets[hairpin_mask]
    hairpin_probabilities = probabilities[hairpin_mask]
    four_class_prediction = np.argmax(hairpin_probabilities, axis=1)
    predicted_ordinary = four_class_prediction < HEAD_CLASS
    head_mask = hairpin_targets == HEAD_CLASS
    limb_mask = hairpin_targets == LIMB_CLASS

    # Counterfactual known-mask diagnostic: discard ordinary-class scores and
    # force a choice between the head and limb scores on every true hairpin row.
    conditional_prediction = (
        np.argmax(hairpin_probabilities[:, HEAD_CLASS : LIMB_CLASS + 1], axis=1)
        + HEAD_CLASS
    )
    conditional = _two_class_scores(hairpin_targets, conditional_prediction)
    baseline = majority_limb_baseline(hairpin_targets)

    return {
        "hairpin_support": int(len(hairpin_targets)),
        "hairpin_head_support": int(np.count_nonzero(head_mask)),
        "hairpin_limb_support": int(np.count_nonzero(limb_mask)),
        "hairpin_to_ordinary_count": int(np.count_nonzero(predicted_ordinary)),
        "hairpin_to_ordinary_rate": _safe_ratio(
            np.count_nonzero(predicted_ordinary), len(hairpin_targets)
        ),
        "hairpin_head_to_ordinary_count": int(
            np.count_nonzero(predicted_ordinary & head_mask)
        ),
        "hairpin_head_to_ordinary_rate": _safe_ratio(
            np.count_nonzero(predicted_ordinary & head_mask), np.count_nonzero(head_mask)
        ),
        "hairpin_limb_to_ordinary_count": int(
            np.count_nonzero(predicted_ordinary & limb_mask)
        ),
        "hairpin_limb_to_ordinary_rate": _safe_ratio(
            np.count_nonzero(predicted_ordinary & limb_mask), np.count_nonzero(limb_mask)
        ),
        "known_hairpin_conditional_head_limb_macro_f1": conditional["macro_f1"],
        "known_hairpin_conditional_head_limb_accuracy": conditional["accuracy"],
        "known_hairpin_conditional_head_limb": conditional,
        "majority_limb_baseline_macro_f1": baseline["macro_f1"],
        "majority_limb_baseline_accuracy": baseline["accuracy"],
        "known_hairpin_conditional_macro_f1_minus_majority_limb_baseline": float(
            conditional["macro_f1"] - baseline["macro_f1"]
        ),
    }


def hairpin_diagnostics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    vortex_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute voxel- and VortexId-equal diagnostics for one held-out split."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    vortex_ids = np.asarray(vortex_ids, dtype=np.int64).reshape(-1)
    _require(len(targets) == len(vortex_ids), "VortexId/target row count differs")
    result = _base_hairpin_diagnostics(targets, probabilities)

    hairpin_ids = sorted(int(value) for value in np.unique(vortex_ids[targets >= 2]))
    _require(bool(hairpin_ids) and all(value > 0 for value in hairpin_ids), "invalid hairpin IDs")
    per_id: dict[str, dict[str, Any]] = {}
    id_metric_names = (
        "hairpin_to_ordinary_rate",
        "hairpin_head_to_ordinary_rate",
        "hairpin_limb_to_ordinary_rate",
        "known_hairpin_conditional_head_limb_macro_f1",
        "known_hairpin_conditional_head_limb_accuracy",
    )
    for vortex_id in hairpin_ids:
        mask = vortex_ids == vortex_id
        _require(
            set(np.unique(targets[mask])) == {HEAD_CLASS, LIMB_CLASS},
            f"VortexId {vortex_id} does not contain both head and limb labels",
        )
        per_id[str(vortex_id)] = _base_hairpin_diagnostics(
            targets[mask], probabilities[mask]
        )

    equal_mean = {
        name: float(np.mean([per_id[str(vortex_id)][name] for vortex_id in hairpin_ids]))
        for name in id_metric_names
    }
    result["vortex_id_count"] = len(hairpin_ids)
    result["per_vortex_id"] = per_id
    result["vortex_id_equal_mean"] = equal_mean
    for name, value in equal_mean.items():
        result[f"vortex_id_equal_mean_{name}"] = value
    return result


def _mean_sample_sd(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    _require(array.ndim == 1 and len(array) >= 2, "sample SD requires at least two values")
    _require(np.isfinite(array).all(), "aggregate values contain non-finite entries")
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)),
        "n": int(len(array)),
    }


def _load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    _require(cache_path.is_file(), f"cache not found: {cache_path}")
    required = {"labels", "split_codes", "vortex_ids"}
    with np.load(cache_path, allow_pickle=False) as cache:
        _require(required.issubset(cache.files), f"cache lacks {sorted(required - set(cache.files))}")
        arrays = {
            "labels": np.asarray(cache["labels"], dtype=np.int64).reshape(-1),
            "split_codes": np.asarray(cache["split_codes"], dtype=np.int64).reshape(-1),
            "vortex_ids": np.asarray(cache["vortex_ids"], dtype=np.int64).reshape(-1),
        }
    row_count = len(arrays["labels"])
    _require(row_count > 0, "cache is empty")
    _require(all(len(value) == row_count for value in arrays.values()), "cache arrays differ")
    _require(set(np.unique(arrays["labels"])) == {0, 1, 2, 3}, "labels are not 0..3")
    _require(set(np.unique(arrays["split_codes"])) == {0, 1, 2}, "splits are not 0,1,2")
    _require(
        np.array_equal(arrays["vortex_ids"] > 0, arrays["labels"] >= 2),
        "positive VortexId rows do not equal the hairpin-label rows",
    )
    return arrays


def _validate_summary_manifest(
    summary_path: Path, expected_prediction_names: set[str]
) -> dict[str, Any]:
    """Verify that the formal summary maps every variant/seed to one artifact."""

    _require(summary_path.is_file(), f"formal summary not found: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisError(f"cannot parse formal summary: {summary_path}") from exc
    _require(summary.get("experiment") == EXPERIMENT, "summary experiment name differs")
    taxonomy = summary.get("taxonomy", {}).get("classes", {})
    observed_classes = tuple(
        str(taxonomy.get(str(index), taxonomy.get(index, ""))) for index in range(4)
    )
    _require(observed_classes == CLASS_NAMES, f"summary taxonomy differs: {observed_classes}")
    runs = summary.get("runs", [])
    _require(isinstance(runs, list) and len(runs) == 12, "summary must contain 12 runs")
    observed_pairs: set[tuple[str, int]] = set()
    observed_names: set[str] = set()
    for run in runs:
        _require(isinstance(run, dict), "summary run is not an object")
        pair = (str(run.get("variant", "")), int(run.get("seed", -1)))
        _require(pair not in observed_pairs, f"duplicate summary run: {pair}")
        observed_pairs.add(pair)
        prediction_name = Path(str(run.get("prediction_path", ""))).name
        expected_name = f"{pair[0]}_seed{pair[1]}.npz"
        _require(
            prediction_name == expected_name,
            f"summary run {pair} points to {prediction_name!r}, expected {expected_name!r}",
        )
        observed_names.add(prediction_name)
    expected_pairs = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    _require(observed_pairs == expected_pairs, "summary variant/seed set differs")
    _require(observed_names == expected_prediction_names, "summary prediction set differs")
    return {
        "summary_sha256": _sha256(summary_path),
        "summary_run_count": len(runs),
        "variant_seed_to_prediction_filename_mapping_verified": True,
    }


def _validate_prediction(
    prediction_path: Path,
    cache: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    required = {
        "validation_source_indices",
        "validation_targets",
        "validation_probabilities",
        "test_source_indices",
        "test_targets",
        "test_probabilities",
    }
    with np.load(prediction_path, allow_pickle=False) as artifact:
        _require(
            set(artifact.files) == required,
            f"{prediction_path.name}: unexpected/missing arrays {sorted(set(artifact.files) ^ required)}",
        )
        loaded = {name: np.asarray(artifact[name]) for name in required}

    diagnostics: dict[str, dict[str, Any]] = {}
    mapping: dict[str, Any] = {}
    for split_name, split_code in SPLITS.items():
        expected_indices = np.flatnonzero(cache["split_codes"] == split_code).astype(np.int64)
        source_indices = np.asarray(
            loaded[f"{split_name}_source_indices"], dtype=np.int64
        ).reshape(-1)
        targets = np.asarray(loaded[f"{split_name}_targets"], dtype=np.int64).reshape(-1)
        probabilities = np.asarray(
            loaded[f"{split_name}_probabilities"], dtype=np.float64
        )
        _require(
            np.array_equal(source_indices, expected_indices),
            f"{prediction_path.name}: {split_name} source-index order/membership differs from cache",
        )
        _require(
            np.array_equal(targets, cache["labels"][source_indices]),
            f"{prediction_path.name}: {split_name} targets do not map to cache labels",
        )
        _require(
            probabilities.shape == (len(source_indices), 4),
            f"{prediction_path.name}: {split_name} probabilities have shape {probabilities.shape}",
        )
        _require(np.isfinite(probabilities).all(), f"{prediction_path.name}: non-finite probabilities")
        _require(
            np.all(probabilities >= -PROBABILITY_TOLERANCE)
            and np.all(probabilities <= 1.0 + PROBABILITY_TOLERANCE),
            f"{prediction_path.name}: probabilities leave [0,1]",
        )
        _require(
            np.allclose(probabilities.sum(axis=1), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0.0),
            f"{prediction_path.name}: probability rows do not sum to one",
        )
        diagnostics[split_name] = hairpin_diagnostics(
            targets, probabilities, cache["vortex_ids"][source_indices]
        )
        mapping[split_name] = {
            "source_indices_exact_cache_split_order": True,
            "targets_exact_cache_label_mapping": True,
            "sample_count": int(len(source_indices)),
            "source_index_sha256": hashlib.sha256(
                np.ascontiguousarray(source_indices).view(np.uint8)
            ).hexdigest(),
        }
    return diagnostics, mapping


def _baseline_bundle(cache: dict[str, np.ndarray]) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for split_name, split_code in SPLITS.items():
        targets = cache["labels"][cache["split_codes"] == split_code]
        hairpin_targets = targets[targets >= 2]
        baselines[split_name] = majority_limb_baseline(hairpin_targets)
    baselines["test_minus_validation"] = {
        metric: float(baselines["test"][metric] - baselines["validation"][metric])
        for metric in ("macro_f1", "accuracy")
    }
    return baselines


def diagnose(cache_path: Path, experiment_dir: Path) -> dict[str, Any]:
    """Validate all 12 artifacts and return the complete diagnosis payload."""

    cache_path = cache_path.resolve()
    experiment_dir = experiment_dir.resolve()
    predictions_dir = experiment_dir / "predictions"
    _require(predictions_dir.is_dir(), f"predictions directory not found: {predictions_dir}")
    cache = _load_cache(cache_path)

    expected_names = {
        f"{variant}_seed{seed}.npz" for variant in VARIANTS for seed in SEEDS
    }
    actual_names = {path.name for path in predictions_dir.glob("*.npz")}
    _require(
        actual_names == expected_names,
        "prediction set differs; missing="
        f"{sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}",
    )
    summary_validation = _validate_summary_manifest(
        experiment_dir / "summary.json", expected_names
    )

    runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            prediction_path = predictions_dir / f"{variant}_seed{seed}.npz"
            split_metrics, mapping = _validate_prediction(prediction_path, cache)
            paired_difference = {
                metric: float(split_metrics["test"][metric] - split_metrics["validation"][metric])
                for metric in CORE_METRICS
            }
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "prediction_file": prediction_path.name,
                    "prediction_sha256": _sha256(prediction_path),
                    "mapping_validation": mapping,
                    "validation": split_metrics["validation"],
                    "test": split_metrics["test"],
                    "test_minus_validation": paired_difference,
                }
            )

    aggregates: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_runs = [run for run in runs if run["variant"] == variant]
        _require(len(variant_runs) == len(SEEDS), f"{variant}: not exactly three seeds")
        aggregate = {"validation": {}, "test": {}, "test_minus_validation": {}}
        for split_name in ("validation", "test"):
            for metric in CORE_METRICS:
                aggregate[split_name][metric] = _mean_sample_sd(
                    run[split_name][metric] for run in variant_runs
                )
        for metric in CORE_METRICS:
            aggregate["test_minus_validation"][metric] = _mean_sample_sd(
                run["test_minus_validation"][metric] for run in variant_runs
            )
        aggregates[variant] = aggregate

    split_support: dict[str, Any] = {}
    for split_name, split_code in SPLITS.items():
        mask = cache["split_codes"] == split_code
        targets = cache["labels"][mask]
        split_support[split_name] = {
            "all_four_class_rows": int(np.count_nonzero(mask)),
            "hairpin_rows_used_by_diagnostics": int(np.count_nonzero(targets >= 2)),
            "hairpin_head": int(np.count_nonzero(targets == HEAD_CLASS)),
            "hairpin_limb": int(np.count_nonzero(targets == LIMB_CLASS)),
            "vortex_ids": sorted(int(value) for value in np.unique(cache["vortex_ids"][mask]) if value > 0),
        }

    return {
        "diagnosis": DIAGNOSIS_VERSION,
        "source_experiment": EXPERIMENT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "experiment_directory": str(experiment_dir),
        "taxonomy": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "contract": {
            "prediction_artifact_count": 12,
            "all_prediction_source_and_target_mappings_verified": True,
            "ordinary_leak_definition": (
                "Among true class-2/class-3 hairpin rows, the four-class argmax is class 0 or 1."
            ),
            "known_hairpin_conditional_definition": (
                "On true hairpin rows only, choose argmax between probability columns 2 and 3; "
                "ordinary-class scores are deliberately excluded."
            ),
            "claim_boundary": (
                "Known-hairpin conditional head/limb metrics are counterfactual two-class "
                "diagnostics and are not four-class validation/test metrics."
            ),
            "vortex_id_mean_definition": (
                "Compute each rate/score inside each positive VortexId, then average IDs equally."
            ),
            "dispersion_definition": "sample standard deviation across the three training seeds (ddof=1)",
            "test_minus_validation_definition": "paired seed-wise test value minus validation value",
        },
        "formal_summary_validation": summary_validation,
        "split_support": split_support,
        "majority_limb_known_hairpin_baseline": _baseline_bundle(cache),
        "runs": runs,
        "aggregates_mean_and_sample_sd": aggregates,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--write-json",
        type=Path,
        default=DEFAULT_JSON,
        help="Output JSON path (default: frozen experiment directory/holdout_error_diagnosis.json).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = diagnose(args.cache, args.experiment_dir)
    output_path = args.write_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "diagnosis": DIAGNOSIS_VERSION,
                "status": "PASS",
                "prediction_artifacts_verified": len(payload["runs"]),
                "json": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
