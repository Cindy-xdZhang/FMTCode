"""Independent audit for ``mainExp_Task4B_ChannelToTBL_2.3``.

This file deliberately does not import either Task4-b training driver.  It
reconstructs the channel-only normalization from the frozen source cache,
maps every saved TBL prediction back to the ordered target-cache chunks, and
recomputes the four-class and diagnostic metrics from saved probabilities.

An incomplete 4-variant x 3-seed experiment is a failed audit, not a partial
success.  The command-line entry point always writes a JSON report and exits
non-zero when the audit status is ``FAIL``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

# Permit both ``python -m experiments...`` and direct script execution while
# still importing only the low-level frozen encoder, never a training driver.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d


AUDIT_VERSION = "Audit_Task4B_ChannelToTBL_2.3"
EXPERIMENT = "mainExp_Task4B_ChannelToTBL_2.3"
DEFAULT_CONFIG = Path("config/mainExp_Task4B_ChannelToTBL_2.3.yaml")
VARIANTS = ("raw", "raw_wide", "fmt_only", "raw_fmt")
SEEDS = (7068, 7069, 7070)
CLASS_NAMES = (
    "ordinary_streamwise",
    "ordinary_spanwise",
    "hairpin_head",
    "hairpin_limb",
)
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
NUMERIC_TOLERANCE = 5.0e-10
SOURCE_METRIC_TOLERANCE = 2.0e-6
PROBABILITY_TOLERANCE = 2.0e-6


class AuditError(RuntimeError):
    """Raised when an artifact violates the frozen experiment contract."""


class IncompleteAuditError(AuditError):
    """Raised when one or more of the twelve formal runs are absent."""

    def __init__(self, message: str, inventory: dict[str, Any]):
        super().__init__(message)
        self.inventory = inventory


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _resolve_file(
    value: str | Path,
    *,
    repo_root: Path,
) -> Path:
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((repo_root / path, Path.cwd() / path))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    raise AuditError(
        "required file not found; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _assert_close(
    label: str,
    actual: Any,
    expected: Any,
    *,
    atol: float = NUMERIC_TOLERANCE,
) -> None:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric") from exc
    _require(
        np.isfinite(actual_value) and np.isfinite(expected_value),
        f"{label} is non-finite",
    )
    difference = abs(actual_value - expected_value)
    _require(
        difference <= float(atol),
        f"{label} differs by {difference:.12g}: "
        f"reported={actual_value:.17g}, recomputed={expected_value:.17g}",
    )


def _compare_structure(
    label: str,
    actual: Any,
    expected: Any,
    *,
    atol: float = NUMERIC_TOLERANCE,
) -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{label} is not a dictionary")
        _require(
            set(actual) == set(expected),
            f"{label} keys differ: missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}",
        )
        for key in expected:
            _compare_structure(
                f"{label}.{key}", actual[key], expected[key], atol=atol
            )
        return
    if isinstance(expected, (list, tuple)):
        _require(isinstance(actual, (list, tuple)), f"{label} is not a sequence")
        _require(len(actual) == len(expected), f"{label} length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_structure(f"{label}[{index}]", left, right, atol=atol)
        return
    if isinstance(expected, (bool, np.bool_)):
        _require(bool(actual) is bool(expected), f"{label} differs")
        return
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        _require(
            isinstance(actual, (int, np.integer)) and int(actual) == int(expected),
            f"{label} differs: reported={actual!r}, recomputed={expected!r}",
        )
        return
    if isinstance(expected, (float, np.floating)):
        _assert_close(label, actual, expected, atol=atol)
        return
    _require(actual == expected, f"{label} differs")


def _validate_config(spec: dict[str, Any]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    _require(spec.get("experiment") == EXPERIMENT, "unexpected experiment name")
    _require(
        spec.get("supersedes") == "mainExp_Task4B_ChannelToTBL_2.2",
        "formal 2.3 config does not supersede the rejected 2.2 preprocessing run",
    )
    _require(
        spec.get("task") == "Task4-b channel-to-TBL supervised four-class transfer",
        "unexpected task description",
    )
    variants = tuple(str(value) for value in spec.get("variants", ()))
    seeds = tuple(int(value) for value in spec.get("training", {}).get("seeds", ()))
    _require(variants == VARIANTS, f"variants must be {VARIANTS}, found {variants}")
    _require(seeds == SEEDS, f"seeds must be {SEEDS}, found {seeds}")

    source = spec.get("source", {})
    target = spec.get("target", {})
    training = spec.get("training", {})
    evaluation = spec.get("evaluation", {})
    _require(source.get("role") == "training_only", "channel must be training-only")
    _require(
        source.get("population")
        == "all_20641_rows_including_historical_channel_splits",
        "source population is not the complete channel cache",
    )
    _require(target.get("role") == "test_only", "TBL must be test-only")
    _require(
        target.get("vorticity_validation_role")
        == "numerical_curl_audit_only_never_labels_or_features",
        "stored TBL vorticity has an invalid role",
    )
    _require(
        training.get("sampling")
        == "one_without_replacement_permutation_of_every_source_row_per_epoch",
        "source training sampler is not a complete without-replacement pass",
    )
    _require(training.get("loss") == "unweighted_cross_entropy", "loss changed")
    _require(float(training.get("weight_decay", -1.0)) == 0.0, "weight decay changed")
    _require(float(spec.get("model", {}).get("dropout", -1.0)) == 0.0, "dropout changed")
    stop_gate = training.get("stop_gate", {})
    _require(
        int(stop_gate.get("consecutive_source_zero_error_epochs", 0)) == 3,
        "source memorization streak is not three epochs",
    )
    _require(
        stop_gate.get("positive_minimum_true_logit_margin") is True,
        "source gate does not require positive true-class margin",
    )
    _require(
        evaluation.get("target_policy")
        == "evaluate_once_only_after_source_memorization_gate",
        "target evaluation policy changed",
    )
    _require(
        evaluation.get("target_selection_prohibited") is True,
        "target-based selection is not prohibited",
    )
    _require(
        spec.get("encoder", {}).get("frozen") is True,
        "FMT encoder is not frozen",
    )
    _require(
        tuple(spec.get("core_comparisons", ()))
        == ("raw_fmt_minus_raw", "raw_fmt_minus_raw_wide"),
        "core comparison set changed",
    )
    _require(
        list(source.get("required_class_support", ())) == [5000, 5000, 2829, 7812],
        "source class support contract changed",
    )
    return variants, seeds


def _artifact_inventory(
    output_dir: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    expected_stems = {f"{variant}_seed{seed}" for variant in variants for seed in seeds}
    specifications = {
        "runs": (output_dir / "runs", ".json"),
        "predictions": (output_dir / "predictions", ".npz"),
        "histories": (output_dir / "histories", ".csv"),
    }
    inventory: dict[str, Any] = {
        "expected_run_count": len(expected_stems),
        "expected_run_keys": sorted(expected_stems),
    }
    complete = True
    for description, (directory, suffix) in specifications.items():
        actual = {
            path.stem
            for path in directory.glob(f"*{suffix}")
            if path.is_file()
        } if directory.is_dir() else set()
        missing = sorted(expected_stems - actual)
        unexpected = sorted(actual - expected_stems)
        inventory[description] = {
            "count": len(actual),
            "missing": missing,
            "unexpected": unexpected,
        }
        complete = complete and not missing and not unexpected
    inventory["complete"] = bool(complete)
    return inventory


def _load_source_cache(
    spec: dict[str, Any], repo_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, str]:
    cache_path = _resolve_file(spec["source"]["cache"], repo_root=repo_root)
    actual_hash = _sha256(cache_path)
    _require(
        actual_hash == str(spec["source"]["cache_sha256"]),
        "channel source cache SHA-256 differs from config",
    )
    required = {
        "raw_features",
        "fmt_features",
        "labels",
        "split_codes",
        "vortex_ids",
        "metadata_json",
    }
    with np.load(cache_path, allow_pickle=False) as cache:
        _require(required.issubset(cache.files), "channel cache lacks required arrays")
        arrays = {
            "raw_features": np.asarray(cache["raw_features"], dtype=np.float32),
            "fmt_features": np.asarray(cache["fmt_features"], dtype=np.float32),
            "labels": np.asarray(cache["labels"], dtype=np.int64).reshape(-1),
            "split_codes": np.asarray(cache["split_codes"], dtype=np.int8).reshape(-1),
            "vortex_ids": np.asarray(cache["vortex_ids"], dtype=np.int32).reshape(-1),
        }
        metadata = json.loads(str(cache["metadata_json"]))
    count = len(arrays["labels"])
    _require(count == 20641, f"channel row count is {count}, expected 20641")
    _require(
        all(len(value) == count for value in arrays.values()),
        "channel cache row counts differ",
    )
    _require(
        arrays["raw_features"].ndim == 2
        and arrays["fmt_features"].ndim == 2,
        "channel feature arrays are not matrices",
    )
    _require(
        arrays["fmt_features"].shape[1] == int(spec["encoder"]["expected_feature_dim"]),
        "channel cached FMT width changed",
    )
    _require(
        all(np.isfinite(value).all() for value in arrays.values()),
        "channel cache contains non-finite values",
    )
    _require(
        np.bincount(arrays["labels"], minlength=4).tolist()
        == [int(value) for value in spec["source"]["required_class_support"]],
        "channel class support differs from config",
    )
    _require(
        set(np.unique(arrays["split_codes"])) == {0, 1, 2},
        "historical channel split codes changed",
    )
    _require(
        np.array_equal(arrays["vortex_ids"] > 0, arrays["labels"] >= 2),
        "channel VortexId support disagrees with hairpin classes",
    )
    return arrays, metadata, cache_path, actual_hash


def _fmt_base_width(encoder: dict[str, Any]) -> int:
    num_freq = int(encoder["num_freq"])
    mode_width = 1 if str(encoder["mode"]) == "magnitude" else 3
    return num_freq * mode_width + (
        num_freq - 1 if bool(encoder["include_chirality"]) else 0
    )


def _validate_channel_only_normalization(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    _require(path.is_file(), f"missing normalization artifact: {path}")
    sampled_steps = int(metadata.get("streamlines", {}).get("sampled_steps", 0))
    spatial_step = float(metadata.get("streamlines", {}).get("spatial_step", 0.0))
    source_offset = float(metadata.get("streamlines", {}).get("offset", 0.0))
    _require(sampled_steps > 0 and spatial_step > 0.0, "source metadata lacks streamline scale")
    source_offset_over_step = source_offset / spatial_step
    _assert_close(
        "channel source offset/integration-step ratio",
        source_offset_over_step,
        spec["streamlines"]["source_offset_over_spatial_step"],
        atol=1.0e-12,
    )
    primitive = arrays["raw_features"].reshape(-1, 7, sampled_steps, 3)
    dimensionless = (
        primitive - primitive[:, :1, :1, :]
    ) / np.float32(spatial_step)
    dimensionless = dimensionless.astype(np.float32, copy=False)
    encoder = spec["encoder"]
    recomputed_fmt = pathline_dft_features_3d(
        torch.from_numpy(dimensionless),
        num_freq=int(encoder["num_freq"]),
        neighbor_weight=1.0,
        neighbor_scale=float(encoder["neighbor_scale"]),
        neighbor_pool=str(encoder["neighbor_pool"]),
        mode=str(encoder["mode"]),
        include_chirality=bool(encoder["include_chirality"]),
    ).astype(np.float32, copy=False)
    _require(
        recomputed_fmt.shape
        == (len(dimensionless), int(encoder["expected_feature_dim"])),
        "recomputed channel FMT shape differs",
    )
    expected = {
        "raw_mean": dimensionless.mean(
            axis=(0, 1, 2), keepdims=True, dtype=np.float64
        ).astype(np.float32),
        "raw_std": np.maximum(
            dimensionless.std(
                axis=(0, 1, 2), keepdims=True, dtype=np.float64
            ).astype(np.float32),
            1.0e-6,
        ),
        "fmt_mean": recomputed_fmt.mean(
            axis=0, keepdims=True, dtype=np.float64
        ).astype(np.float32),
        "fmt_std": np.maximum(
            recomputed_fmt.std(
                axis=0, keepdims=True, dtype=np.float64
            ).astype(np.float32),
            1.0e-6,
        ),
    }
    expected_keys = {
        *expected,
        "sampled_steps",
        "fmt_base_width",
        "neighbor_weight",
        "fitted_row_count",
        "fitted_domain",
    }
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == expected_keys, "normalization artifact keys differ")
        for key, wanted in expected.items():
            actual = np.asarray(saved[key])
            _require(actual.dtype == wanted.dtype, f"normalization {key} dtype differs")
            _require(actual.shape == wanted.shape, f"normalization {key} shape differs")
            _require(
                np.array_equal(actual, wanted),
                f"normalization {key} is not the channel-only recomputation",
            )
        saved_steps = int(np.asarray(saved["sampled_steps"]).item())
        saved_base_width = int(np.asarray(saved["fmt_base_width"]).item())
        saved_weight = float(np.asarray(saved["neighbor_weight"]).item())
        saved_count = int(np.asarray(saved["fitted_row_count"]).item())
        saved_domain = str(np.asarray(saved["fitted_domain"]).item())
    _require(saved_steps == sampled_steps, "normalization sampled_steps differs")
    _require(saved_base_width == _fmt_base_width(encoder), "FMT base width differs")
    _assert_close(
        "normalization neighbor weight",
        saved_weight,
        encoder["neighbor_weight_after_source_standardization"],
    )
    _require(saved_count == len(arrays["labels"]), "normalization row count differs")
    _require(saved_domain == "channel_only", "normalization was not marked channel_only")
    return {
        "artifact_sha256": _sha256(path),
        "fit_domain": saved_domain,
        "fit_row_count": saved_count,
        "sampled_steps": saved_steps,
        "source_spatial_step": spatial_step,
        "source_offset": source_offset,
        "source_offset_over_spatial_step": source_offset_over_step,
        "fmt_base_width": saved_base_width,
        "raw_statistics_bitwise_equal": True,
        "fmt_statistics_bitwise_equal": True,
        "target_rows_used_for_fit": 0,
    }


def _validate_curl_gate(manifest: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    audit = manifest.get("curl_validation_audit", {})
    gate = spec["target"]["curl_validation_gate"]
    correlations = np.asarray(
        audit.get("vorticity_component_correlation_xyz", ()), dtype=np.float64
    )
    sign_agreements = np.asarray(
        audit.get("vorticity_component_sign_agreement_xyz", ()), dtype=np.float64
    )
    _require(correlations.shape == (3,), "curl audit lacks three component correlations")
    _require(sign_agreements.shape == (3,), "curl audit lacks three sign agreements")
    _require(
        np.isfinite(correlations).all() and np.isfinite(sign_agreements).all(),
        "curl audit contains non-finite component values",
    )
    relative_rmse = float(audit.get("vorticity_vector_relative_rmse", np.inf))
    coordinate_error = float(audit.get("point_coordinate_max_abs_error", np.inf))
    stored_oyf_error = float(audit.get("stored_oyf_max_abs_error", np.inf))
    _require(
        float(np.min(correlations)) >= float(gate["minimum_component_correlation"]),
        "TBL curl component-correlation gate failed",
    )
    _require(
        float(np.min(sign_agreements))
        >= float(gate["minimum_component_sign_agreement"]),
        "TBL curl sign-agreement gate failed",
    )
    _require(
        relative_rmse <= float(gate["maximum_vector_relative_rmse"]),
        "TBL curl relative-root-mean-square-error gate failed",
    )
    _require(coordinate_error <= 1.0e-5, "TBL PointIds coordinate gate failed")
    _require(stored_oyf_error <= 1.0e-7, "stored oyf identity gate failed")
    derived = manifest.get("flow_metadata", {}).get(
        "derived_omega_y_prime_vs_stored_oyf", {}
    )
    derived_correlation = float(derived.get("pearson_correlation", -np.inf))
    _require(
        derived_correlation
        >= float(spec["target"]["minimum_derived_oyf_correlation"]),
        "full-volume derived/stored oyf correlation gate failed",
    )
    _require(
        manifest.get("flow_metadata", {}).get("vorticity_source")
        == "second_order_finite_difference_curl_of_velocity",
        "TBL vorticity source changed",
    )
    return {
        "passed": True,
        "component_correlation_xyz": correlations.tolist(),
        "minimum_component_correlation": float(np.min(correlations)),
        "component_sign_agreement_xyz": sign_agreements.tolist(),
        "minimum_component_sign_agreement": float(np.min(sign_agreements)),
        "vector_relative_rmse": relative_rmse,
        "point_coordinate_max_abs_error": coordinate_error,
        "stored_oyf_max_abs_error": stored_oyf_error,
        "derived_oyf_correlation": derived_correlation,
    }


def _load_target_chunks(
    manifest_path: Path,
    manifest: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    entries = manifest.get("chunks", ())
    _require(isinstance(entries, list) and entries, "target manifest has no chunks")
    _require(
        int(manifest.get("chunk_count", -1)) == len(entries),
        "target manifest chunk count differs",
    )
    pieces: dict[str, list[np.ndarray]] = {
        "source_candidate_indices": [],
        "voxel_indices_xyz": [],
        "seeds_xyz": [],
        "vortex_ids": [],
        "targets": [],
    }
    referenced_names: set[str] = set()
    sampled_steps = int(manifest["streamlines"]["sampled_steps"])
    raw_width = 7 * sampled_steps * 3
    fmt_width = int(spec["encoder"]["expected_feature_dim"])
    required = {
        "raw_features",
        "fmt_features",
        "labels",
        "seeds_xyz",
        "vortex_ids",
        "voxel_indices_xyz",
        "source_candidate_indices",
    }
    total = 0
    for ordinal, entry in enumerate(entries):
        _require(
            set(entry)
            == {
                "path_relative_to_manifest",
                "sha256",
                "row_count",
                "source_candidate_index_min",
                "source_candidate_index_max",
            },
            f"target chunk entry {ordinal} keys differ",
        )
        relative = Path(str(entry["path_relative_to_manifest"]))
        _require(not relative.is_absolute(), "target chunk path must be relative")
        path = (manifest_path.parent / relative).resolve()
        manifest_directory = manifest_path.parent.resolve()
        _require(
            path == manifest_directory or manifest_directory in path.parents,
            "target chunk relative path escapes the manifest directory",
        )
        _require(path.is_file(), f"target chunk is missing: {relative}")
        _require(path.name not in referenced_names, "target manifest repeats a chunk")
        referenced_names.add(path.name)
        _require(_sha256(path) == entry["sha256"], f"target chunk hash differs: {path.name}")
        with np.load(path, allow_pickle=False) as chunk:
            _require(set(chunk.files) == required, f"{path.name} array keys differ")
            raw = np.asarray(chunk["raw_features"])
            fmt = np.asarray(chunk["fmt_features"])
            labels = np.asarray(chunk["labels"])
            seeds = np.asarray(chunk["seeds_xyz"])
            vortex_ids = np.asarray(chunk["vortex_ids"])
            voxels = np.asarray(chunk["voxel_indices_xyz"])
            candidate = np.asarray(chunk["source_candidate_indices"])
        count = int(entry["row_count"])
        _require(count > 0, f"{path.name} is empty")
        _require(raw.shape == (count, raw_width), f"{path.name} Raw shape differs")
        _require(fmt.shape == (count, fmt_width), f"{path.name} FMT shape differs")
        _require(labels.shape == (count,), f"{path.name} label shape differs")
        _require(seeds.shape == (count, 3), f"{path.name} seed shape differs")
        _require(vortex_ids.shape == (count,), f"{path.name} VortexId shape differs")
        _require(voxels.shape == (count, 3), f"{path.name} voxel shape differs")
        _require(candidate.shape == (count,), f"{path.name} candidate-index shape differs")
        expected_dtypes = {
            "raw": (raw.dtype, np.dtype(np.float32)),
            "fmt": (fmt.dtype, np.dtype(np.float32)),
            "labels": (labels.dtype, np.dtype(np.int8)),
            "seeds": (seeds.dtype, np.dtype(np.float32)),
            "vortex_ids": (vortex_ids.dtype, np.dtype(np.int32)),
            "voxels": (voxels.dtype, np.dtype(np.int32)),
            "candidate": (candidate.dtype, np.dtype(np.int64)),
        }
        for name, (actual_dtype, expected_dtype) in expected_dtypes.items():
            _require(actual_dtype == expected_dtype, f"{path.name} {name} dtype differs")
        _require(
            all(
                np.isfinite(value).all()
                for value in (raw, fmt, labels, seeds, vortex_ids, voxels, candidate)
            ),
            f"{path.name} contains non-finite values",
        )
        _require(
            int(candidate.min()) == int(entry["source_candidate_index_min"])
            and int(candidate.max()) == int(entry["source_candidate_index_max"]),
            f"{path.name} candidate-index range differs from manifest",
        )
        _require(
            np.all(np.diff(candidate) > 0),
            f"{path.name} candidate indices are not strictly increasing",
        )
        pieces["source_candidate_indices"].append(candidate)
        pieces["voxel_indices_xyz"].append(voxels)
        pieces["seeds_xyz"].append(seeds)
        pieces["vortex_ids"].append(vortex_ids)
        pieces["targets"].append(labels)
        total += count
        del raw, fmt

    chunk_dir = manifest_path.parent / "chunks"
    actual_names = {path.name for path in chunk_dir.glob("*.npz") if path.is_file()}
    _require(
        actual_names == referenced_names,
        "target chunk directory and manifest differ: "
        f"missing={sorted(referenced_names - actual_names)}, "
        f"unexpected={sorted(actual_names - referenced_names)}",
    )
    target = {key: np.concatenate(value) for key, value in pieces.items()}
    _require(
        total == int(manifest["streamlines"]["valid_count"]),
        "concatenated target row count differs from manifest",
    )
    candidate = target["source_candidate_indices"]
    _require(np.all(np.diff(candidate) > 0), "target candidate indices repeat or reorder rows")
    _require(
        len(np.unique(candidate)) == total,
        "target source_candidate_indices contain duplicate rows",
    )
    _require(
        np.array_equal(target["vortex_ids"] > 0, target["targets"] >= 2),
        "target VortexId support disagrees with hairpin classes",
    )

    resolution = np.asarray(manifest["target_grid"]["resolution_xyz"], dtype=np.int64)
    voxels = target["voxel_indices_xyz"].astype(np.int64)
    _require(
        np.all((voxels >= 0) & (voxels < resolution[None, :])),
        "target voxel indices fall outside the target grid",
    )
    linear = (voxels[:, 2] * resolution[1] + voxels[:, 1]) * resolution[0] + voxels[:, 0]
    _require(len(np.unique(linear)) == total, "target voxel indices contain duplicates")
    domain_min = np.asarray(manifest["target_grid"]["domain_min_xyz"], dtype=np.float64)
    voxel_size = np.asarray(manifest["target_grid"]["voxel_size_xyz"], dtype=np.float64)
    expected_seeds = (
        domain_min[None, :] + (voxels.astype(np.float64) + 0.5) * voxel_size[None, :]
    ).astype(np.float32)
    _require(
        np.array_equal(target["seeds_xyz"], expected_seeds),
        "target seeds do not exactly map to the recorded voxel centers",
    )
    support = np.bincount(target["targets"].astype(np.int64), minlength=4)
    reported_support = manifest["streamlines"]["valid_class_counts"]
    _require(
        support.tolist() == [int(reported_support[name]) for name in CLASS_NAMES],
        "target class support differs from manifest",
    )
    before_support = manifest["proxy"]["class_counts_before_streamline_validity"]
    invalid_support = manifest["streamlines"]["invalid_class_counts"]
    for class_id, name in enumerate(CLASS_NAMES):
        _require(
            int(before_support[name])
            == int(support[class_id]) + int(invalid_support[name]),
            f"target pre-validity support is inconsistent for {name}",
        )
    _require(
        sum(int(before_support[name]) for name in CLASS_NAMES)
        == int(manifest["streamlines"]["candidate_count"]),
        "target pre-validity class support differs from candidate count",
    )
    validity_by_id = manifest["streamlines"].get("per_vortex_id_validity", {})
    _require(validity_by_id, "target manifest lacks per-VortexId validity")
    zero_valid_ids = []
    for key, row in validity_by_id.items():
        vortex_id = int(key)
        valid_count = int(np.count_nonzero(target["vortex_ids"] == vortex_id))
        _require(
            int(row["valid"]) == valid_count,
            f"target valid count differs for VortexId {vortex_id}",
        )
        if valid_count == 0:
            zero_valid_ids.append(vortex_id)
    _require(
        sorted(zero_valid_ids)
        == sorted(int(value) for value in manifest["streamlines"]["vortex_ids_with_zero_valid_rows"]),
        "zero-valid-row VortexId list differs",
    )
    return target, {
        "chunk_count": len(entries),
        "row_count": total,
        "class_support": {
            name: int(support[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "unique_source_candidate_index_count": int(len(np.unique(candidate))),
        "unique_voxel_count": int(len(np.unique(linear))),
        "vortex_id_count_before_validity": int(len(validity_by_id)),
        "vortex_id_count_with_valid_rows": int(len(validity_by_id) - len(zero_valid_ids)),
        "vortex_ids_with_zero_valid_rows": sorted(zero_valid_ids),
        "ordered_test_row_sha256": _array_sha256(
            target["source_candidate_indices"],
            target["voxel_indices_xyz"],
            target["seeds_xyz"],
            target["vortex_ids"],
            target["targets"],
        ),
    }


def _average_precision(binary_targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(binary_targets, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    _require(targets.shape == values.shape, "average-precision inputs differ in shape")
    _require(set(np.unique(targets)).issubset({0, 1}), "average-precision target is not binary")
    positive_count = int(np.count_nonzero(targets))
    _require(0 < positive_count < len(targets), "average precision needs both binary classes")
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_targets = targets[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_values[1:] != sorted_values[:-1]), len(values) - 1
    ]
    true_positives = np.cumsum(sorted_targets, dtype=np.int64)[group_ends]
    prediction_counts = group_ends.astype(np.int64) + 1
    precision = true_positives / prediction_counts
    positive_increments = np.diff(np.r_[0, true_positives])
    return float(np.sum(precision * positive_increments / positive_count))


def _per_class_statistics(
    targets: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    _require(targets.shape == predicted.shape, "targets and predictions differ in shape")
    _require(np.isin(targets, np.arange(4)).all(), "targets contain invalid classes")
    _require(np.isin(predicted, np.arange(4)).all(), "predictions contain invalid classes")
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(4, dtype=np.float64),
        where=predicted_support > 0,
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
    return confusion, support, precision, recall, f1


def primary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Recompute the trainer's four-class metrics without importing it."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64)
    _require(probability.shape == (len(targets), 4), "four-class probability shape differs")
    predicted = np.argmax(probability, axis=1)
    confusion, support, precision, recall, f1 = _per_class_statistics(targets, predicted)
    average_precision = np.asarray(
        [_average_precision(targets == index, probability[:, index]) for index in range(4)]
    )
    return {
        "sample_count": int(len(targets)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_average_precision_ovr": float(np.mean(average_precision)),
        "per_class_f1": {
            name: float(f1[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_precision": {
            name: float(precision[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_recall": {
            name: float(recall[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_average_precision_ovr": {
            name: float(average_precision[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "support": {
            name: int(support[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def _class_f1(targets: np.ndarray, predicted: np.ndarray, class_id: int) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    true_positive = int(np.count_nonzero((targets == class_id) & (predicted == class_id)))
    false_positive = int(np.count_nonzero((targets != class_id) & (predicted == class_id)))
    false_negative = int(np.count_nonzero((targets == class_id) & (predicted != class_id)))
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def _macro_f1_for_classes(
    targets: np.ndarray, predicted: np.ndarray, classes: tuple[int, ...]
) -> float:
    return float(np.mean([_class_f1(targets, predicted, value) for value in classes]))


def diagnostic_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    vortex_ids: np.ndarray,
    validity_by_id: dict[str, Any],
) -> dict[str, Any]:
    """Recompute valid-row and coverage-adjusted hierarchy diagnostics."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities)
    vortex_ids = np.asarray(vortex_ids, dtype=np.int32).reshape(-1)
    _require(probability.shape == (len(targets), 4), "diagnostic probability shape differs")
    _require(vortex_ids.shape == targets.shape, "diagnostic VortexId shape differs")
    predicted = np.argmax(probability, axis=1)
    true_hairpin = targets >= 2
    predicted_hairpin = predicted >= 2
    hairpin_rows = np.flatnonzero(true_hairpin)
    ordinary_rows = np.flatnonzero(~true_hairpin)
    _require(len(hairpin_rows) > 0 and len(ordinary_rows) > 0, "target lacks a hierarchy branch")
    hairpin_probability = probability[:, 2] + probability[:, 3]
    oracle_hairpin = 2 + np.argmax(probability[hairpin_rows, 2:4], axis=1)
    oracle_ordinary = np.argmax(probability[ordinary_rows, 0:2], axis=1)
    per_vortex: dict[str, Any] = {}
    valid_id_errors: list[float] = []
    valid_id_f1: list[float] = []
    valid_id_oracle_f1: list[float] = []
    coverage_adjusted_id_errors: list[float] = []
    total_hairpin_before = 0
    total_hairpin_invalid = 0
    _require(validity_by_id, "manifest contains no pre-validity VortexIds")
    for key in sorted(validity_by_id, key=int):
        vortex_id = int(key)
        member = vortex_ids == int(vortex_id)
        valid_support = int(np.count_nonzero(member))
        validity = validity_by_id[key]
        before = int(validity["before"])
        recorded_valid = int(validity["valid"])
        _require(
            before > 0 and recorded_valid == valid_support and before >= valid_support,
            f"VortexId {vortex_id} validity counts disagree with target rows",
        )
        _assert_close(
            f"VortexId {vortex_id} validity fraction",
            validity["fraction"],
            valid_support / before,
        )
        invalid_support = before - valid_support
        total_hairpin_before += before
        total_hairpin_invalid += invalid_support
        valid_ordinary_errors = int(np.count_nonzero(predicted[member] < 2))
        adjusted_error = float((invalid_support + valid_ordinary_errors) / before)
        coverage_adjusted_id_errors.append(adjusted_error)
        before_class_counts = validity.get("class_counts_before", {})
        valid_class_counts = validity.get("class_counts_valid", {})
        _require(
            set(before_class_counts) == set(CLASS_NAMES)
            and set(valid_class_counts) == set(CLASS_NAMES),
            f"VortexId {vortex_id} class-count keys differ",
        )
        _require(
            sum(int(value) for value in before_class_counts.values()) == before,
            f"VortexId {vortex_id} pre-validity class counts differ",
        )
        _require(
            int(before_class_counts[CLASS_NAMES[0]]) == 0
            and int(before_class_counts[CLASS_NAMES[1]]) == 0,
            f"VortexId {vortex_id} has ordinary pre-validity labels",
        )
        independently_valid_counts = {
            name: int(np.count_nonzero(member & (targets == class_id)))
            for class_id, name in enumerate(CLASS_NAMES)
        }
        _require(
            {name: int(valid_class_counts[name]) for name in CLASS_NAMES}
            == independently_valid_counts,
            f"VortexId {vortex_id} valid class counts differ from target rows",
        )
        row: dict[str, Any] = {
            "support_before_streamline_validity": before,
            "valid_support": valid_support,
            "invalid_support": invalid_support,
            "valid_fraction": float(valid_support / before),
            "hairpin_to_ordinary_or_invalid_error_rate": adjusted_error,
        }
        if valid_support:
            member_targets = targets[member]
            member_predicted = predicted[member]
            member_oracle = 2 + np.argmax(probability[member, 2:4], axis=1)
            valid_error = float(valid_ordinary_errors / valid_support)
            valid_four_class_f1 = _macro_f1_for_classes(
                member_targets, member_predicted, (2, 3)
            )
            valid_oracle_f1_value = _macro_f1_for_classes(
                member_targets, member_oracle, (2, 3)
            )
            row.update(
                {
                    "valid_hairpin_to_ordinary_error_rate": valid_error,
                    "valid_four_class_head_limb_macro_f1": valid_four_class_f1,
                    "valid_known_hairpin_mask_head_limb_macro_f1": valid_oracle_f1_value,
                }
            )
            valid_id_errors.append(valid_error)
            valid_id_f1.append(valid_four_class_f1)
            valid_id_oracle_f1.append(valid_oracle_f1_value)
        else:
            row.update(
                {
                    "valid_hairpin_to_ordinary_error_rate": None,
                    "valid_four_class_head_limb_macro_f1": None,
                    "valid_known_hairpin_mask_head_limb_macro_f1": None,
                }
            )
        per_vortex[str(key)] = row
    _require(
        total_hairpin_before > 0
        and total_hairpin_before - total_hairpin_invalid == len(hairpin_rows),
        "pooled hairpin validity counts disagree with target rows",
    )
    _require(valid_id_errors, "no VortexId retains a valid streamline row")
    binary_targets = true_hairpin.astype(np.int8)
    binary_predicted = predicted_hairpin.astype(np.int8)
    valid_pooled_error = float(np.mean(predicted[hairpin_rows] < 2))
    return {
        "hairpin_membership_f1": _class_f1(binary_targets, binary_predicted, 1),
        "hairpin_membership_average_precision": _average_precision(
            binary_targets, hairpin_probability
        ),
        "valid_hairpin_to_ordinary_error_rate": valid_pooled_error,
        "hairpin_to_ordinary_error_rate": valid_pooled_error,
        "valid_vortex_id_equal_hairpin_to_ordinary_error_rate": float(
            np.mean(valid_id_errors)
        ),
        "hairpin_to_ordinary_or_invalid_error_rate": float(
            (
                total_hairpin_invalid
                + int(np.count_nonzero(predicted[hairpin_rows] < 2))
            )
            / total_hairpin_before
        ),
        "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate": float(
            np.mean(coverage_adjusted_id_errors)
        ),
        "hairpin_streamline_valid_fraction": float(
            len(hairpin_rows) / total_hairpin_before
        ),
        "target_vortex_id_count_before_validity": int(len(validity_by_id)),
        "target_vortex_id_count_with_valid_rows": int(len(valid_id_errors)),
        "known_hairpin_mask_head_limb_macro_f1": _macro_f1_for_classes(
            targets[hairpin_rows], oracle_hairpin, (2, 3)
        ),
        "known_ordinary_mask_orientation_macro_f1": _macro_f1_for_classes(
            targets[ordinary_rows], oracle_ordinary, (0, 1)
        ),
        "valid_vortex_id_equal_four_class_head_limb_macro_f1": float(
            np.mean(valid_id_f1)
        ),
        "valid_vortex_id_equal_known_mask_head_limb_macro_f1": float(
            np.mean(valid_id_oracle_f1)
        ),
        "per_vortex_id": per_vortex,
    }


def _validate_probability_matrix(label: str, probability: np.ndarray) -> None:
    _require(probability.ndim == 2 and probability.shape[1] == 4, f"{label} shape differs")
    _require(np.isfinite(probability).all(), f"{label} contains non-finite values")
    _require(
        np.all((probability >= 0.0) & (probability <= 1.0)),
        f"{label} contains values outside [0,1]",
    )
    _require(
        np.allclose(
            probability.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=PROBABILITY_TOLERANCE,
        ),
        f"{label} rows do not sum to one",
    )


def _read_history(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"missing history: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(rows, f"history is empty: {path.name}")
    return rows


def _permutation_sha256(sample_count: int, seed: int, epoch: int) -> str:
    order = np.random.default_rng(int(seed) + int(epoch)).permutation(
        int(sample_count)
    ).astype(np.int64, copy=False)
    return hashlib.sha256(np.ascontiguousarray(order).view(np.uint8)).hexdigest()


def _validate_history(
    path: Path,
    run: dict[str, Any],
    *,
    sample_count: int,
    seed: int,
    required_streak: int,
) -> dict[str, Any]:
    rows = _read_history(path)
    selected_epoch = int(run.get("source_selected_epoch", -1))
    _require(len(rows) == selected_epoch, f"{path.name} length differs from selected epoch")
    for index, row in enumerate(rows, start=1):
        epoch = int(row["epoch"])
        _require(epoch == index, f"{path.name} epochs are not consecutive")
        _require(int(row["draw_count"]) == sample_count, f"{path.name} draw count differs")
        _require(int(row["unique_count"]) == sample_count, f"{path.name} misses source rows")
        _require(int(row["duplicate_count"]) == 0, f"{path.name} duplicates source rows")
        _require(int(row["missing_count"]) == 0, f"{path.name} misses source rows")
        _require(
            row["order_sha256"] == _permutation_sha256(sample_count, seed, epoch),
            f"{path.name} epoch {epoch} permutation hash differs",
        )
    final = rows[-1]
    _require(
        int(run.get("source_terminal_zero_error_streak", -1)) == required_streak,
        f"{path.name} terminal streak differs",
    )
    _require(
        int(final["zero_error_streak"]) == required_streak,
        f"{path.name} final streak is not {required_streak}",
    )
    _require(len(rows) >= required_streak, f"{path.name} is shorter than its stop gate")
    for offset, row in enumerate(rows[-required_streak:], start=1):
        _require(int(row["source_error_count"]) == 0, f"{path.name} terminal error is nonzero")
        _assert_close(f"{path.name} terminal accuracy", row["source_accuracy"], 1.0)
        _assert_close(f"{path.name} terminal macro F1", row["source_macro_f1"], 1.0)
        _require(
            float(row["minimum_true_logit_margin"]) > 0.0,
            f"{path.name} terminal true-class margin is not positive",
        )
        _require(
            int(row["zero_error_streak"]) == offset,
            f"{path.name} terminal streak sequence differs",
        )
    source_metrics = run["source_metrics"]
    _assert_close(
        f"{path.name} final/reported accuracy",
        final["source_accuracy"],
        source_metrics["accuracy"],
    )
    _require(
        int(final["source_error_count"]) == int(source_metrics["error_count"]),
        f"{path.name} final/reported source error count differs",
    )
    _assert_close(
        f"{path.name} final/reported macro F1",
        final["source_macro_f1"],
        source_metrics["macro_f1"],
    )
    _assert_close(
        f"{path.name} final/reported minimum logit margin",
        final["minimum_true_logit_margin"],
        source_metrics["minimum_true_logit_margin"],
    )
    return {
        "epoch_count": len(rows),
        "terminal_zero_error_streak": required_streak,
        "all_epochs_exact_without_replacement": True,
        "final_minimum_true_logit_margin": float(final["minimum_true_logit_margin"]),
    }


def _compare_reported_metrics(
    label: str,
    reported: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    atol: float,
) -> None:
    for key, value in recomputed.items():
        _require(key in reported, f"{label} lacks {key}")
        _compare_structure(f"{label}.{key}", reported[key], value, atol=atol)


def _validate_run(
    *,
    variant: str,
    seed: int,
    output_dir: Path,
    source_labels: np.ndarray,
    target: dict[str, np.ndarray],
    source_cache_hash: str,
    manifest_hash: str,
    config_hash: str,
    normalization_hash: str,
    validity_by_id: dict[str, Any],
    required_streak: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stem = f"{variant}_seed{seed}"
    run_path = output_dir / "runs" / f"{stem}.json"
    prediction_path = output_dir / "predictions" / f"{stem}.npz"
    history_path = output_dir / "histories" / f"{stem}.csv"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    _require(run.get("variant") == variant, f"{stem} variant differs")
    _require(int(run.get("seed", -1)) == seed, f"{stem} seed differs")
    _require(run.get("source_gate_passed") is True, f"{stem} source gate is false")
    _require(
        run.get("checkpoint_policy")
        == "model_state_in_memory_only_no_persistent_checkpoint",
        f"{stem} checkpoint policy differs",
    )
    _require(int(run.get("target_evaluation_count", -1)) == 1, f"{stem} target evaluated more than once")
    _require(run.get("source_cache_sha256") == source_cache_hash, f"{stem} source hash differs")
    _require(run.get("target_cache_manifest_sha256") == manifest_hash, f"{stem} manifest hash differs")
    _require(run.get("config_sha256") == config_hash, f"{stem} config hash differs")
    _require(
        run.get("normalization_sha256") == normalization_hash,
        f"{stem} normalization hash differs",
    )
    _require(_sha256(prediction_path) == run.get("prediction_sha256"), f"{stem} prediction hash differs")
    _require(_sha256(history_path) == run.get("history_sha256"), f"{stem} history hash differs")
    _require(Path(str(run.get("prediction_path", ""))).name == prediction_path.name, f"{stem} prediction path differs")
    _require(Path(str(run.get("history_path", ""))).name == history_path.name, f"{stem} history path differs")

    required_arrays = {
        "source_row_indices",
        "source_targets",
        "source_probabilities",
        "source_predicted_labels",
        "target_source_candidate_indices",
        "target_voxel_indices_xyz",
        "target_seeds_xyz",
        "target_vortex_ids",
        "target_targets",
        "target_probabilities",
        "target_predicted_labels",
    }
    with np.load(prediction_path, allow_pickle=False) as artifact:
        _require(set(artifact.files) == required_arrays, f"{stem} prediction keys differ")
        source_rows = np.asarray(artifact["source_row_indices"])
        saved_source_targets = np.asarray(artifact["source_targets"])
        source_probability = np.asarray(artifact["source_probabilities"])
        source_predicted = np.asarray(artifact["source_predicted_labels"])
        target_candidate = np.asarray(artifact["target_source_candidate_indices"])
        target_voxels = np.asarray(artifact["target_voxel_indices_xyz"])
        target_seeds = np.asarray(artifact["target_seeds_xyz"])
        target_vortex_ids = np.asarray(artifact["target_vortex_ids"])
        target_targets = np.asarray(artifact["target_targets"])
        target_probability = np.asarray(artifact["target_probabilities"])
        target_predicted = np.asarray(artifact["target_predicted_labels"])

    source_count = len(source_labels)
    target_count = len(target["targets"])
    _require(source_rows.dtype == np.int64, f"{stem} source row dtype differs")
    _require(saved_source_targets.dtype == np.int8, f"{stem} source target dtype differs")
    _require(source_probability.dtype == np.float32, f"{stem} source probability dtype differs")
    _require(source_predicted.dtype == np.int8, f"{stem} source prediction dtype differs")
    _require(target_candidate.dtype == np.int64, f"{stem} target candidate dtype differs")
    _require(target_voxels.dtype == np.int32, f"{stem} target voxel dtype differs")
    _require(target_seeds.dtype == np.float32, f"{stem} target seed dtype differs")
    _require(target_vortex_ids.dtype == np.int32, f"{stem} target VortexId dtype differs")
    _require(target_targets.dtype == np.int8, f"{stem} target dtype differs")
    _require(target_probability.dtype == np.float32, f"{stem} target probability dtype differs")
    _require(target_predicted.dtype == np.int8, f"{stem} target prediction dtype differs")
    _require(source_rows.shape == (source_count,), f"{stem} source row shape differs")
    _require(source_probability.shape == (source_count, 4), f"{stem} source probability shape differs")
    _require(target_probability.shape == (target_count, 4), f"{stem} target probability shape differs")
    _require(int(run.get("target_row_count", -1)) == target_count, f"{stem} target row count differs")
    _require(
        np.array_equal(source_rows, np.arange(source_count, dtype=np.int64)),
        f"{stem} does not evaluate every channel source row in cache order",
    )
    _require(
        np.array_equal(saved_source_targets.astype(np.int64), source_labels),
        f"{stem} source targets differ from channel cache",
    )
    expected_target_arrays = {
        "candidate": (target_candidate, target["source_candidate_indices"]),
        "voxel": (target_voxels, target["voxel_indices_xyz"]),
        "seed": (target_seeds, target["seeds_xyz"]),
        "VortexId": (target_vortex_ids, target["vortex_ids"]),
        "target": (target_targets, target["targets"]),
    }
    for name, (actual, expected) in expected_target_arrays.items():
        _require(np.array_equal(actual, expected), f"{stem} target {name} rows differ from chunks")

    _validate_probability_matrix(f"{stem} source probabilities", source_probability)
    _validate_probability_matrix(f"{stem} target probabilities", target_probability)
    source_argmax = np.argmax(source_probability, axis=1).astype(np.int8)
    target_argmax = np.argmax(target_probability, axis=1).astype(np.int8)
    _require(np.array_equal(source_predicted, source_argmax), f"{stem} source argmax differs")
    _require(np.array_equal(target_predicted, target_argmax), f"{stem} target argmax differs")
    source_errors = int(np.count_nonzero(source_argmax.astype(np.int64) != source_labels))
    _require(source_errors == 0, f"{stem} has {source_errors} channel source errors")
    true_probability = source_probability[np.arange(source_count), source_labels]
    masked_probability = source_probability.copy()
    masked_probability[np.arange(source_count), source_labels] = -np.inf
    minimum_probability_margin = float(np.min(true_probability - masked_probability.max(axis=1)))
    _require(minimum_probability_margin > 0.0, f"{stem} source probability margin is not positive")

    recomputed_source = primary_metrics(source_labels, source_probability)
    recomputed_source.update({"accuracy": 1.0, "error_count": 0})
    _compare_reported_metrics(
        f"{stem}.source_metrics",
        run.get("source_metrics", {}),
        recomputed_source,
        atol=SOURCE_METRIC_TOLERANCE,
    )
    _require(
        float(run["source_metrics"].get("minimum_true_logit_margin", -np.inf)) > 0.0,
        f"{stem} reported source logit margin is not positive",
    )
    _require(
        float(run["source_metrics"].get("mean_true_logit_margin", -np.inf))
        >= float(run["source_metrics"]["minimum_true_logit_margin"]),
        f"{stem} reported source mean margin is below its minimum",
    )

    target_primary = primary_metrics(target_targets, target_probability)
    target_diagnostics = diagnostic_metrics(
        target_targets,
        target_probability,
        target_vortex_ids,
        validity_by_id,
    )
    reported_target = run.get("target_metrics", {})
    _compare_structure(
        f"{stem}.target_metrics.primary",
        reported_target.get("primary"),
        target_primary,
    )
    _compare_structure(
        f"{stem}.target_metrics.diagnostics",
        reported_target.get("diagnostics"),
        target_diagnostics,
    )
    history_evidence = _validate_history(
        history_path,
        run,
        sample_count=source_count,
        seed=seed,
        required_streak=required_streak,
    )
    return run, {
        "variant": variant,
        "seed": seed,
        "source_error_count": source_errors,
        "source_minimum_true_probability_margin": minimum_probability_margin,
        "target_ordered_row_sha256": _array_sha256(
            target_candidate,
            target_voxels,
            target_seeds,
            target_vortex_ids,
            target_targets,
        ),
        "recomputed_source_metrics": recomputed_source,
        "recomputed_target_metrics": {
            "primary": target_primary,
            "diagnostics": target_diagnostics,
        },
        "history": history_evidence,
        "prediction_sha256": _sha256(prediction_path),
        "history_sha256": _sha256(history_path),
    }


def _aggregate(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    metric_paths = {
        "macro_f1": ("primary", "macro_f1"),
        "balanced_accuracy": ("primary", "balanced_accuracy"),
        "macro_average_precision_ovr": ("primary", "macro_average_precision_ovr"),
        "hairpin_membership_f1": ("diagnostics", "hairpin_membership_f1"),
        "valid_hairpin_to_ordinary_error_rate": (
            "diagnostics",
            "valid_hairpin_to_ordinary_error_rate",
        ),
        "valid_vortex_id_equal_hairpin_to_ordinary_error_rate": (
            "diagnostics",
            "valid_vortex_id_equal_hairpin_to_ordinary_error_rate",
        ),
        "hairpin_to_ordinary_or_invalid_error_rate": (
            "diagnostics",
            "hairpin_to_ordinary_or_invalid_error_rate",
        ),
        "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate": (
            "diagnostics",
            "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate",
        ),
        "known_hairpin_mask_head_limb_macro_f1": (
            "diagnostics",
            "known_hairpin_mask_head_limb_macro_f1",
        ),
        "valid_vortex_id_equal_four_class_head_limb_macro_f1": (
            "diagnostics",
            "valid_vortex_id_equal_four_class_head_limb_macro_f1",
        ),
    }
    for variant in spec["variants"]:
        selected = [row for row in rows if row["variant"] == variant]
        _require(len(selected) == 3, f"aggregate lacks three {variant} runs")
        item: dict[str, Any] = {
            "run_count": len(selected),
            "parameter_count": int(selected[0]["parameter_count"]),
            "source_gate_passed_count": int(
                sum(bool(row["source_gate_passed"]) for row in selected)
            ),
            "source_selected_epochs": [
                int(row["source_selected_epoch"]) for row in selected
            ],
        }
        _require(
            len({int(row["parameter_count"]) for row in selected}) == 1,
            f"{variant} parameter count changes across seeds",
        )
        for name, path in metric_paths.items():
            values = np.asarray(
                [row["target_metrics"][path[0]][path[1]] for row in selected],
                dtype=np.float64,
            )
            item[f"target_{name}_mean"] = float(np.mean(values))
            item[f"target_{name}_std"] = float(np.std(values, ddof=1))
        for class_name in CLASS_NAMES:
            values = np.asarray(
                [
                    row["target_metrics"]["primary"]["per_class_f1"][class_name]
                    for row in selected
                ],
                dtype=np.float64,
            )
            item[f"target_f1_{class_name}_mean"] = float(np.mean(values))
            item[f"target_f1_{class_name}_std"] = float(np.std(values, ddof=1))
        by_variant[variant] = item

    lookup = {(row["variant"], int(row["seed"])): row for row in rows}
    paired: dict[str, Any] = {}
    lower_is_better = {
        "valid_hairpin_to_ordinary_error_rate",
        "valid_vortex_id_equal_hairpin_to_ordinary_error_rate",
        "hairpin_to_ordinary_or_invalid_error_rate",
        "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate",
    }
    for baseline in ("raw", "raw_wide"):
        for metric, path in metric_paths.items():
            raw_differences = np.asarray(
                [
                    lookup[("raw_fmt", int(seed))]["target_metrics"][path[0]][path[1]]
                    - lookup[(baseline, int(seed))]["target_metrics"][path[0]][path[1]]
                    for seed in spec["training"]["seeds"]
                ],
                dtype=np.float64,
            )
            improvements = (
                -raw_differences if metric in lower_is_better else raw_differences
            )
            paired[f"raw_fmt_vs_{baseline}_{metric}_improvement"] = {
                "direction": (
                    "baseline_minus_raw_fmt"
                    if metric in lower_is_better
                    else "raw_fmt_minus_baseline"
                ),
                "mean": float(np.mean(improvements)),
                "std": float(np.std(improvements, ddof=1)),
                "improved_seed_count": int(np.count_nonzero(improvements > 0.0)),
                "per_seed": {
                    str(seed): float(value)
                    for seed, value in zip(spec["training"]["seeds"], improvements)
                },
            }
    return {
        "expected_run_count": 12,
        "completed_run_count": 12,
        "experiment_complete": True,
        "all_source_memorization_gates_passed": True,
        "variants": by_variant,
        "paired_comparisons": paired,
    }


def _validate_per_run_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _require(path.is_file(), "missing per_run_metrics.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    _require(len(csv_rows) == 12, "per_run_metrics.csv does not contain twelve rows")
    by_key = {(row["variant"], int(row["seed"])): row for row in rows}
    seen: set[tuple[str, int]] = set()
    for row in csv_rows:
        key = (row["variant"], int(row["seed"]))
        _require(key in by_key and key not in seen, f"invalid or repeated CSV run {key}")
        seen.add(key)
        source = by_key[key]
        _require(int(row["source_error_count"]) == 0, f"CSV source error for {key}")
        primary = source["target_metrics"]["primary"]
        diagnostics = source["target_metrics"]["diagnostics"]
        comparisons = {
            "target_macro_f1": primary["macro_f1"],
            "target_balanced_accuracy": primary["balanced_accuracy"],
            "target_macro_average_precision_ovr": primary["macro_average_precision_ovr"],
            "target_hairpin_membership_f1": diagnostics["hairpin_membership_f1"],
            "target_valid_hairpin_to_ordinary_error_rate": diagnostics[
                "valid_hairpin_to_ordinary_error_rate"
            ],
            "target_valid_vortex_id_equal_hairpin_to_ordinary_error_rate": diagnostics[
                "valid_vortex_id_equal_hairpin_to_ordinary_error_rate"
            ],
            "target_hairpin_to_ordinary_or_invalid_error_rate": diagnostics[
                "hairpin_to_ordinary_or_invalid_error_rate"
            ],
            "target_vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate": diagnostics[
                "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate"
            ],
            "target_known_hairpin_mask_head_limb_macro_f1": diagnostics[
                "known_hairpin_mask_head_limb_macro_f1"
            ],
            "target_valid_vortex_id_equal_four_class_head_limb_macro_f1": diagnostics[
                "valid_vortex_id_equal_four_class_head_limb_macro_f1"
            ],
        }
        for class_name in CLASS_NAMES:
            comparisons[f"target_f1_{class_name}"] = primary["per_class_f1"][class_name]
        for name, expected in comparisons.items():
            _assert_close(f"CSV {key} {name}", row[name], expected)
    _require(len(seen) == 12, "CSV run set is incomplete")
    return {"row_count": 12, "run_keys_match": True, "reported_metrics_match": True}


def _validate_summary(
    path: Path,
    *,
    spec: dict[str, Any],
    config_path: Path,
    source_hash: str,
    manifest_hash: str,
    normalization_path: Path,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    _require(path.is_file(), "missing summary.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    _require(summary.get("experiment") == EXPERIMENT, "summary experiment differs")
    _require(summary.get("config_sha256") == _sha256(config_path), "summary config hash differs")
    _require(summary.get("source", {}).get("sha256") == source_hash, "summary source hash differs")
    _require(
        summary.get("target_cache_manifest_sha256") == manifest_hash,
        "summary manifest hash differs",
    )
    _require(
        summary.get("normalization", {}).get("sha256") == _sha256(normalization_path),
        "summary normalization hash differs",
    )
    _require(
        summary.get("normalization", {}).get("fit_domain") == "channel only",
        "summary normalization domain differs",
    )
    _require(
        int(summary.get("normalization", {}).get("fit_row_count", -1)) == 20641,
        "summary normalization row count differs",
    )
    _require(
        summary.get("checkpoint_policy") == "no persistent model checkpoint",
        "summary checkpoint policy differs",
    )
    _require(
        summary.get("stale_run_paths_ignored") == [],
        "summary ignored one or more stale run artifacts",
    )
    shift = summary.get("target_distribution_shift", {})
    _require(
        shift.get("identity")
        == {
            "target_cache_manifest_sha256": manifest_hash,
            "normalization_sha256": _sha256(normalization_path),
        },
        "target distribution-shift identity differs",
    )
    _require(
        set(shift.get("statistics", {})) == {"raw", "fmt"},
        "target distribution-shift statistics are incomplete",
    )
    _compare_structure("summary.aggregate", summary.get("aggregate"), aggregate)
    summary_runs = summary.get("runs", ())
    _require(len(summary_runs) == 12, "summary does not contain twelve runs")
    indexed_summary = {
        (row["variant"], int(row["seed"])): row for row in summary_runs
    }
    indexed_files = {(row["variant"], int(row["seed"])): row for row in rows}
    _require(set(indexed_summary) == set(indexed_files), "summary run set differs")
    for key in indexed_files:
        _require(
            _canonical_json(indexed_summary[key]) == _canonical_json(indexed_files[key]),
            f"summary run {key} differs from run JSON",
        )
    interpretation = str(summary.get("interpretation_boundary", ""))
    _require("channel rows only" in interpretation, "summary lacks channel-only fit boundary")
    _require("evaluated once" in interpretation, "summary lacks one-shot TBL boundary")
    return {
        "summary_sha256": _sha256(path),
        "run_count": len(summary_runs),
        "aggregate_independently_recomputed": True,
    }


def _audit_complete(
    config_path: Path,
    spec: dict[str, Any],
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    variants, seeds = _validate_config(spec)
    inventory = _artifact_inventory(output_dir, variants, seeds)
    if not inventory["complete"]:
        raise IncompleteAuditError(
            "formal experiment is incomplete: exactly 12 run JSON, prediction NPZ, "
            "and history CSV artifacts are required",
            inventory,
        )

    snapshot_path = output_dir / "config_snapshot.yaml"
    _require(snapshot_path.is_file(), "missing frozen config snapshot")
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    _require(
        _canonical_json(snapshot) == _canonical_json(spec),
        "config snapshot differs from current frozen config",
    )
    source, source_metadata, source_path, source_hash = _load_source_cache(spec, repo_root)
    normalization_path = output_dir / "normalization_source_channel_only.npz"
    normalization = _validate_channel_only_normalization(
        normalization_path, source, source_metadata, spec
    )

    manifest_path = output_dir / "target_cache" / "target_cache_manifest.json"
    _require(manifest_path.is_file(), "missing target cache manifest")
    manifest_hash = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("experiment") == EXPERIMENT, "target manifest experiment differs")
    _require(manifest.get("config_sha256") == _sha256(config_path), "manifest config hash differs")
    _require(manifest.get("source_cache_sha256") == source_hash, "manifest source hash differs")
    _require(
        _canonical_json(manifest.get("representation"))
        == _canonical_json(spec.get("representation")),
        "target representation differs from config",
    )
    _require(
        _canonical_json(manifest.get("encoder")) == _canonical_json(spec.get("encoder")),
        "target encoder differs from config",
    )
    _require(
        int(manifest.get("streamlines", {}).get("steps_per_direction", -1))
        == int(spec["streamlines"]["steps_per_direction"]),
        "target streamline length differs from config",
    )
    _assert_close(
        "target offset/integration-step ratio",
        manifest.get("streamlines", {}).get("offset_over_spatial_step"),
        spec["streamlines"]["source_offset_over_spatial_step"],
        atol=1.0e-12,
    )
    _assert_close(
        "target proxy percentile",
        manifest.get("proxy", {}).get("percentile"),
        spec["target_proxy"]["percentile"],
        atol=1.0e-12,
    )
    _require(
        manifest.get("test_selection")
        == "all proxy candidate cubes, followed only by common nonperiodic streamline validity",
        "target test-row selection changed",
    )
    _require(
        manifest.get("target_use_boundary")
        == "target labels/features never select normalization, model weights, epoch, or hyperparameters",
        "target-use boundary changed",
    )

    external_hashes: dict[str, Any] = {}
    manifest_fields = {
        "target_flow": "flow_path",
        "target_gt": "gt_path",
        "target_vorticity_validation": "vorticity_validation_path",
    }
    for manifest_key, config_key in manifest_fields.items():
        configured = _resolve_file(spec["target"][config_key], repo_root=repo_root)
        recorded = _resolve_file(manifest[manifest_key], repo_root=repo_root)
        _require(configured == recorded, f"{manifest_key} path differs from config")
        recorded_hash = str(manifest[f"{manifest_key}_sha256"])
        actual_hash = _sha256(recorded)
        _require(actual_hash == recorded_hash, f"{manifest_key} SHA-256 differs")
        external_hashes[manifest_key] = {
            "path": str(recorded),
            "sha256": actual_hash,
        }
    curl_gate = _validate_curl_gate(manifest, spec)
    target, target_evidence = _load_target_chunks(manifest_path, manifest, spec)

    preparation_path = output_dir / "target_preparation_summary.json"
    _require(preparation_path.is_file(), "missing target preparation summary")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    _require(preparation.get("experiment") == EXPERIMENT, "preparation experiment differs")
    _require(
        preparation.get("target_cache_manifest_sha256") == manifest_hash,
        "preparation manifest hash differs",
    )
    _require(
        _canonical_json(preparation.get("curl_validation_audit"))
        == _canonical_json(manifest.get("curl_validation_audit")),
        "preparation and manifest curl audits differ",
    )
    _require(
        _canonical_json(preparation.get("target_class_support"))
        == _canonical_json(manifest["streamlines"]["valid_class_counts"]),
        "preparation target support differs",
    )

    model_files = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )
    _require(not model_files, f"persistent model checkpoints found: {model_files}")

    runs: list[dict[str, Any]] = []
    run_evidence: list[dict[str, Any]] = []
    required_streak = int(
        spec["training"]["stop_gate"]["consecutive_source_zero_error_epochs"]
    )
    for variant in variants:
        for seed in seeds:
            run, evidence = _validate_run(
                variant=variant,
                seed=seed,
                output_dir=output_dir,
                source_labels=source["labels"],
                target=target,
                source_cache_hash=source_hash,
                manifest_hash=manifest_hash,
                config_hash=_sha256(config_path),
                normalization_hash=_sha256(normalization_path),
                validity_by_id=manifest["streamlines"]["per_vortex_id_validity"],
                required_streak=required_streak,
            )
            _require(
                evidence["target_ordered_row_sha256"]
                == target_evidence["ordered_test_row_sha256"],
                f"{variant} seed {seed} did not use the canonical target rows",
            )
            runs.append(run)
            run_evidence.append(evidence)

    independently_recomputed_rows = [
        {
            **run,
            "source_metrics": evidence["recomputed_source_metrics"],
            "target_metrics": evidence["recomputed_target_metrics"],
        }
        for run, evidence in zip(runs, run_evidence)
    ]
    aggregate = _aggregate(independently_recomputed_rows, spec)
    summary_evidence = _validate_summary(
        output_dir / "summary.json",
        spec=spec,
        config_path=config_path,
        source_hash=source_hash,
        manifest_hash=manifest_hash,
        normalization_path=normalization_path,
        rows=runs,
        aggregate=aggregate,
    )
    csv_evidence = _validate_per_run_csv(output_dir / "per_run_metrics.csv", runs)
    row_signatures = {row["target_ordered_row_sha256"] for row in run_evidence}
    _require(len(row_signatures) == 1, "the twelve runs use different target test rows")
    return {
        "inventory": inventory,
        "gates": {
            "exactly_twelve_variant_seed_runs": True,
            "all_source_error_counts_zero": True,
            "all_source_terminal_streaks_pass": True,
            "all_source_epochs_are_complete_without_replacement_permutations": True,
            "all_prediction_probabilities_valid": True,
            "all_prediction_targets_match_source_or_target_cache": True,
            "all_twelve_runs_use_identical_ordered_target_rows": True,
            "all_primary_metrics_independently_recomputed": True,
            "all_diagnostic_metrics_independently_recomputed": True,
            "summary_aggregate_independently_recomputed": True,
            "normalization_recomputed_from_channel_rows_only": True,
            "curl_validation_gate_passed": True,
            "persistent_model_checkpoint_count": 0,
        },
        "source": {
            "path": str(source_path),
            "sha256": source_hash,
            "row_count": int(len(source["labels"])),
            "class_support": np.bincount(source["labels"], minlength=4).tolist(),
        },
        "normalization": normalization,
        "target_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_hash,
        },
        "external_target_sources": external_hashes,
        "curl_gate": curl_gate,
        "target_rows": target_evidence,
        "runs": run_evidence,
        "recomputed_aggregate": aggregate,
        "summary": summary_evidence,
        "per_run_csv": csv_evidence,
        "checkpoint_files": model_files,
    }


def audit(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    write_json: str | Path | None = None,
    raise_on_failure: bool = False,
) -> Path:
    """Run the audit, write a PASS/FAIL report, and return its path."""

    config_path = Path(config_path).resolve()
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent.resolve()
    configured_output = Path(spec["output_dir"])
    output_dir = (
        configured_output.resolve()
        if configured_output.is_absolute()
        else (repo_root / configured_output).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(write_json).resolve()
        if write_json is not None
        else output_dir / "independent_audit.json"
    )
    generated = datetime.now(timezone.utc).isoformat()
    failure: Exception | None = None
    try:
        evidence = _audit_complete(config_path, spec, repo_root, output_dir)
        report: dict[str, Any] = {
            "audit_version": AUDIT_VERSION,
            "experiment": EXPERIMENT,
            "generated_utc": generated,
            "status": "PASS",
            "passed": True,
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            **evidence,
        }
    except Exception as exc:
        failure = exc
        report = {
            "audit_version": AUDIT_VERSION,
            "experiment": EXPERIMENT,
            "generated_utc": generated,
            "status": "FAIL",
            "passed": False,
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "failure_type": type(exc).__name__,
            "failure": str(exc),
        }
        if isinstance(exc, IncompleteAuditError):
            report["incomplete_artifact_inventory"] = exc.inventory
    report["audit_script"] = str(Path(__file__).resolve())
    report["audit_script_sha256"] = _sha256(Path(__file__).resolve())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failure is not None and raise_on_failure:
        raise failure
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--write-json",
        help="Optional audit-report path; defaults to OUTPUT_DIR/independent_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = audit(args.config, write_json=args.write_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": report["status"],
                "audit_report": str(report_path.resolve()),
                "failure": report.get("failure"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
