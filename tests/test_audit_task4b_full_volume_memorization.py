import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.Audit_Task4B_FullVolumeMemorization_1_1 import (
    AuditError,
    CLASS_NAMES,
    SEEDS,
    VARIANTS,
    audit,
    epoch_permutation_sha256,
    exact_row_label_evidence,
    metric_bundle,
    recompute_all_sample_normalization,
    validate_cache_identity,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def complete_fixture(tmp_path: Path) -> tuple[Path, Path]:
    count = 8
    labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int8)
    split_codes = np.asarray([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int8)
    vortex_ids = np.asarray([0, 0, 2, 2, 0, 0, 4, 4], dtype=np.int32)
    source_candidates = np.arange(count, dtype=np.int64) + 100
    voxel_indices = np.arange(count * 3, dtype=np.int32).reshape(count, 3)
    seeds_xyz = (np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 10.0)
    raw = np.arange(count * 42, dtype=np.float32).reshape(count, 42) / 100.0
    fmt = np.arange(count * 7, dtype=np.float32).reshape(count, 7) / 50.0
    encoder = {
        "num_freq": 2,
        "mode": "magnitude",
        "include_chirality": False,
        "neighbor_pool": "sort",
        "neighbor_scale": 100.0,
        "neighbor_weight_after_train_standardization": 0.5,
        "expected_feature_dim": 7,
        "frozen": True,
    }
    metadata = {
        "encoder": {
            key: encoder[key]
            for key in (
                "num_freq",
                "mode",
                "include_chirality",
                "neighbor_pool",
                "neighbor_scale",
                "neighbor_weight_after_train_standardization",
            )
        },
        "streamlines": {"sampled_steps": 2},
    }
    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(
        cache_path,
        raw_features=raw,
        fmt_features=fmt,
        labels=labels,
        split_codes=split_codes,
        vortex_ids=vortex_ids,
        source_candidate_indices=source_candidates,
        voxel_indices_xyz=voxel_indices,
        seeds_xyz=seeds_xyz,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    spec = {
        "experiment": "Verify_Task4B_FullVolumeMemorization_1.1",
        "task": "test fixture",
        "output_dir": str(tmp_path / "out"),
        "cache": str(cache_path),
        "cache_sha256": _file_sha256(cache_path),
        "diagnostic_contract": {
            "fit_set": "all_20641_frozen_cache_rows",
            "evaluation_set": "exactly_the_same_all_20641_rows",
            "holdout": False,
            "allowed_conclusion": "finite_sample_memorization_only",
            "forbidden_conclusions": ["generalization"],
        },
        "variants": list(VARIANTS),
        "encoder": encoder,
        "model": {"dropout": 0.0},
        "training": {
            "seeds": list(SEEDS),
            "sampling": "one_without_replacement_permutation_of_every_row_per_epoch",
            "early_stopping": False,
            "weight_decay": 0.0,
        },
        "pass_gate": {
            "required_consecutive_zero_error_epochs": 1,
            "require_positive_minimum_true_logit_margin": True,
            "experiment_requires_every_variant_seed_run_to_pass": True,
            "expected_sample_count": count,
            "expected_class_support": [2, 2, 2, 2],
        },
    }
    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "out"
    for directory in ("runs", "predictions", "histories"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    normalization = recompute_all_sample_normalization(
        raw,
        fmt,
        sampled_steps=2,
        encoder_spec=encoder,
    )
    np.savez_compressed(output_dir / "normalization_all_samples.npz", **normalization)
    identity_sources = {
        "trainer_sha256": tmp_path
        / "experiments"
        / "Verify_Task4B_FullVolumeMemorization_1_1.py",
        "shared_training_helpers_sha256": tmp_path
        / "experiments"
        / "Train_Task4B_FourClassClassifier_1_1.py",
        "model_sha256": tmp_path / "FMT_Utils" / "Task4B_Classifier_3D.py",
        "raw_geometry_encoder_sha256": tmp_path
        / "FMT_Utils"
        / "PathlineClassifier_3D.py",
    }
    for source_path in identity_sources.values():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"fixture source: {source_path.name}\n", encoding="utf-8")
    artifact_identity = {
        "git_head": "1" * 40,
        "config_sha256": _file_sha256(config_path),
        **{name: _file_sha256(path) for name, path in identity_sources.items()},
    }

    logits = np.full((count, 4), -4.0, dtype=np.float32)
    logits[np.arange(count), labels] = 4.0
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    metrics = metric_bundle(labels, labels)
    run_documents = []
    for variant_index, variant in enumerate(VARIANTS):
        for seed in SEEDS:
            prediction_path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
            np.savez_compressed(
                prediction_path,
                cache_row_indices=np.arange(count, dtype=np.int64),
                source_candidate_indices=source_candidates,
                voxel_indices_xyz=voxel_indices,
                seeds_xyz=seeds_xyz,
                targets=labels,
                logits=logits,
                probabilities=probabilities,
                predicted_labels=labels,
                original_split_codes=split_codes,
                vortex_ids=vortex_ids,
            )
            history_path = output_dir / "histories" / f"{variant}_seed{seed}.csv"
            history_row = {
                "epoch": 1,
                "train_cross_entropy": 0.01,
                "fit_accuracy": 1.0,
                "fit_error_count": 0,
                "fit_macro_f1": 1.0,
                "fit_balanced_accuracy": 1.0,
                "minimum_true_logit_margin": 8.0,
                "mean_true_logit_margin": 8.0,
                "zero_error_streak": 1,
                "learning_rate": 0.001,
                "draw_count": count,
                "unique_count": count,
                "duplicate_count": 0,
                "missing_count": 0,
                "order_sha256": epoch_permutation_sha256(count, seed, 1),
            }
            with history_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history_row))
                writer.writeheader()
                writer.writerow(history_row)
            run = {
                "variant": variant,
                "seed": seed,
                "parameter_count": 1000 + variant_index,
                "passed": True,
                "selected_epoch": 1,
                "epochs_executed": 1,
                "terminal_zero_error_streak": 1,
                "required_zero_error_streak": 1,
                "elapsed_seconds": 1.0,
                **metrics,
                "minimum_true_logit_margin": 8.0,
                "mean_true_logit_margin": 8.0,
                "prediction_path": str(prediction_path),
                "history_path": str(history_path),
                "history_sha256": _file_sha256(history_path),
                "artifact_identity": artifact_identity,
                "checkpoint_policy": "selected_state_in_memory_only_no_model_file",
            }
            run_path = output_dir / "runs" / f"{variant}_seed{seed}.json"
            run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
            run_documents.append(run)
    summary = {
        "experiment": spec["experiment"],
        "cache_sha256": spec["cache_sha256"],
        "artifact_identity": artifact_identity,
        "normalization_sha256": _file_sha256(
            output_dir / "normalization_all_samples.npz"
        ),
        "fit_equals_evaluation": True,
        "holdout": False,
        "sample_count": count,
        "class_support": [2, 2, 2, 2],
        "selected_subset_run": False,
        "aggregate": {
            "expected_run_count": 9,
            "completed_run_count": 9,
            "all_expected_runs_present": True,
            "passed_run_count": 9,
            "experiment_passed": True,
            "maximum_error_count": 0,
            "runs": run_documents,
        },
        "interpretation_boundary": (
            "Fit and evaluation use exactly the same frozen cache rows. This is "
            "only a finite-sample memorization sanity check and contains no "
            "test or generalization evidence."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return config_path, output_dir


def test_independent_audit_accepts_complete_nine_run_zero_error_fixture(complete_fixture):
    config_path, output_dir = complete_fixture
    result = audit(config_path, output_dir)
    assert result["status"] == "PASS"
    assert result["strict_gate"]["passed_run_count"] == 9
    assert result["strict_gate"]["all_original_splits_zero_error"] is True
    assert result["strict_gate"]["all_positive_vortex_ids_zero_error"] is True
    identity = result["cache_identity_evidence"]
    assert identity["strict_gate"]["all_representation_conflict_group_counts_zero"]
    assert identity["source_candidate_indices"]["exact_duplicate_row_count"] == 0
    assert identity["voxel_indices_xyz"]["exact_duplicate_row_count"] == 0


def test_incomplete_prediction_set_fails_with_missing_filename(complete_fixture):
    config_path, output_dir = complete_fixture
    missing = output_dir / "predictions" / "raw_fmt_seed7070.npz"
    missing.unlink()
    with pytest.raises(AuditError, match="missing=.*raw_fmt_seed7070.npz"):
        audit(config_path, output_dir)


def test_permutation_hash_is_default_rng_seed_plus_epoch():
    expected = np.random.default_rng(7071).permutation(17).astype(np.int64)
    expected_hash = hashlib.sha256(np.ascontiguousarray(expected).view(np.uint8)).hexdigest()
    assert epoch_permutation_sha256(17, 7068, 3) == expected_hash


def test_metric_bundle_reports_exact_four_class_fit():
    targets = np.asarray([0, 1, 2, 3, 3], dtype=np.int64)
    metrics = metric_bundle(targets, targets)
    assert metrics["error_count"] == 0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert set(metrics["per_class_f1"]) == set(CLASS_NAMES)


def test_exact_row_conflict_reports_irreducible_error_and_is_rejected():
    conflicting_raw = np.asarray(
        [[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    evidence = exact_row_label_evidence(conflicting_raw, labels)
    assert evidence["exact_unique_row_count"] == 3
    assert evidence["exact_duplicate_group_count"] == 1
    assert evidence["cross_label_conflict_group_count"] == 1
    assert evidence["irreducible_minimum_error_count"] == 1
    arrays = {
        "raw_features": conflicting_raw,
        "fmt_features": np.eye(4, dtype=np.float32),
        "labels": labels,
        "source_candidate_indices": np.arange(4, dtype=np.int64),
        "voxel_indices_xyz": np.arange(12, dtype=np.int32).reshape(4, 3),
    }
    with pytest.raises(AuditError, match="bitwise-exact cross-label"):
        validate_cache_identity(arrays)
