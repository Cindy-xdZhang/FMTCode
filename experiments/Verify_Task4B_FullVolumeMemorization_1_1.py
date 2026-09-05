"""Verify that the Task4-b neural path can memorize the frozen full cache.

This is deliberately not a train/validation/test experiment: the fit set and
the evaluation set are the same 20,641 cached primitives.  Its only purpose is
to expose data/label/index/training-loop defects or insufficient finite-sample
capacity.  It must never be reported as generalization performance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np
import torch
from torch import nn
import yaml

from experiments.Train_Task4B_FourClassClassifier_1_1 import (
    _metrics,
    _normalize_train_only,
    _validate_cache_sha256,
)
from FMT_Utils.Task4B_Classifier_3D import (
    PathlineMulticlassClassifier3D,
    trainable_parameter_count,
)
from FMT_Utils.Task4B_ProxyLabels_3D import CLASS_NAMES


DEFAULT_CONFIG = Path("config/Verify_Task4B_FullVolumeMemorization_1.1.yaml")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256_bytes(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_spec_contract(spec: dict) -> None:
    """Fail closed if the executable no longer matches the frozen YAML."""

    expected = {
        "experiment": "Verify_Task4B_FullVolumeMemorization_1.1",
        "variants": ["raw", "fmt_only", "raw_fmt"],
        "diagnostic_contract": {
            "fit_set": "all_20641_frozen_cache_rows",
            "evaluation_set": "exactly_the_same_all_20641_rows",
            "holdout": False,
            "allowed_conclusion": "finite_sample_memorization_only",
            "forbidden_conclusions": [
                "generalization",
                "test_performance",
                "fmt_advantage",
            ],
        },
        "model": {
            "family": "PathlineMulticlassClassifier3D",
            "temporal_width": 128,
            "embedding_dim": 1024,
            "auxiliary_dim": 1024,
            "dropout": 0.0,
        },
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise RuntimeError(f"frozen executable contract mismatch for {key}")
    training_expected = {
        "seeds": [7068, 7069, 7070],
        "batch_size": 512,
        "evaluation_batch_size": 1024,
        "sampling": "one_without_replacement_permutation_of_every_row_per_epoch",
        "loss": "unweighted_cross_entropy",
        "label_smoothing": 0.0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "scheduler": "cosine_annealing",
        "minimum_learning_rate": 1.0e-6,
        "max_epochs": 1000,
        "early_stopping": False,
        "gradient_clipping": False,
        "mixed_precision": False,
        "checkpoint_policy": "in_memory_only_no_persistent_model_file",
    }
    if spec.get("training") != training_expected:
        raise RuntimeError("frozen executable contract mismatch for training")
    pass_expected = {
        "required_consecutive_zero_error_epochs": 3,
        "require_positive_minimum_true_logit_margin": True,
        "experiment_requires_every_variant_seed_run_to_pass": True,
        "expected_sample_count": 20641,
        "expected_class_support": [5000, 5000, 2829, 7812],
    }
    if spec.get("pass_gate") != pass_expected:
        raise RuntimeError("frozen executable contract mismatch for pass_gate")


def epoch_permutation(sample_count: int, seed: int, epoch: int) -> np.ndarray:
    """Return the frozen, one-pass, without-replacement order for an epoch."""

    if int(sample_count) < 1 or int(epoch) < 1:
        raise ValueError("sample_count and epoch must be positive")
    return np.random.default_rng(int(seed) + int(epoch)).permutation(
        int(sample_count)
    ).astype(np.int64, copy=False)


def permutation_certificate(order: np.ndarray, sample_count: int) -> dict:
    """Certify that an epoch visits every cache row exactly once."""

    order = np.asarray(order, dtype=np.int64).reshape(-1)
    counts = np.bincount(order, minlength=int(sample_count))
    if len(counts) != int(sample_count):
        raise RuntimeError("epoch order contains an out-of-range row")
    return {
        "draw_count": int(len(order)),
        "unique_count": int(np.count_nonzero(counts)),
        "duplicate_count": int(np.maximum(counts - 1, 0).sum()),
        "missing_count": int(np.count_nonzero(counts == 0)),
        "order_sha256": _sha256_bytes(order),
    }


def _write_history_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a small shared artifact with deterministic content."""

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Atomically replace a shared NumPy artifact.

    A process-specific ``.npz`` suffix prevents NumPy from changing the target
    name.  Concurrent sharded invocations all compute identical content.
    """

    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


@torch.no_grad()
def _predict_logits(
    model: nn.Module,
    raw: torch.Tensor,
    fmt: torch.Tensor,
    *,
    variant: str,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(raw), int(batch_size)):
        stop = min(len(raw), start + int(batch_size))
        chunks.append(
            model(
                raw[start:stop],
                fmt[start:stop] if variant in {"fmt_only", "raw_fmt"} else None,
            ).detach().cpu()
        )
    return torch.cat(chunks, dim=0)


def _fit_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics = _metrics(labels, probabilities)
    predictions = np.argmax(logits, axis=1)
    true_logits = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    margins = true_logits - masked.max(axis=1)
    metrics.update(
        {
            "accuracy": float(np.mean(predictions == labels)),
            "error_count": int(np.count_nonzero(predictions != labels)),
            "minimum_true_logit_margin": float(np.min(margins)),
            "mean_true_logit_margin": float(np.mean(margins)),
        }
    )
    return metrics


def _is_exact(metrics: dict) -> bool:
    return bool(
        metrics["error_count"] == 0
        and metrics["minimum_true_logit_margin"] > 0.0
    )


def _train_one(
    spec: dict,
    raw_np: np.ndarray,
    fmt_np: np.ndarray,
    labels_np: np.ndarray,
    split_codes: np.ndarray,
    vortex_ids: np.ndarray,
    source_candidate_indices: np.ndarray,
    voxel_indices_xyz: np.ndarray,
    seeds_xyz: np.ndarray,
    artifact_identity: dict,
    *,
    variant: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    _set_seed(seed)
    training = spec["training"]
    model_spec = spec["model"]
    pass_spec = spec["pass_gate"]
    sample_count = len(labels_np)
    if sample_count != int(pass_spec["expected_sample_count"]):
        raise RuntimeError("cache row count differs from the frozen pass gate")
    support = np.bincount(labels_np, minlength=4)
    if support.tolist() != [int(value) for value in pass_spec["expected_class_support"]]:
        raise RuntimeError("class support differs from the frozen pass gate")

    raw = torch.from_numpy(raw_np).to(device)
    fmt = torch.from_numpy(fmt_np).to(device)
    labels = torch.from_numpy(labels_np.astype(np.int64)).to(device)
    model = PathlineMulticlassClassifier3D(
        variant=variant,
        fmt_dim=fmt_np.shape[1],
        num_classes=4,
        temporal_width=int(model_spec["temporal_width"]),
        embedding_dim=int(model_spec["embedding_dim"]),
        auxiliary_dim=int(model_spec["auxiliary_dim"]),
        dropout=float(model_spec["dropout"]),
    ).to(device)
    parameter_count = trainable_parameter_count(model)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training["label_smoothing"])
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["max_epochs"]),
        eta_min=float(training["minimum_learning_rate"]),
    )

    history_path = output_dir / "histories" / f"{variant}_seed{seed}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()
    required_streak = int(pass_spec["required_consecutive_zero_error_epochs"])
    exact_streak = 0
    passed = False
    selected_epoch = 0
    selected_state = None
    selected_metrics = None
    best_key = None
    started = time.perf_counter()

    for epoch in range(1, int(training["max_epochs"]) + 1):
        order = epoch_permutation(sample_count, seed, epoch)
        certificate = permutation_certificate(order, sample_count)
        if (
            certificate["unique_count"] != sample_count
            or certificate["duplicate_count"] != 0
            or certificate["missing_count"] != 0
        ):
            raise RuntimeError("epoch permutation is not one complete cache pass")
        model.train()
        loss_sum = 0.0
        for start in range(0, sample_count, int(training["batch_size"])):
            batch_indices = torch.from_numpy(
                order[start : start + int(training["batch_size"])]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                raw[batch_indices],
                fmt[batch_indices]
                if variant in {"fmt_only", "raw_fmt"}
                else None,
            )
            loss = criterion(logits, labels[batch_indices])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_indices)

        logits_cpu = _predict_logits(
            model,
            raw,
            fmt,
            variant=variant,
            batch_size=int(training["evaluation_batch_size"]),
        )
        metrics = _fit_metrics(labels_np, logits_cpu.numpy())
        exact_streak = exact_streak + 1 if _is_exact(metrics) else 0
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_cross_entropy": loss_sum / sample_count,
            "fit_accuracy": metrics["accuracy"],
            "fit_error_count": metrics["error_count"],
            "fit_macro_f1": metrics["macro_f1"],
            "fit_balanced_accuracy": metrics["balanced_accuracy"],
            "minimum_true_logit_margin": metrics["minimum_true_logit_margin"],
            "mean_true_logit_margin": metrics["mean_true_logit_margin"],
            "zero_error_streak": exact_streak,
            "learning_rate": current_lr,
            **certificate,
        }
        _write_history_row(history_path, row)
        if (
            epoch <= 5
            or epoch % 10 == 0
            or metrics["error_count"] < 10
            or exact_streak > 0
        ):
            print(
                f"{variant} seed={seed} epoch={epoch:04d} "
                f"loss={row['train_cross_entropy']:.7f} "
                f"errors={metrics['error_count']} "
                f"macro_F1={metrics['macro_f1']:.7f} "
                f"min_margin={metrics['minimum_true_logit_margin']:.7g} "
                f"streak={exact_streak}/{required_streak}",
                flush=True,
            )
        key = (
            int(metrics["error_count"]),
            -float(metrics["minimum_true_logit_margin"]),
            float(row["train_cross_entropy"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            selected_epoch = epoch
            selected_metrics = copy.deepcopy(metrics)
            selected_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if exact_streak >= required_streak:
            passed = True
            selected_epoch = epoch
            selected_metrics = copy.deepcopy(metrics)
            selected_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            break
        scheduler.step()

    if selected_state is None or selected_metrics is None:
        raise RuntimeError("memorization training produced no selectable state")
    model.load_state_dict(selected_state)
    final_logits = _predict_logits(
        model,
        raw,
        fmt,
        variant=variant,
        batch_size=int(training["evaluation_batch_size"]),
    ).numpy()
    final_metrics = _fit_metrics(labels_np, final_logits)
    if (
        int(final_metrics["error_count"]) != int(selected_metrics["error_count"])
        or not np.isclose(
            final_metrics["minimum_true_logit_margin"],
            selected_metrics["minimum_true_logit_margin"],
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise RuntimeError("reloaded selected state does not reproduce its metrics")
    shifted = final_logits - final_logits.max(axis=1, keepdims=True)
    final_probabilities = np.exp(shifted)
    final_probabilities /= final_probabilities.sum(axis=1, keepdims=True)
    prediction_path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        cache_row_indices=np.arange(sample_count, dtype=np.int64),
        source_candidate_indices=source_candidate_indices.astype(np.int64),
        voxel_indices_xyz=voxel_indices_xyz.astype(np.int32),
        seeds_xyz=seeds_xyz.astype(np.float32),
        targets=labels_np.astype(np.int8),
        logits=final_logits.astype(np.float32),
        probabilities=final_probabilities.astype(np.float32),
        predicted_labels=np.argmax(final_logits, axis=1).astype(np.int8),
        original_split_codes=split_codes.astype(np.int8),
        vortex_ids=vortex_ids.astype(np.int32),
    )
    row = {
        "variant": variant,
        "seed": int(seed),
        "parameter_count": parameter_count,
        "passed": bool(passed and _is_exact(final_metrics)),
        "selected_epoch": int(selected_epoch),
        "epochs_executed": int(epoch),
        "terminal_zero_error_streak": int(exact_streak),
        "required_zero_error_streak": required_streak,
        "elapsed_seconds": float(time.perf_counter() - started),
        "sample_count": sample_count,
        "accuracy": final_metrics["accuracy"],
        "error_count": final_metrics["error_count"],
        "macro_f1": final_metrics["macro_f1"],
        "balanced_accuracy": final_metrics["balanced_accuracy"],
        "minimum_true_logit_margin": final_metrics[
            "minimum_true_logit_margin"
        ],
        "per_class_precision": final_metrics["per_class_precision"],
        "per_class_recall": final_metrics["per_class_recall"],
        "per_class_f1": final_metrics["per_class_f1"],
        "support": final_metrics["support"],
        "confusion_matrix_true_rows_predicted_columns": final_metrics[
            "confusion_matrix_true_rows_predicted_columns"
        ],
        "prediction_path": str(prediction_path.resolve()),
        "history_path": str(history_path.resolve()),
        "history_sha256": _sha256_file(history_path),
        "artifact_identity": artifact_identity,
        "checkpoint_policy": "selected_state_in_memory_only_no_model_file",
    }
    run_path = output_dir / "runs" / f"{variant}_seed{seed}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    del selected_state, model, raw, fmt, labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _load_and_validate(spec: dict) -> tuple[dict, dict]:
    cache_path = Path(spec["cache"])
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    actual_sha = _validate_cache_sha256(cache_path, spec["cache_sha256"])
    with np.load(cache_path) as cache:
        arrays = {
            "raw_features": np.asarray(cache["raw_features"], dtype=np.float32),
            "fmt_features": np.asarray(cache["fmt_features"], dtype=np.float32),
            "labels": np.asarray(cache["labels"], dtype=np.int64),
            "split_codes": np.asarray(cache["split_codes"], dtype=np.int8),
            "vortex_ids": np.asarray(cache["vortex_ids"], dtype=np.int32),
            "source_candidate_indices": np.asarray(
                cache["source_candidate_indices"], dtype=np.int64
            ),
            "voxel_indices_xyz": np.asarray(
                cache["voxel_indices_xyz"], dtype=np.int32
            ),
            "seeds_xyz": np.asarray(cache["seeds_xyz"], dtype=np.float32),
        }
        metadata = json.loads(str(cache["metadata_json"]))
    cache_encoder = metadata.get("encoder", {})
    for key in (
        "num_freq",
        "mode",
        "include_chirality",
        "neighbor_pool",
        "neighbor_scale",
        "neighbor_weight_after_train_standardization",
    ):
        if cache_encoder.get(key) != spec["encoder"].get(key):
            raise RuntimeError(f"cache/config encoder mismatch for {key}")
    if arrays["fmt_features"].shape[1] != int(spec["encoder"]["expected_feature_dim"]):
        raise RuntimeError("unexpected FMT feature width")
    if not np.array_equal(arrays["vortex_ids"] > 0, arrays["labels"] >= 2):
        raise RuntimeError("hairpin VortexId and four-class target disagree")
    return arrays, {"sha256": actual_sha, "metadata": metadata}


def _aggregate(spec: dict, rows: list[dict]) -> dict:
    expected = {
        (str(variant), int(seed))
        for variant in spec["variants"]
        for seed in spec["training"]["seeds"]
    }
    keys = [(row["variant"], int(row["seed"])) for row in rows]
    actual = set(keys)
    keys_are_unique = len(keys) == len(actual)
    counts_match = len(rows) == len(expected) == len(actual) and len(expected) > 0
    every_run_passed = counts_match and all(bool(row["passed"]) for row in rows)
    return {
        "expected_run_count": len(expected),
        "completed_run_count": len(rows),
        "run_keys_are_unique": keys_are_unique,
        "all_expected_runs_present": bool(
            counts_match and keys_are_unique and actual == expected
        ),
        "passed_run_count": int(sum(bool(row["passed"]) for row in rows)),
        "experiment_passed": bool(
            counts_match
            and keys_are_unique
            and actual == expected
            and every_run_passed
        ),
        "maximum_error_count": int(max((row["error_count"] for row in rows), default=-1)),
        "runs": rows,
    }


def run(
    config_path: str | Path,
    *,
    output_override: str | None = None,
    selected_variant: str | None = None,
    selected_seed: int | None = None,
    smoke_test: bool = False,
) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_spec_contract(spec)
    if smoke_test:
        spec = copy.deepcopy(spec)
        spec["experiment"] = "Verify_Task4B_FullVolumeMemorizationSmoke_1.1"
        spec["variants"] = [selected_variant or "fmt_only"]
        spec["training"]["seeds"] = [
            int(selected_seed) if selected_seed is not None else 7068
        ]
        spec["training"]["max_epochs"] = 2
        spec["pass_gate"]["required_consecutive_zero_error_epochs"] = 1
        spec["output_dir"] = (
            "outputs/Verify_Task4B_FullVolumeMemorizationSmoke_1.1/channel_GTs"
        )
    output_dir = Path(output_override or spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.yaml"
    if snapshot_path.exists():
        previous = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        if json.loads(json.dumps(previous, sort_keys=True)) != json.loads(
            json.dumps(spec, sort_keys=True)
        ):
            raise RuntimeError("configuration changed; use a new experiment version")
    _atomic_write_text(snapshot_path, yaml.safe_dump(spec, sort_keys=False))
    artifact_identity = {
        "git_head": _git_head(),
        "config_sha256": _sha256_file(config_path),
        "trainer_sha256": _sha256_file(Path(__file__).resolve()),
        "model_sha256": _sha256_file(
            Path("FMT_Utils/Task4B_Classifier_3D.py").resolve()
        ),
        "shared_training_helpers_sha256": _sha256_file(
            Path("experiments/Train_Task4B_FourClassClassifier_1_1.py").resolve()
        ),
        "raw_geometry_encoder_sha256": _sha256_file(
            Path("FMT_Utils/PathlineClassifier_3D.py").resolve()
        ),
    }

    arrays, cache_info = _load_and_validate(spec)
    sampled_steps = int(cache_info["metadata"]["streamlines"]["sampled_steps"])
    all_fit_codes = np.zeros_like(arrays["split_codes"])
    raw, fmt, normalization = _normalize_train_only(
        arrays["raw_features"],
        arrays["fmt_features"],
        all_fit_codes,
        sampled_steps=sampled_steps,
        encoder_spec=spec["encoder"],
    )
    _atomic_savez(output_dir / "normalization_all_samples.npz", **normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = (
        [str(selected_variant)]
        if selected_variant is not None
        else [str(value) for value in spec["variants"]]
    )
    seeds = (
        [int(selected_seed)]
        if selected_seed is not None
        else [int(value) for value in spec["training"]["seeds"]]
    )
    if not set(variants).issubset(set(spec["variants"])):
        raise ValueError("selected variant is outside the frozen config")
    if not set(seeds).issubset({int(value) for value in spec["training"]["seeds"]}):
        raise ValueError("selected seed is outside the frozen config")
    if (selected_variant is None) != (selected_seed is None):
        raise ValueError("--variant and --seed must be provided together")

    rows = []
    for variant in variants:
        for seed in seeds:
            rows.append(
                _train_one(
                    spec,
                    raw,
                    fmt,
                    arrays["labels"],
                    arrays["split_codes"],
                    arrays["vortex_ids"],
                    arrays["source_candidate_indices"],
                    arrays["voxel_indices_xyz"],
                    arrays["seeds_xyz"],
                    artifact_identity,
                    variant=variant,
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                )
            )
    aggregate = _aggregate(spec, rows)
    summary = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "cache": str(Path(spec["cache"]).resolve()),
        "cache_sha256": cache_info["sha256"],
        "artifact_identity": artifact_identity,
        "normalization_sha256": _sha256_file(
            output_dir / "normalization_all_samples.npz"
        ),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
        },
        "fit_equals_evaluation": True,
        "holdout": False,
        "sample_count": int(len(arrays["labels"])),
        "class_support": np.bincount(arrays["labels"], minlength=4).tolist(),
        "selected_subset_run": bool(
            selected_variant is not None or selected_seed is not None or smoke_test
        ),
        "aggregate": aggregate,
        "interpretation_boundary": (
            "Fit and evaluation use exactly the same frozen cache rows. This is "
            "only a finite-sample memorization sanity check and contains no "
            "test or generalization evidence."
        ),
    }
    suffix = "summary.json"
    if selected_variant is not None or selected_seed is not None:
        suffix = f"summary_{variants[0]}_seed{seeds[0]}.json"
    (output_dir / suffix).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--variant", choices=("raw", "fmt_only", "raw_fmt"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.config,
        output_override=args.output_dir,
        selected_variant=args.variant,
        selected_seed=args.seed,
        smoke_test=args.smoke_test,
    )
