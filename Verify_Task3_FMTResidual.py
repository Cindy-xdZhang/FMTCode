"""Verify Task3 with a frozen Raw backbone and an additive FMT residual."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
import yaml

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D, PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Verify_Task3_FMTClassifier import (
    _append_csv, _classification_metrics, _config_without_seeds,
    _load_dataset, _loader, _normalize_train_only, _predict,
    _select_f1_threshold,
    _set_seed, _stack_split,
)


def _validate_snapshot(spec, path):
    path = Path(path)
    if not path.exists():
        return
    previous = yaml.safe_load(path.read_text(encoding="utf-8"))
    if _config_without_seeds(previous) != _config_without_seeds(spec):
        raise RuntimeError(
            f"configuration changed for existing output {path.parent}; "
            "use a new experiment version/output directory"
        )


def _load_raw_model(checkpoint_path, fmt_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "raw":
        raise ValueError(f"expected Raw checkpoint, got {checkpoint['variant']}")
    source = checkpoint["config"]
    model = PathlineBinaryClassifier3D(
        variant="raw", fmt_dim=fmt_dim,
        temporal_width=source["model"]["temporal_width"],
        embedding_dim=source["model"]["embedding_dim"],
        auxiliary_dim=source["model"]["auxiliary_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


@torch.no_grad()
def _predict_components(model, loader, device):
    model.eval()
    targets, raw_logits, residual_logits = [], [], []
    for raw, fmt, labels in loader:
        raw_logit, residual_logit = model.forward_components(
            raw.to(device, non_blocking=True),
            fmt.to(device, non_blocking=True),
        )
        targets.append(labels.numpy())
        raw_logits.append(raw_logit.cpu().numpy())
        residual_logits.append(residual_logit.cpu().numpy())
    raw_logits = np.concatenate(raw_logits)
    residual_logits = np.concatenate(residual_logits)
    if not np.isfinite(raw_logits).all():
        raise FloatingPointError("non-finite frozen Raw validation logits")
    if not np.isfinite(residual_logits).all():
        raise FloatingPointError("non-finite residual validation logits")
    return (
        np.concatenate(targets).astype(bool),
        raw_logits,
        residual_logits,
    )


def _select_alpha(targets, raw_logits, residual_logits, alpha_grid,
                  objective="average_precision", baseline_metrics=None,
                  minimum_f1_gain=0.02):
    if objective == "average_precision":
        scores = np.asarray([
            average_precision_score(targets, raw_logits + alpha * residual_logits)
            for alpha in alpha_grid
        ])
    elif objective == "f1":
        values = []
        for alpha in alpha_grid:
            probabilities = _probabilities(raw_logits, residual_logits, alpha)
            threshold = _select_f1_threshold(targets, probabilities)
            values.append(_classification_metrics(
                targets, probabilities, threshold
            )["f1"])
        scores = np.asarray(values)
    elif objective == "minimum_gain":
        if baseline_metrics is None:
            raise ValueError("minimum_gain selection requires baseline_metrics")
        values = []
        for alpha in alpha_grid:
            probabilities = _probabilities(raw_logits, residual_logits, alpha)
            threshold = _select_f1_threshold(targets, probabilities)
            metrics = _classification_metrics(targets, probabilities, threshold)
            values.append(min(
                metrics["f1"] - float(baseline_metrics["f1"]),
                metrics["average_precision"]
                - float(baseline_metrics["average_precision"]),
            ))
        scores = np.asarray(values)
    elif objective == "constrained_average_precision":
        if baseline_metrics is None:
            raise ValueError(
                "constrained_average_precision requires baseline_metrics"
            )
        candidates = []
        for alpha in alpha_grid:
            probabilities = _probabilities(raw_logits, residual_logits, alpha)
            threshold = _select_f1_threshold(targets, probabilities)
            metrics = _classification_metrics(targets, probabilities, threshold)
            candidates.append((float(alpha), metrics))
        feasible = [
            item for item in candidates
            if item[1]["f1"] >= (
                float(baseline_metrics["f1"]) + float(minimum_f1_gain)
            )
        ]
        if feasible:
            selected = max(
                feasible,
                key=lambda item: (
                    item[1]["average_precision"], item[1]["f1"]
                ),
            )
            # Feasible epochs must dominate every fallback epoch.
            return selected[0], 1.0 + selected[1]["average_precision"]
        selected = max(
            candidates,
            key=lambda item: min(
                item[1]["f1"] - float(baseline_metrics["f1"]),
                item[1]["average_precision"]
                - float(baseline_metrics["average_precision"]),
            ),
        )
        fallback_score = min(
            selected[1]["f1"] - float(baseline_metrics["f1"]),
            selected[1]["average_precision"]
            - float(baseline_metrics["average_precision"]),
        )
        return selected[0], float(fallback_score)
    else:
        raise ValueError(
            "fusion.selection_metric must be average_precision, f1, "
            "minimum_gain, or constrained_average_precision"
        )
    index = int(np.argmax(scores))
    return float(alpha_grid[index]), float(scores[index])


def _probabilities(raw_logits, residual_logits, alpha):
    logits = raw_logits + float(alpha) * residual_logits
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


class _WeightedFocalBCEWithLogitsLoss(nn.Module):
    """Class-weighted binary cross entropy with optional focal modulation.

    ``gamma=0`` is exactly weighted BCE.  Keeping both cases in one
    implementation makes the Task3 optimization search change only the
    declared loss recipe while preserving the same positive-class weighting
    rule for the paired FMT and Raw-PCA arms.
    """

    def __init__(self, pos_weight: float, gamma: float = 0.0):
        super().__init__()
        if not np.isfinite(pos_weight) or float(pos_weight) <= 0.0:
            raise ValueError("pos_weight must be finite and positive")
        if not np.isfinite(gamma) or float(gamma) < 0.0:
            raise ValueError("focal gamma must be finite and non-negative")
        self.register_buffer(
            "pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32)
        )
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if self.gamma > 0.0:
            # Mathematically, 1 - p_t = sigmoid(-s), where
            # s=(2y-1)*logit.  Computing ``(1-p_t)**gamma`` directly gives an
            # infinite derivative at p_t=1 for 0<gamma<1; the subsequent
            # sigmoid derivative is zero and autograd can produce 0*inf=NaN.
            # The log-domain form is equivalent but has a finite derivative
            # even for saturated, correctly classified samples.
            signed_logits = (2.0 * targets - 1.0) * logits
            focal_factor = torch.exp(
                self.gamma * F.logsigmoid(-signed_logits)
            )
            loss = loss * focal_factor
        return loss.mean()


def _build_training_loss(training: dict, positive: float, negative: float,
                         device: torch.device):
    """Build the declared paired Task3 loss and return its audit metadata."""
    if positive <= 0.0 or negative <= 0.0:
        raise ValueError("Task3 training labels must contain both classes")
    name = str(training.get("loss", "weighted_bce")).lower()
    if name not in {"weighted_bce", "focal"}:
        raise ValueError("training.loss must be 'weighted_bce' or 'focal'")
    scale = float(training.get("positive_weight_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("positive_weight_scale must be finite and positive")
    gamma = float(training.get("focal_gamma", 0.0 if name == "weighted_bce" else 2.0))
    if name == "weighted_bce" and gamma != 0.0:
        raise ValueError("weighted_bce requires focal_gamma=0")
    positive_weight = (negative / positive) * scale
    criterion = _WeightedFocalBCEWithLogitsLoss(
        pos_weight=positive_weight, gamma=gamma
    ).to(device)
    return criterion, {
        "loss": name,
        "positive_weight_scale": scale,
        "positive_weight": positive_weight,
        "focal_gamma": gamma,
    }


def _apply_raw_pca_transform(raw, transform):
    flat = np.asarray(raw, dtype=np.float32).reshape(len(raw), -1)
    projected = (
        flat - np.asarray(transform["input_mean"], dtype=np.float32)
    ) @ np.asarray(transform["components"], dtype=np.float32).T
    projected = (
        projected - np.asarray(transform["output_mean"], dtype=np.float32)
    ) / np.asarray(transform["output_std"], dtype=np.float32)
    projected = projected.astype(np.float32)
    if not np.isfinite(projected).all():
        raise ValueError("non-finite Raw-PCA residual features")
    return projected


def _fit_raw_pca_auxiliary(splits, n_components, random_state=0):
    """Build a train-only Raw control with the same width as the FMT input.

    PCA is fitted only on normalized training primitives.  Its output is then
    standardized with training statistics so the matched residual receives
    the same auxiliary dimensionality and the same trainable architecture as
    the FMT residual.
    """
    train, validation, test = splits
    flat_train = train[0].reshape(len(train[0]), -1)
    n_components = int(n_components)
    if n_components <= 0 or n_components > min(flat_train.shape):
        raise ValueError(
            f"raw_pca_components={n_components} is invalid for {flat_train.shape}"
        )
    pca = PCA(
        n_components=n_components, svd_solver="randomized",
        random_state=int(random_state), iterated_power=4,
    )
    projected_train = pca.fit_transform(flat_train).astype(np.float32)
    projected_mean = projected_train.mean(
        axis=0, keepdims=True, dtype=np.float64
    ).astype(np.float32)
    projected_std = projected_train.std(
        axis=0, keepdims=True, dtype=np.float64
    ).astype(np.float32)
    projected_std = np.maximum(projected_std, 1e-6)

    def transform(split, precomputed=None):
        if split is None:
            return None
        raw, _, labels = split
        if precomputed is None:
            projected = _apply_raw_pca_transform(raw, transform_state)
        else:
            projected = (
                (precomputed - projected_mean) / projected_std
            ).astype(np.float32)
        return raw, projected, labels

    transform_state = {
        "kind": "raw_pca",
        "input_mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "output_mean": projected_mean,
        "output_std": projected_std,
        "explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "random_state": int(random_state),
    }
    return (
        transform(train, projected_train), transform(validation), transform(test)
    ), transform_state


def _strong_validation_baseline(spec, dataset, seed, validation_loader,
                                raw_model, raw_checkpoint, fmt_dim, device,
                                normalization):
    """Return metric-wise stronger Raw/Raw-wide validation performance."""
    targets, probabilities = _predict(raw_model, validation_loader, device)
    metrics = [_classification_metrics(
        targets, probabilities, raw_checkpoint["threshold"]
    )]
    wide_path = (
        Path(spec["raw_checkpoint_dir"]) / f"{dataset}_raw_wide_seed{seed}.pt"
    )
    wide_checkpoint = torch.load(
        wide_path, map_location="cpu", weights_only=False
    )
    if wide_checkpoint["variant"] != "raw_wide":
        raise ValueError(
            f"expected Raw-wide checkpoint, got {wide_checkpoint['variant']}"
        )
    for key in ("raw_mean", "raw_std"):
        if not np.array_equal(
            np.asarray(wide_checkpoint["normalization"][key]),
            np.asarray(normalization[key]),
        ):
            raise RuntimeError(f"Raw-wide checkpoint disagrees on {key}")
    source = wide_checkpoint["config"]
    wide_model = PathlineBinaryClassifier3D(
        variant="raw_wide", fmt_dim=fmt_dim,
        temporal_width=source["model"]["temporal_width"],
        embedding_dim=source["model"]["embedding_dim"],
        auxiliary_dim=source["model"]["auxiliary_dim"],
    ).to(device)
    wide_model.load_state_dict(wide_checkpoint["state_dict"])
    targets_wide, probabilities_wide = _predict(
        wide_model.eval(), validation_loader, device
    )
    if not np.array_equal(targets, targets_wide):
        raise RuntimeError("Raw and Raw-wide validation targets differ")
    metrics.append(_classification_metrics(
        targets_wide, probabilities_wide, wide_checkpoint["threshold"]
    ))
    return {
        "f1": max(row["f1"] for row in metrics),
        "average_precision": max(row["average_precision"] for row in metrics),
    }


def _train_one(spec, dataset, seed, splits, stats, device, output_dir):
    _set_seed(seed)
    auxiliary_source = str(spec.get("auxiliary_source", "fmt"))
    if auxiliary_source not in {"fmt", "raw_pca"}:
        raise ValueError("auxiliary_source must be 'fmt' or 'raw_pca'")
    auxiliary_transform = None
    if auxiliary_source == "raw_pca":
        splits, auxiliary_transform = _fit_raw_pca_auxiliary(
            splits, int(spec["raw_pca_components"]),
            int(spec.get("raw_pca_random_state", 0)),
        )
    train, validation, test = splits
    variant = (
        "raw_fmt_residual" if auxiliary_source == "fmt"
        else "raw_pca_residual"
    )
    pin = device.type == "cuda"
    train_loader = _loader(train, spec["training"]["batch_size"], True, seed, pin)
    validation_loader = _loader(validation, spec["training"]["batch_size"], False, seed, pin)
    test_loader = None if test is None else _loader(
        test, spec["training"]["batch_size"], False, seed, pin
    )
    raw_checkpoint_path = (
        Path(spec["raw_checkpoint_dir"]) / f"{dataset}_raw_seed{seed}.pt"
    )
    raw_model, raw_checkpoint = _load_raw_model(
        raw_checkpoint_path, train[1].shape[1], device
    )
    for key in ("raw_mean", "raw_std"):
        if not np.array_equal(
            np.asarray(raw_checkpoint["normalization"][key]),
            np.asarray(stats[key]),
        ):
            raise RuntimeError(
                f"frozen Raw checkpoint disagrees with residual {key}"
            )
    model = PathlineFMTResidualClassifier3D(
        raw_model, fmt_dim=train[1].shape[1],
        **residual_model_kwargs(spec["model"]),
    ).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters >= int(spec["raw_wide_parameter_count"]):
        raise RuntimeError(
            f"residual model has {total_parameters} parameters, not below Raw-wide "
            f"{spec['raw_wide_parameter_count']}"
        )
    positive = float(train[2].sum())
    negative = float(len(train[2]) - positive)
    criterion, loss_metadata = _build_training_loss(
        spec["training"], positive, negative, device
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(spec["training"]["learning_rate"]),
        weight_decay=float(spec["training"]["weight_decay"]),
    )
    scheduler_name = str(spec["training"].get("scheduler", "none")).lower()
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        minimum_ratio = float(spec["training"].get("minimum_learning_rate_ratio", 0.05))
        if not 0.0 <= minimum_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(spec["training"]["max_epochs"]),
            eta_min=float(spec["training"]["learning_rate"]) * minimum_ratio,
        )
    else:
        raise ValueError("training.scheduler must be 'none' or 'cosine'")
    training_alpha = float(spec["training"].get("training_alpha", 1.0))
    if not np.isfinite(training_alpha) or training_alpha <= 0.0:
        raise ValueError("training_alpha must be finite and positive")
    if "fixed_alpha" in spec["fusion"]:
        alpha_grid = np.asarray(
            [float(spec["fusion"]["fixed_alpha"])], dtype=np.float64
        )
    else:
        alpha_grid = np.linspace(
            float(spec["fusion"]["alpha_min"]),
            float(spec["fusion"]["alpha_max"]),
            int(spec["fusion"]["alpha_steps"]),
        )
        if not np.any(alpha_grid == 0.0):
            raise ValueError("searched fusion alpha grid must contain 0")
    selection_metric = str(spec["fusion"].get(
        "selection_metric", "average_precision"
    ))
    selection_baseline = None
    if selection_metric in {"minimum_gain", "constrained_average_precision"}:
        selection_baseline = _strong_validation_baseline(
            spec, dataset, seed, validation_loader, raw_model, raw_checkpoint,
            train[1].shape[1], device, stats,
        )
    best_score, best_epoch, best_alpha, best_state = -np.inf, -1, 0.0, None
    stale = 0
    history_path = output_dir / "histories" / f"{dataset}_{variant}_seed{seed}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()
    started = time.perf_counter()
    for epoch in range(int(spec["training"]["max_epochs"])):
        model.train()
        total_loss, count = 0.0, 0
        for raw, fmt, labels in train_loader:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                raw.to(device, non_blocking=True),
                fmt.to(device, non_blocking=True),
                alpha=training_alpha,
            )
            loss = criterion(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch + 1}"
                )
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            ):
                raise FloatingPointError(
                    f"non-finite training gradient at epoch {epoch + 1}"
                )
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        val_targets, val_raw, val_residual = _predict_components(
            model, validation_loader, device
        )
        alpha, score = _select_alpha(
            val_targets, val_raw, val_residual, alpha_grid, selection_metric,
            selection_baseline,
            float(spec["fusion"].get("minimum_f1_gain", 0.02)),
        )
        selected_probabilities = _probabilities(
            val_raw, val_residual, alpha
        )
        selected_threshold = _select_f1_threshold(
            val_targets, selected_probabilities
        )
        selected_metrics = _classification_metrics(
            val_targets, selected_probabilities, selected_threshold
        )
        improved = score > best_score + float(spec["training"]["min_delta"])
        if improved:
            best_score, best_epoch, best_alpha = score, epoch + 1, alpha
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("raw_model.")
            }
            stale = 0
        else:
            stale += 1
        _append_csv(history_path, {
            "epoch": epoch + 1, "train_loss": total_loss / count,
            "selection_metric": selection_metric,
            "validation_selection_score": score,
            "validation_average_precision": selected_metrics["average_precision"],
            "validation_f1": selected_metrics["f1"],
            "validation_alpha": alpha, "is_best": int(improved),
            "stale_epochs": stale,
            "learning_rate": current_learning_rate,
            "training_alpha": training_alpha,
            **loss_metadata,
        })
        print(
            f"{dataset} {variant} seed={seed} epoch={epoch + 1:02d} "
            f"loss={total_loss / count:.5f} select_{selection_metric}={score:.5f} "
            f"val_AP={selected_metrics['average_precision']:.5f} "
            f"val_F1={selected_metrics['f1']:.5f} alpha={alpha:.3f}",
            flush=True,
        )
        if stale >= int(spec["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError("residual training produced no checkpoint")
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    val_targets, val_raw, val_residual = _predict_components(
        model, validation_loader, device
    )
    val_probabilities = _probabilities(val_raw, val_residual, best_alpha)
    threshold = _select_f1_threshold(val_targets, val_probabilities)
    val_metrics = _classification_metrics(val_targets, val_probabilities, threshold)
    if test_loader is None:
        test_metrics = None
    else:
        test_targets, test_raw, test_residual = _predict_components(
            model, test_loader, device
        )
        test_probabilities = _probabilities(test_raw, test_residual, best_alpha)
        test_metrics = _classification_metrics(
            test_targets, test_probabilities, threshold
        )
    checkpoint_path = (
        output_dir / "checkpoints" / f"{dataset}_{variant}_seed{seed}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "residual_state_dict": best_state, "raw_checkpoint": str(raw_checkpoint_path),
        "variant": variant, "dataset": dataset, "seed": seed,
        "alpha": best_alpha, "threshold": threshold, "best_epoch": best_epoch,
        "total_parameter_count": total_parameters,
        "trainable_residual_parameter_count": trainable_parameters,
        "normalization": stats, "config": spec,
        "selection_baseline": selection_baseline,
        "auxiliary_source": auxiliary_source,
        "auxiliary_transform": auxiliary_transform,
        "loss_metadata": loss_metadata,
        "scheduler": scheduler_name,
        "training_alpha": training_alpha,
    }, checkpoint_path)
    result = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "parameter_count": total_parameters,
        "trainable_residual_parameter_count": trainable_parameters,
        "raw_best_epoch": int(raw_checkpoint["best_epoch"]),
        "best_epoch": best_epoch, "validation_alpha": best_alpha,
        "validation_selection_metric": selection_metric,
        "validation_selection_baseline": selection_baseline,
        "auxiliary_source": auxiliary_source,
        "auxiliary_explained_variance_ratio": (
            "" if auxiliary_transform is None
            else auxiliary_transform["explained_variance_ratio_sum"]
        ),
        "validation_threshold": threshold,
        "training_loss": loss_metadata["loss"],
        "training_positive_weight_scale": loss_metadata["positive_weight_scale"],
        "training_positive_weight": loss_metadata["positive_weight"],
        "training_focal_gamma": loss_metadata["focal_gamma"],
        "training_scheduler": scheduler_name,
        "training_alpha": training_alpha,
        "train_positive_fraction": float(train[2].mean()),
        "validation_positive_fraction": float(validation[2].mean()),
        **{f"validation_{key}": value for key, value in val_metrics.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
    }
    if test_metrics is not None:
        result["test_positive_fraction"] = float(test[2].mean())
        result.update({f"test_{key}": value for key, value in test_metrics.items()})
    return result


def run(config_path, datasets=None, output_dir_override=None):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if datasets is not None:
        requested = list(datasets)
        unknown = sorted(set(requested) - set(spec["datasets"]))
        if unknown:
            raise ValueError(
                f"dataset override contains entries absent from config: {unknown}"
            )
        if not requested:
            raise ValueError("dataset override must not be empty")
        spec["datasets"] = requested
    if output_dir_override is not None:
        spec["output_dir"] = str(output_dir_override)
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.yaml"
    _validate_snapshot(spec, snapshot_path)
    snapshot_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    results_path = output_dir / "per_run.csv"
    auxiliary_source = str(spec.get("auxiliary_source", "fmt"))
    expected_variant = (
        "raw_fmt_residual" if auxiliary_source == "fmt"
        else "raw_pca_residual"
    )
    completed = set()
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as handle:
            completed = {
                (row["dataset"], int(row["seed"]))
                for row in csv.DictReader(handle)
                if row.get("variant", expected_variant) == expected_variant
            }
    device_name = spec["training"].get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    print(f"device={device}", flush=True)
    for dataset in spec["datasets"]:
        test_enabled = bool(spec.get("evaluation", {}).get("test_enabled", True))
        required_ordinals = set(spec["split"]["train_ordinals"]) | set(
            spec["split"]["validation_ordinals"]
        )
        if test_enabled:
            required_ordinals |= set(spec["split"]["test_ordinals"])
        records = _load_dataset(
            Path(spec["source_cache_root"]) / dataset,
            Path(spec["label_cache_root"]) / dataset,
            spec["sampled_steps"], spec["fmt_subset"], required_ordinals,
            spec.get("fmt_gram_num_freq", 6),
            spec.get("expected_slices", 10),
        )
        train = _stack_split(records, spec["split"]["train_ordinals"])
        validation = _stack_split(records, spec["split"]["validation_ordinals"])
        test = _stack_split(
            records, spec["split"]["test_ordinals"]
        ) if test_enabled else None
        train, validation, test, stats = _normalize_train_only(train, validation, test)
        for seed in spec["training"]["seeds"]:
            key = (dataset, int(seed))
            if key in completed:
                print(f"cached result: {key}", flush=True)
                continue
            row = _train_one(
                spec, dataset, int(seed), (train, validation, test), stats,
                device, output_dir,
            )
            _append_csv(results_path, row)
            completed.add(key)
            if test_enabled:
                outcome = (
                    f"test F1={row['test_f1']:.5f}, "
                    f"AP={row['test_average_precision']:.5f}"
                )
            else:
                outcome = (
                    f"validation F1={row['validation_f1']:.5f}, "
                    f"AP={row['validation_average_precision']:.5f}"
                )
            print(
                f"DONE {key}: alpha={row['validation_alpha']:.3f} {outcome}",
                flush=True,
            )
    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset", action="append", dest="datasets",
        help="Run only this configured dataset; repeat to select several.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir_override",
        help="Write this shard to an isolated output directory.",
    )
    args = parser.parse_args()
    run(args.config, args.datasets, args.output_dir_override)
