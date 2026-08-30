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


def _residual_gate_parameters(model_spec=None):
    """Validate the label-free frozen-Raw residual gate configuration."""
    model_spec = {} if model_spec is None else dict(model_spec)
    kind = str(model_spec.get("residual_gate", "none")).lower()
    if kind not in {"none", "raw_uncertainty"}:
        raise ValueError(
            "model.residual_gate must be 'none' or 'raw_uncertainty'"
        )
    temperature = float(model_spec.get("residual_gate_temperature", 1.0))
    floor = float(model_spec.get("residual_gate_floor", 0.0))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "model.residual_gate_temperature must be finite and positive"
        )
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("model.residual_gate_floor must be in [0, 1]")
    if kind == "none" and (
        not np.isclose(temperature, 1.0) or not np.isclose(floor, 0.0)
    ):
        raise ValueError(
            "the none residual gate requires temperature=1 and floor=0"
        )
    return kind, temperature, floor


def _residual_gate_numpy(raw_logits, model_spec=None):
    """Return a deterministic gate derived only from frozen Raw logits."""
    kind, temperature, floor = _residual_gate_parameters(model_spec)
    raw_logits = np.asarray(raw_logits)
    if kind == "none":
        return np.ones_like(raw_logits)
    scaled = np.clip(raw_logits / temperature, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-scaled))
    uncertainty = 4.0 * probability * (1.0 - probability)
    return floor + (1.0 - floor) * uncertainty


def _residual_gate_torch(raw_logits, model_spec=None):
    """Torch equivalent of :func:`_residual_gate_numpy` for training."""
    kind, temperature, floor = _residual_gate_parameters(model_spec)
    if kind == "none":
        return torch.ones_like(raw_logits)
    probability = torch.sigmoid(raw_logits / temperature)
    uncertainty = 4.0 * probability * (1.0 - probability)
    return floor + (1.0 - floor) * uncertainty


def _gradient_clip_norm(training_spec=None):
    """Return a validated global gradient-norm limit, or ``None`` to disable."""
    training_spec = {} if training_spec is None else dict(training_spec)
    value = training_spec.get("gradient_clip_norm")
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    return value


def _clip_trainable_gradients(model, maximum_norm):
    """Clip the global norm of existing trainable gradients.

    The returned value is the norm before clipping. A disabled limit is a
    strict no-op so that the control reproduces the historical optimizer step.
    """
    if maximum_norm is None:
        return None
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        raise RuntimeError("gradient clipping found no trainable gradients")
    norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=float(maximum_norm),
        error_if_nonfinite=True,
    )
    return float(norm.detach().cpu())


def _gated_residual_logits(raw_logits, residual_logits, model_spec=None):
    return residual_logits * _residual_gate_numpy(raw_logits, model_spec)


def _select_alpha(targets, raw_logits, residual_logits, alpha_grid,
                  objective="average_precision", baseline_metrics=None,
                  minimum_f1_gain=0.02, model_spec=None):
    effective_residual = _gated_residual_logits(
        raw_logits, residual_logits, model_spec
    )
    if objective == "average_precision":
        scores = np.asarray([
            average_precision_score(
                targets, raw_logits + alpha * effective_residual
            )
            for alpha in alpha_grid
        ])
    elif objective == "f1":
        values = []
        for alpha in alpha_grid:
            probabilities = _probabilities(
                raw_logits, residual_logits, alpha, model_spec
            )
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
            probabilities = _probabilities(
                raw_logits, residual_logits, alpha, model_spec
            )
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
            probabilities = _probabilities(
                raw_logits, residual_logits, alpha, model_spec
            )
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


def _probabilities(raw_logits, residual_logits, alpha, model_spec=None):
    logits = raw_logits + float(alpha) * _gated_residual_logits(
        raw_logits, residual_logits, model_spec
    )
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

    def forward(self, logits, targets, sample_weights=None):
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
        if sample_weights is not None:
            sample_weights = torch.as_tensor(
                sample_weights, dtype=loss.dtype, device=loss.device
            )
            if sample_weights.shape != loss.shape:
                raise ValueError("sample_weights must match logits shape")
            if (
                not bool(torch.isfinite(sample_weights).all())
                or bool(torch.any(sample_weights <= 0.0))
            ):
                raise ValueError("sample_weights must be finite and positive")
            loss = loss * sample_weights
        return loss.mean()


def _build_training_loss(training: dict, positive: float, negative: float,
                         device: torch.device,
                         sampled_positive_fraction=None):
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
    if sampled_positive_fraction is None:
        positive_weight = (negative / positive) * scale
    else:
        sampled_positive_fraction = float(sampled_positive_fraction)
        if (
            not np.isfinite(sampled_positive_fraction)
            or not 0.0 < sampled_positive_fraction < 1.0
        ):
            raise ValueError(
                "sampled_positive_fraction must be finite and in (0, 1)"
            )
        # Preserve the same class-balanced expected objective after changing
        # the class prevalence seen by the sampler.  Under sampled prevalence
        # q, q * pos_weight == (1-q) when scale=1.
        positive_weight = (
            (1.0 - sampled_positive_fraction) / sampled_positive_fraction
        ) * scale
    hardness_scale = float(training.get("raw_hardness_scale", 0.0))
    hardness_power = float(training.get("raw_hardness_power", 1.0))
    hardness_temperature = float(
        training.get("raw_hardness_temperature", 1.0)
    )
    error_boost = float(training.get("raw_error_boost", 0.0))
    margin_loss_weight = float(training.get("raw_margin_loss_weight", 0.0))
    margin_target = float(training.get("raw_margin_target", 1.0))
    margin_huber_delta = float(
        training.get("raw_margin_huber_delta", 1.0)
    )
    margin_max_correction = float(
        training.get("raw_margin_max_correction", 20.0)
    )
    ranking_loss_weight = float(
        training.get("pairwise_ranking_loss_weight", 0.0)
    )
    ranking_margin = float(training.get("pairwise_ranking_margin", 0.0))
    ranking_temperature = float(
        training.get("pairwise_ranking_temperature", 1.0)
    )
    overlap_loss_weight = float(
        training.get("overlap_loss_weight", 0.0)
    )
    overlap_false_positive_weight = float(
        training.get("overlap_false_positive_weight", 0.5)
    )
    overlap_false_negative_weight = float(
        training.get("overlap_false_negative_weight", 0.5)
    )
    overlap_smoothing = float(training.get("overlap_smoothing", 1.0))
    contrastive_loss_weight = float(
        training.get("supervised_contrastive_loss_weight", 0.0)
    )
    contrastive_temperature = float(
        training.get("supervised_contrastive_temperature", 0.1)
    )
    auxiliary_supervision_loss_weight = float(
        training.get("auxiliary_supervision_loss_weight", 0.0)
    )
    if not np.isfinite(hardness_scale) or hardness_scale < 0.0:
        raise ValueError("raw_hardness_scale must be finite and non-negative")
    if not np.isfinite(hardness_power) or hardness_power <= 0.0:
        raise ValueError("raw_hardness_power must be finite and positive")
    if not np.isfinite(hardness_temperature) or hardness_temperature <= 0.0:
        raise ValueError(
            "raw_hardness_temperature must be finite and positive"
        )
    if not np.isfinite(error_boost) or error_boost < 0.0:
        raise ValueError("raw_error_boost must be finite and non-negative")
    if not np.isfinite(margin_loss_weight) or margin_loss_weight < 0.0:
        raise ValueError(
            "raw_margin_loss_weight must be finite and non-negative"
        )
    if not np.isfinite(margin_target) or margin_target <= 0.0:
        raise ValueError("raw_margin_target must be finite and positive")
    if not np.isfinite(margin_huber_delta) or margin_huber_delta <= 0.0:
        raise ValueError(
            "raw_margin_huber_delta must be finite and positive"
        )
    if (
        not np.isfinite(margin_max_correction)
        or margin_max_correction <= 0.0
    ):
        raise ValueError(
            "raw_margin_max_correction must be finite and positive"
        )
    if not np.isfinite(ranking_loss_weight) or ranking_loss_weight < 0.0:
        raise ValueError(
            "pairwise_ranking_loss_weight must be finite and non-negative"
        )
    if not np.isfinite(ranking_margin) or ranking_margin < 0.0:
        raise ValueError(
            "pairwise_ranking_margin must be finite and non-negative"
        )
    if not np.isfinite(ranking_temperature) or ranking_temperature <= 0.0:
        raise ValueError(
            "pairwise_ranking_temperature must be finite and positive"
        )
    if not np.isfinite(overlap_loss_weight) or overlap_loss_weight < 0.0:
        raise ValueError(
            "overlap_loss_weight must be finite and non-negative"
        )
    if (
        not np.isfinite(overlap_false_positive_weight)
        or overlap_false_positive_weight < 0.0
    ):
        raise ValueError(
            "overlap_false_positive_weight must be finite and non-negative"
        )
    if (
        not np.isfinite(overlap_false_negative_weight)
        or overlap_false_negative_weight < 0.0
    ):
        raise ValueError(
            "overlap_false_negative_weight must be finite and non-negative"
        )
    if overlap_false_positive_weight + overlap_false_negative_weight <= 0.0:
        raise ValueError(
            "overlap false-positive and false-negative weights cannot both "
            "be zero"
        )
    if not np.isfinite(overlap_smoothing) or overlap_smoothing <= 0.0:
        raise ValueError("overlap_smoothing must be finite and positive")
    if (
        not np.isfinite(contrastive_loss_weight)
        or contrastive_loss_weight < 0.0
    ):
        raise ValueError(
            "supervised_contrastive_loss_weight must be finite and "
            "non-negative"
        )
    if (
        not np.isfinite(contrastive_temperature)
        or contrastive_temperature <= 0.0
    ):
        raise ValueError(
            "supervised_contrastive_temperature must be finite and positive"
        )
    if (
        not np.isfinite(auxiliary_supervision_loss_weight)
        or auxiliary_supervision_loss_weight < 0.0
    ):
        raise ValueError(
            "auxiliary_supervision_loss_weight must be finite and "
            "non-negative"
        )
    criterion = _WeightedFocalBCEWithLogitsLoss(
        pos_weight=positive_weight, gamma=gamma
    ).to(device)
    return criterion, {
        "loss": name,
        "positive_weight_scale": scale,
        "positive_weight": positive_weight,
        "sampled_positive_fraction": sampled_positive_fraction,
        "focal_gamma": gamma,
        "raw_hardness_scale": hardness_scale,
        "raw_hardness_power": hardness_power,
        "raw_hardness_temperature": hardness_temperature,
        "raw_error_boost": error_boost,
        "raw_margin_loss_weight": margin_loss_weight,
        "raw_margin_target": margin_target,
        "raw_margin_huber_delta": margin_huber_delta,
        "raw_margin_max_correction": margin_max_correction,
        "pairwise_ranking_loss_weight": ranking_loss_weight,
        "pairwise_ranking_margin": ranking_margin,
        "pairwise_ranking_temperature": ranking_temperature,
        "overlap_loss_weight": overlap_loss_weight,
        "overlap_false_positive_weight": overlap_false_positive_weight,
        "overlap_false_negative_weight": overlap_false_negative_weight,
        "overlap_smoothing": overlap_smoothing,
        "supervised_contrastive_loss_weight": contrastive_loss_weight,
        "supervised_contrastive_temperature": contrastive_temperature,
        "auxiliary_supervision_loss_weight": (
            auxiliary_supervision_loss_weight
        ),
    }


def _raw_hardness_weights(raw_logits, targets, training):
    """Return mean-one weights derived only from the frozen Raw arm.

    ``difficulty = 1 - p_t(raw)`` is detached from autograd.  Both FMT and
    train-only Raw-PCA residual arms therefore receive the exact same weights
    for the same seed and mini-batch.
    """
    scale = float(training.get("raw_hardness_scale", 0.0))
    power = float(training.get("raw_hardness_power", 1.0))
    temperature = float(training.get("raw_hardness_temperature", 1.0))
    error_boost = float(training.get("raw_error_boost", 0.0))
    if scale == 0.0 and error_boost == 0.0:
        return None
    with torch.no_grad():
        signed_raw = (2.0 * targets - 1.0) * raw_logits
        difficulty = torch.sigmoid(-signed_raw / temperature).pow(power)
        weights = 1.0 + scale * difficulty
        if error_boost > 0.0:
            weights = weights + error_boost * (signed_raw < 0.0).to(weights)
        weights = weights / weights.mean().clamp_min(
            torch.finfo(weights.dtype).tiny
        )
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights <= 0.0)):
        raise FloatingPointError("non-finite Raw-hardness sample weights")
    return weights.detach()


def _raw_margin_residual_targets(raw_logits, targets, training,
                                 training_alpha):
    """Build the correction required to reach a signed Raw-logit margin.

    The target is zero where the frozen Raw classifier already exceeds the
    requested correct-class margin.  Elsewhere it asks the residual branch for
    exactly the missing signed logit correction, capped for numerical safety.
    It depends only on frozen Raw logits and training labels, so paired FMT and
    Raw-PCA arms receive an identical target for every mini-batch.
    """
    margin = float(training.get("raw_margin_target", 1.0))
    maximum = float(training.get("raw_margin_max_correction", 20.0))
    alpha = float(training_alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("training_alpha must be finite and positive")
    with torch.no_grad():
        sign = 2.0 * targets - 1.0
        signed_raw = sign * raw_logits
        shortfall = torch.clamp(margin - signed_raw, min=0.0, max=maximum)
        correction = sign * shortfall / alpha
    if not bool(torch.isfinite(correction).all()):
        raise FloatingPointError("non-finite Raw-margin residual targets")
    return correction.detach()


def _raw_margin_residual_loss(raw_logits, residual_logits, targets, training,
                              training_alpha, sample_weights=None):
    """Return the optional paired Smooth-L1 residual-correction objective."""
    weight = float(training.get("raw_margin_loss_weight", 0.0))
    if weight == 0.0:
        return residual_logits.sum() * 0.0
    target = _raw_margin_residual_targets(
        raw_logits.detach(), targets, training, training_alpha
    )
    delta = float(training.get("raw_margin_huber_delta", 1.0))
    loss = F.smooth_l1_loss(
        residual_logits, target, reduction="none", beta=delta
    )
    if sample_weights is not None:
        sample_weights = torch.as_tensor(
            sample_weights, dtype=loss.dtype, device=loss.device
        )
        if sample_weights.shape != loss.shape:
            raise ValueError("sample_weights must match residual logits shape")
        loss = loss * sample_weights
    return weight * loss.mean()


def _paired_ranking_loss(logits, targets, training, sample_weights=None):
    """Encourage every positive logit to rank above every negative logit.

    The smooth pairwise objective is applied identically to the FMT and
    train-only Raw-PCA arms.  A zero weight is an exact no-op.  Mini-batches
    containing only one class also contribute zero instead of making the
    training result depend on an arbitrary fallback pair.
    """
    weight = float(training.get("pairwise_ranking_loss_weight", 0.0))
    if weight == 0.0:
        return logits.sum() * 0.0
    positive_mask = targets >= 0.5
    negative_mask = ~positive_mask
    if not bool(positive_mask.any()) or not bool(negative_mask.any()):
        return logits.sum() * 0.0
    margin = float(training.get("pairwise_ranking_margin", 0.0))
    temperature = float(training.get("pairwise_ranking_temperature", 1.0))
    positive_logits = logits[positive_mask]
    negative_logits = logits[negative_mask]
    gaps = positive_logits[:, None] - negative_logits[None, :]
    pair_loss = temperature * F.softplus(
        (margin - gaps) / temperature
    )
    if sample_weights is not None:
        sample_weights = torch.as_tensor(
            sample_weights, dtype=logits.dtype, device=logits.device
        )
        if sample_weights.shape != logits.shape:
            raise ValueError("sample_weights must match logits shape")
        pair_weights = torch.sqrt(
            sample_weights[positive_mask, None]
            * sample_weights[None, negative_mask]
        )
        pair_weights = pair_weights / pair_weights.mean().clamp_min(
            torch.finfo(pair_weights.dtype).tiny
        )
        pair_loss = pair_loss * pair_weights
    return weight * pair_loss.mean()


def _soft_tversky_loss(logits, targets, training, sample_weights=None):
    """Return a differentiable mini-batch overlap loss.

    Equal false-positive and false-negative weights (0.5, 0.5) give the
    standard soft Dice loss. Asymmetric weights give the Tversky loss and
    test whether scarce IVD-positive samples benefit from a stronger
    false-negative penalty. Both Task3 arms receive the exact same recipe.
    """
    weight = float(training.get("overlap_loss_weight", 0.0))
    if weight == 0.0:
        return logits.sum() * 0.0
    if logits.shape != targets.shape:
        raise ValueError("targets must match logits shape")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite overlap logits")
    if (
        not bool(torch.isfinite(targets).all())
        or bool(torch.any(targets < 0.0))
        or bool(torch.any(targets > 1.0))
    ):
        raise ValueError("overlap targets must be finite and in [0, 1]")
    false_positive_weight = float(
        training.get("overlap_false_positive_weight", 0.5)
    )
    false_negative_weight = float(
        training.get("overlap_false_negative_weight", 0.5)
    )
    smoothing = float(training.get("overlap_smoothing", 1.0))
    probabilities = torch.sigmoid(logits)
    if sample_weights is None:
        weights = torch.ones_like(probabilities)
    else:
        weights = torch.as_tensor(
            sample_weights,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        if weights.shape != probabilities.shape:
            raise ValueError("sample_weights must match logits shape")
        if (
            not bool(torch.isfinite(weights).all())
            or bool(torch.any(weights <= 0.0))
        ):
            raise ValueError("sample_weights must be finite and positive")
        weights = weights / weights.mean().clamp_min(
            torch.finfo(weights.dtype).tiny
        )
    true_positive = torch.sum(weights * probabilities * targets)
    false_positive = torch.sum(weights * probabilities * (1.0 - targets))
    false_negative = torch.sum(weights * (1.0 - probabilities) * targets)
    score = (true_positive + smoothing) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + smoothing
    )
    loss = weight * (1.0 - score)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite overlap loss")
    return loss


def _supervised_contrastive_loss(embeddings, targets, training,
                                 sample_weights=None):
    """Separate binary IVD classes in the trainable auxiliary embedding.

    This is the supervised contrastive objective of averaging each anchor's
    log-probability over same-class peers. Self-pairs are excluded. Batches
    without both classes or without a same-class peer return an exact zero,
    so the loss never invents an arbitrary fallback pair.
    """
    weight = float(training.get("supervised_contrastive_loss_weight", 0.0))
    if weight == 0.0:
        return embeddings.sum() * 0.0
    if embeddings.ndim != 2:
        raise ValueError("contrastive embeddings must be [batch, features]")
    if targets.ndim != 1 or len(targets) != len(embeddings):
        raise ValueError("contrastive targets must be [batch]")
    if len(targets) < 2:
        return embeddings.sum() * 0.0
    if (
        not bool(torch.isfinite(embeddings).all())
        or not bool(torch.isfinite(targets).all())
    ):
        raise FloatingPointError("non-finite supervised contrastive input")
    if bool(torch.any(targets < 0.0)) or bool(torch.any(targets > 1.0)):
        raise ValueError("contrastive targets must be in [0, 1]")
    binary_targets = targets >= 0.5
    if not bool(binary_targets.any()) or bool(binary_targets.all()):
        return embeddings.sum() * 0.0
    temperature = float(
        training.get("supervised_contrastive_temperature", 0.1)
    )
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarities = normalized @ normalized.transpose(0, 1)
    similarities = similarities / temperature
    similarities = similarities - similarities.max(
        dim=1, keepdim=True
    ).values.detach()
    off_diagonal = ~torch.eye(
        len(targets), dtype=torch.bool, device=embeddings.device
    )
    positive_pairs = (
        binary_targets[:, None] == binary_targets[None, :]
    ) & off_diagonal
    positive_counts = positive_pairs.sum(dim=1)
    valid_anchors = positive_counts > 0
    if not bool(valid_anchors.any()):
        return embeddings.sum() * 0.0
    denominator = (
        torch.exp(similarities) * off_diagonal.to(similarities.dtype)
    ).sum(dim=1).clamp_min(torch.finfo(similarities.dtype).tiny)
    log_probabilities = similarities - torch.log(denominator[:, None])
    anchor_losses = -(
        log_probabilities * positive_pairs.to(log_probabilities.dtype)
    ).sum(dim=1) / positive_counts.clamp_min(1).to(log_probabilities.dtype)
    anchor_losses = anchor_losses[valid_anchors]
    if sample_weights is not None:
        sample_weights = torch.as_tensor(
            sample_weights,
            dtype=anchor_losses.dtype,
            device=anchor_losses.device,
        )
        if sample_weights.shape != targets.shape:
            raise ValueError("sample_weights must match contrastive targets")
        if (
            not bool(torch.isfinite(sample_weights).all())
            or bool(torch.any(sample_weights <= 0.0))
        ):
            raise ValueError("sample_weights must be finite and positive")
        anchor_weights = sample_weights[valid_anchors]
        anchor_weights = anchor_weights / anchor_weights.mean().clamp_min(
            torch.finfo(anchor_weights.dtype).tiny
        )
        anchor_losses = anchor_losses * anchor_weights
    loss = weight * anchor_losses.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite supervised contrastive loss")
    return loss


def _auxiliary_supervision_loss(model, auxiliary_embeddings, targets,
                                training, criterion, sample_weights=None):
    """Directly supervise the projected auxiliary representation.

    This loss is paired: FMT and train-only Raw-PCA receive the same head,
    class weights, mini-batches, labels, and scalar weight.  The auxiliary
    classifier is training-only and does not enter the fused inference logit.
    """
    weight = float(training.get("auxiliary_supervision_loss_weight", 0.0))
    if weight == 0.0:
        return auxiliary_embeddings.sum() * 0.0
    auxiliary_logits = model.auxiliary_classification_logits(
        auxiliary_embeddings
    )
    if not bool(torch.isfinite(auxiliary_logits).all()):
        raise FloatingPointError("non-finite auxiliary classification logits")
    return weight * criterion(
        auxiliary_logits, targets, sample_weights=sample_weights
    )


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


def _optimizer_betas(training=None):
    """Return validated Adam-family momentum coefficients and override state."""
    training = {} if training is None else dict(training)
    beta1_value = training.get("optimizer_beta1")
    beta2_value = training.get("optimizer_beta2")
    overridden = beta1_value is not None or beta2_value is not None
    beta1 = 0.9 if beta1_value is None else float(beta1_value)
    beta2 = 0.999 if beta2_value is None else float(beta2_value)
    for name, value in (("optimizer_beta1", beta1), ("optimizer_beta2", beta2)):
        if not np.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1)")
    return (beta1, beta2), overridden


def _batch_size(training=None):
    """Return a positive integral batch size without silent truncation."""
    training = {} if training is None else dict(training)
    value = training.get("batch_size")
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError("batch_size must be a positive integer")
    numeric = float(value)
    if (
        not np.isfinite(numeric)
        or numeric < 1.0
        or numeric != float(int(numeric))
    ):
        raise ValueError("batch_size must be a positive integer")
    return int(numeric)


def _build_optimizer(training: dict, parameters):
    """Build the registered optimizer without changing the paired arm budget."""
    name = str(training.get("optimizer", "adamw")).lower()
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    amsgrad = bool(training.get("optimizer_amsgrad", False))
    parameters = list(parameters)
    if not parameters:
        raise ValueError("optimizer requires at least one trainable parameter")
    common = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    betas, betas_overridden = _optimizer_betas(training)
    if betas_overridden:
        common["betas"] = betas
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, amsgrad=amsgrad, **common
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            parameters, amsgrad=amsgrad, **common
        )
    elif name == "radam":
        if amsgrad:
            raise ValueError("RAdam does not support optimizer_amsgrad")
        optimizer = torch.optim.RAdam(parameters, **common)
    elif name == "nadam":
        if amsgrad:
            raise ValueError("NAdam does not support optimizer_amsgrad")
        optimizer = torch.optim.NAdam(parameters, **common)
    else:
        raise ValueError(
            "training.optimizer must be one of "
            "'adamw', 'adam', 'radam', or 'nadam'"
        )
    return optimizer, name, amsgrad


def _warmup_parameters(training=None):
    """Validate epoch-level linear warmup while preserving a strict no-op."""
    training = {} if training is None else dict(training)
    epochs_value = training.get("warmup_epochs")
    start_value = training.get("warmup_start_ratio")
    epochs = 0 if epochs_value is None else int(epochs_value)
    start_ratio = 0.1 if start_value is None else float(start_value)
    if epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    max_epochs = int(training.get("max_epochs", max(epochs, 1)))
    if epochs >= max_epochs and epochs > 0:
        raise ValueError("warmup_epochs must be smaller than max_epochs")
    if not np.isfinite(start_ratio) or not 0.0 < start_ratio <= 1.0:
        raise ValueError("warmup_start_ratio must be finite and in (0, 1]")
    scheduler_name = str(training.get("scheduler", "none")).lower()
    if epochs > 0 and scheduler_name != "none":
        raise ValueError("linear warmup currently requires scheduler='none'")
    overridden = epochs_value is not None or start_value is not None
    return epochs, start_ratio, overridden


def _build_scheduler(training: dict, optimizer):
    """Build the registered epoch scheduler with optional linear warmup."""
    warmup_epochs, warmup_start_ratio, _ = _warmup_parameters(training)
    scheduler_name = str(training.get("scheduler", "none")).lower()
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (
                1.0 if epoch >= warmup_epochs else
                warmup_start_ratio
                + (1.0 - warmup_start_ratio) * epoch / warmup_epochs
            ),
        )
        return scheduler, "linear_warmup_constant", warmup_epochs, warmup_start_ratio
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        minimum_ratio = float(training.get("minimum_learning_rate_ratio", 0.05))
        if not 0.0 <= minimum_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["max_epochs"]),
            eta_min=float(training["learning_rate"]) * minimum_ratio,
        )
    else:
        raise ValueError("training.scheduler must be 'none' or 'cosine'")
    return scheduler, scheduler_name, warmup_epochs, warmup_start_ratio


def _train_one(spec, dataset, seed, splits, stats, device, output_dir):
    _set_seed(seed)
    gate_kind, gate_temperature, gate_floor = _residual_gate_parameters(
        spec["model"]
    )
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
    batch_size = _batch_size(spec["training"])
    requested_sampled_positive_fraction = spec["training"].get(
        "minibatch_positive_fraction"
    )
    train_loader = _loader(
        train, batch_size, True, seed, pin,
        positive_fraction=requested_sampled_positive_fraction,
    )
    validation_loader = _loader(validation, batch_size, False, seed, pin)
    test_loader = None if test is None else _loader(
        test, batch_size, False, seed, pin
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
    batch_sampler = train_loader.batch_sampler
    sampled_positive_fraction = getattr(
        batch_sampler, "actual_positive_fraction", None
    )
    criterion, loss_metadata = _build_training_loss(
        spec["training"], positive, negative, device,
        sampled_positive_fraction=sampled_positive_fraction,
    )
    auxiliary_supervision_enabled = (
        float(loss_metadata["auxiliary_supervision_loss_weight"]) > 0.0
    )
    if auxiliary_supervision_enabled != (model.auxiliary_classifier is not None):
        raise ValueError(
            "auxiliary supervision requires a configured auxiliary classifier, "
            "and an auxiliary classifier requires a positive supervision weight"
        )
    optimizer, optimizer_name, optimizer_amsgrad = _build_optimizer(
        spec["training"],
        (
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    optimizer_betas, _ = _optimizer_betas(spec["training"])
    scheduler, scheduler_name, warmup_epochs, warmup_start_ratio = (
        _build_scheduler(spec["training"], optimizer)
    )
    training_alpha = float(spec["training"].get("training_alpha", 1.0))
    if not np.isfinite(training_alpha) or training_alpha <= 0.0:
        raise ValueError("training_alpha must be finite and positive")
    gradient_clip_norm = _gradient_clip_norm(spec["training"])
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
        maximum_preclip_gradient_norm = None
        for raw, fmt, labels in train_loader:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            raw_logits, residual_logits, auxiliary_embeddings = (
                model.forward_components(
                    raw.to(device, non_blocking=True),
                    fmt.to(device, non_blocking=True),
                    return_auxiliary=True,
                )
            )
            effective_residual_logits = residual_logits * _residual_gate_torch(
                raw_logits.detach(), spec["model"]
            )
            logits = raw_logits + training_alpha * effective_residual_logits
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    f"non-finite training logits at epoch {epoch + 1}"
                )
            sample_weights = _raw_hardness_weights(
                raw_logits.detach(), labels, spec["training"]
            )
            classification_loss = criterion(
                logits, labels, sample_weights=sample_weights
            )
            correction_loss = _raw_margin_residual_loss(
                raw_logits,
                effective_residual_logits,
                labels,
                spec["training"],
                training_alpha,
                sample_weights=sample_weights,
            )
            ranking_loss = _paired_ranking_loss(
                logits,
                labels,
                spec["training"],
                sample_weights=sample_weights,
            )
            overlap_loss = _soft_tversky_loss(
                logits,
                labels,
                spec["training"],
                sample_weights=sample_weights,
            )
            contrastive_loss = _supervised_contrastive_loss(
                auxiliary_embeddings,
                labels,
                spec["training"],
                sample_weights=sample_weights,
            )
            auxiliary_supervision_loss = _auxiliary_supervision_loss(
                model,
                auxiliary_embeddings,
                labels,
                spec["training"],
                criterion,
                sample_weights=sample_weights,
            )
            loss = (
                classification_loss
                + correction_loss
                + ranking_loss
                + overlap_loss
                + contrastive_loss
                + auxiliary_supervision_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch + 1}"
                )
            loss.backward()
            invalid_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            if invalid_gradients:
                raise FloatingPointError(
                    f"non-finite training gradient at epoch {epoch + 1}; "
                    f"parameters={','.join(invalid_gradients[:8])}"
                )
            preclip_gradient_norm = _clip_trainable_gradients(
                model, gradient_clip_norm
            )
            if preclip_gradient_norm is not None:
                maximum_preclip_gradient_norm = max(
                    preclip_gradient_norm,
                    maximum_preclip_gradient_norm
                    if maximum_preclip_gradient_norm is not None else 0.0,
                )
            optimizer.step()
            invalid_parameters = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and not bool(torch.isfinite(parameter).all())
            ]
            if invalid_parameters:
                raise FloatingPointError(
                    f"non-finite training parameter after optimizer step at "
                    f"epoch {epoch + 1}; "
                    f"parameters={','.join(invalid_parameters[:8])}"
                )
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
            spec["model"],
        )
        selected_probabilities = _probabilities(
            val_raw, val_residual, alpha, spec["model"]
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
            "gradient_clip_norm": (
                "" if gradient_clip_norm is None else gradient_clip_norm
            ),
            "maximum_preclip_gradient_norm": (
                "" if maximum_preclip_gradient_norm is None
                else maximum_preclip_gradient_norm
            ),
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
    val_probabilities = _probabilities(
        val_raw, val_residual, best_alpha, spec["model"]
    )
    threshold = _select_f1_threshold(val_targets, val_probabilities)
    val_metrics = _classification_metrics(val_targets, val_probabilities, threshold)
    if test_loader is None:
        test_metrics = None
    else:
        test_targets, test_raw, test_residual = _predict_components(
            model, test_loader, device
        )
        test_probabilities = _probabilities(
            test_raw, test_residual, best_alpha, spec["model"]
        )
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
        "optimizer": optimizer_name,
        "optimizer_amsgrad": optimizer_amsgrad,
        "optimizer_betas": optimizer_betas,
        "scheduler": scheduler_name,
        "warmup_epochs": warmup_epochs,
        "warmup_start_ratio": warmup_start_ratio,
        "training_alpha": training_alpha,
        "gradient_clip_norm": gradient_clip_norm,
        "residual_gate": {
            "kind": gate_kind,
            "temperature": gate_temperature,
            "floor": gate_floor,
        },
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
        "residual_gate_kind": gate_kind,
        "residual_gate_temperature": gate_temperature,
        "residual_gate_floor": gate_floor,
        "training_positive_weight_scale": loss_metadata["positive_weight_scale"],
        "training_positive_weight": loss_metadata["positive_weight"],
        "training_requested_minibatch_positive_fraction": (
            "" if requested_sampled_positive_fraction is None
            else float(requested_sampled_positive_fraction)
        ),
        "training_actual_minibatch_positive_fraction": (
            "" if sampled_positive_fraction is None
            else float(sampled_positive_fraction)
        ),
        "training_focal_gamma": loss_metadata["focal_gamma"],
        "training_raw_hardness_scale": loss_metadata["raw_hardness_scale"],
        "training_raw_hardness_power": loss_metadata["raw_hardness_power"],
        "training_raw_hardness_temperature": loss_metadata[
            "raw_hardness_temperature"
        ],
        "training_raw_error_boost": loss_metadata["raw_error_boost"],
        "training_raw_margin_loss_weight": loss_metadata[
            "raw_margin_loss_weight"
        ],
        "training_raw_margin_target": loss_metadata["raw_margin_target"],
        "training_raw_margin_huber_delta": loss_metadata[
            "raw_margin_huber_delta"
        ],
        "training_raw_margin_max_correction": loss_metadata[
            "raw_margin_max_correction"
        ],
        "training_pairwise_ranking_loss_weight": loss_metadata[
            "pairwise_ranking_loss_weight"
        ],
        "training_pairwise_ranking_margin": loss_metadata[
            "pairwise_ranking_margin"
        ],
        "training_pairwise_ranking_temperature": loss_metadata[
            "pairwise_ranking_temperature"
        ],
        "training_overlap_loss_weight": loss_metadata[
            "overlap_loss_weight"
        ],
        "training_overlap_false_positive_weight": loss_metadata[
            "overlap_false_positive_weight"
        ],
        "training_overlap_false_negative_weight": loss_metadata[
            "overlap_false_negative_weight"
        ],
        "training_overlap_smoothing": loss_metadata["overlap_smoothing"],
        "training_supervised_contrastive_loss_weight": loss_metadata[
            "supervised_contrastive_loss_weight"
        ],
        "training_supervised_contrastive_temperature": loss_metadata[
            "supervised_contrastive_temperature"
        ],
        "training_auxiliary_supervision_loss_weight": loss_metadata[
            "auxiliary_supervision_loss_weight"
        ],
        "training_optimizer": optimizer_name,
        "training_optimizer_amsgrad": optimizer_amsgrad,
        "training_optimizer_beta1": optimizer_betas[0],
        "training_optimizer_beta2": optimizer_betas[1],
        "training_scheduler": scheduler_name,
        "training_warmup_epochs": warmup_epochs,
        "training_warmup_start_ratio": warmup_start_ratio,
        "training_alpha": training_alpha,
        "gradient_clip_norm": (
            "" if gradient_clip_norm is None else gradient_clip_norm
        ),
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
