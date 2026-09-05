"""Memorize channel Task4-b labels, then evaluate once on TBL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import torch
from torch import nn
import yaml

from experiments.Train_Task4B_FourClassClassifier_1_1 import _metrics
from experiments.Verify_Task4B_FullVolumeMemorization_1_1 import (
    _fit_metrics,
    _is_exact,
    _predict_logits,
    _set_seed,
    _write_history_row,
    epoch_permutation,
    permutation_certificate,
)
from FMT_Utils.Task4A_PerVortexClustering_3D import fmt_base_feature_width
from FMT_Utils.Task4B_Classifier_3D import (
    PathlineMulticlassClassifier3D,
    trainable_parameter_count,
)
from FMT_Utils.Task4B_CrossFlow_3D import (
    apply_source_normalization,
    dimensionless_primitive_features,
    fit_source_normalization,
)
from FMT_Utils.Task4B_ProxyLabels_3D import CLASS_NAMES


DEFAULT_CONFIG = Path("config/mainExp_Task4B_ChannelToTBL_2.3.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(spec: dict) -> tuple[dict, dict]:
    path = Path(spec["source"]["cache"])
    actual_hash = _sha256(path)
    if actual_hash != str(spec["source"]["cache_sha256"]):
        raise RuntimeError("channel source cache SHA-256 changed")
    with np.load(path) as cache:
        cached_raw = np.asarray(cache["raw_features"], dtype=np.float32)
        labels = np.asarray(cache["labels"], dtype=np.int64)
        split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
        vortex_ids = np.asarray(cache["vortex_ids"], dtype=np.int32)
        metadata = json.loads(str(cache["metadata_json"]))
    required_support = [int(value) for value in spec["source"]["required_class_support"]]
    if np.bincount(labels, minlength=4).tolist() != required_support:
        raise RuntimeError("channel source class support changed")
    sampled_steps = int(metadata["streamlines"]["sampled_steps"])
    source_step = float(metadata["streamlines"]["spatial_step"])
    source_offset = float(metadata["streamlines"]["offset"])
    source_offset_over_step = source_offset / source_step
    if not np.isclose(
        source_offset_over_step,
        float(spec["streamlines"]["source_offset_over_spatial_step"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("channel source offset/step ratio differs from config")
    primitives = cached_raw.reshape(-1, 7, sampled_steps, 3)
    encoder = spec["encoder"]
    raw, fmt = dimensionless_primitive_features(
        primitives,
        source_step,
        num_freq=int(encoder["num_freq"]),
        neighbor_scale=float(encoder["neighbor_scale"]),
        neighbor_pool=str(encoder["neighbor_pool"]),
        mode=str(encoder["mode"]),
        include_chirality=bool(encoder["include_chirality"]),
    )
    if fmt.shape[1] != int(encoder["expected_feature_dim"]):
        raise RuntimeError("recomputed source FMT width differs from config")
    return {
        "raw": raw,
        "fmt": fmt,
        "labels": labels,
        "split_codes": split_codes,
        "vortex_ids": vortex_ids,
    }, {
        "path": str(path.resolve()),
        "sha256": actual_hash,
        "sample_count": int(len(labels)),
        "sampled_steps": sampled_steps,
        "source_spatial_step": source_step,
        "source_offset": source_offset,
        "source_offset_over_spatial_step": source_offset_over_step,
        "source_fmt_recomputed_after_dimensionless_coordinate_conversion": True,
    }


def _load_manifest(spec: dict, config_path: Path) -> tuple[Path, dict, str]:
    path = Path(spec["output_dir"]) / "target_cache" / "target_cache_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run experiments.Prepare_Task4B_ChannelToTBL_2_1"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != spec["experiment"]:
        raise RuntimeError("target cache belongs to a different experiment")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("target cache was prepared from a different config")
    if manifest.get("source_cache_sha256") != spec["source"]["cache_sha256"]:
        raise RuntimeError("target manifest source cache identity changed")
    if manifest.get("representation") != spec["representation"]:
        raise RuntimeError("target manifest representation differs from config")
    if manifest.get("encoder") != spec["encoder"]:
        raise RuntimeError("target manifest encoder differs from config")
    expected_steps = int(spec["streamlines"]["steps_per_direction"])
    if int(manifest["streamlines"]["steps_per_direction"]) != expected_steps:
        raise RuntimeError("target manifest streamline length differs from config")
    if not np.isclose(
        manifest["streamlines"]["offset_over_spatial_step"],
        spec["streamlines"]["source_offset_over_spatial_step"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("target manifest offset/step ratio differs from config")
    if not np.isclose(
        manifest["proxy"]["percentile"],
        spec["target_proxy"]["percentile"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("target manifest proxy percentile differs from config")
    for key, config_key in (
        ("target_flow_sha256", "flow_path"),
        ("target_gt_sha256", "gt_path"),
        ("target_vorticity_validation_sha256", "vorticity_validation_path"),
    ):
        if manifest.get(key) != _sha256(Path(spec["target"][config_key])):
            raise RuntimeError(f"target input identity changed for {config_key}")
    return path, manifest, _sha256(path)


def _normalization(spec: dict, source: dict, source_meta: dict, output_dir: Path):
    sampled_steps = int(source_meta["sampled_steps"])
    values = fit_source_normalization(
        source["raw"], source["fmt"], sampled_steps=sampled_steps
    )
    base_width = fmt_base_feature_width(
        int(spec["encoder"]["num_freq"]),
        str(spec["encoder"]["mode"]),
        bool(spec["encoder"]["include_chirality"]),
    )
    source_raw, source_fmt = apply_source_normalization(
        source["raw"],
        source["fmt"],
        values,
        sampled_steps=sampled_steps,
        fmt_base_width=base_width,
        neighbor_weight=float(
            spec["encoder"]["neighbor_weight_after_source_standardization"]
        ),
    )
    normalization_path = output_dir / "normalization_source_channel_only.npz"
    payload = {
        **values,
        "sampled_steps": np.asarray(sampled_steps, dtype=np.int32),
        "fmt_base_width": np.asarray(base_width, dtype=np.int32),
        "neighbor_weight": np.asarray(
            spec["encoder"]["neighbor_weight_after_source_standardization"],
            dtype=np.float32,
        ),
        "fitted_row_count": np.asarray(len(source["labels"]), dtype=np.int32),
        "fitted_domain": np.asarray("channel_only"),
    }
    # np.savez embeds ZIP metadata, so rewriting an unchanged archive can change
    # its byte hash and incorrectly invalidate runs produced by earlier commands.
    # Preserve the existing artifact only when every named array is identical.
    reuse_existing = False
    if normalization_path.is_file():
        try:
            with np.load(normalization_path, allow_pickle=False) as existing:
                reuse_existing = set(existing.files) == set(payload) and all(
                    np.array_equal(np.asarray(existing[key]), np.asarray(value))
                    for key, value in payload.items()
                )
        except (OSError, ValueError, KeyError):
            reuse_existing = False
    if not reuse_existing:
        np.savez_compressed(normalization_path, **payload)
    return source_raw, source_fmt, values, base_width, normalization_path


def _iter_target_chunks(
    manifest: dict,
    manifest_dir: Path,
    normalization: dict,
    *,
    sampled_steps: int,
    fmt_base_width_value: int,
    neighbor_weight: float,
):
    for entry in manifest["chunks"]:
        relative = entry.get("path_relative_to_manifest")
        if not relative:
            raise RuntimeError("target manifest chunk lacks a relative path")
        path = (manifest_dir / relative).resolve()
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"target chunk SHA-256 changed: {path}")
        with np.load(path) as chunk:
            raw_features = np.asarray(chunk["raw_features"], dtype=np.float32)
            fmt_features = np.asarray(chunk["fmt_features"], dtype=np.float32)
            raw, fmt = apply_source_normalization(
                raw_features,
                fmt_features,
                normalization,
                sampled_steps=sampled_steps,
                fmt_base_width=fmt_base_width_value,
                neighbor_weight=neighbor_weight,
            )
            yield {
                "raw": raw,
                "fmt": fmt,
                "labels": np.asarray(chunk["labels"], dtype=np.int64),
                "vortex_ids": np.asarray(chunk["vortex_ids"], dtype=np.int32),
                "seeds_xyz": np.asarray(chunk["seeds_xyz"], dtype=np.float32),
                "voxel_indices_xyz": np.asarray(
                    chunk["voxel_indices_xyz"], dtype=np.int32
                ),
                "source_candidate_indices": np.asarray(
                    chunk["source_candidate_indices"], dtype=np.int64
                ),
            }


def _distribution_shift(
    spec: dict,
    manifest: dict,
    manifest_dir: Path,
    normalization: dict,
    sampled_steps: int,
    base_width: int,
) -> dict:
    summaries = {
        "raw": {"count": 0, "sum": 0.0, "sum_sq": 0.0, "gt5": 0, "samples": []},
        "fmt": {"count": 0, "sum": 0.0, "sum_sq": 0.0, "gt5": 0, "samples": []},
    }
    for chunk in _iter_target_chunks(
        manifest,
        manifest_dir,
        normalization,
        sampled_steps=sampled_steps,
        fmt_base_width_value=base_width,
        neighbor_weight=float(
            spec["encoder"]["neighbor_weight_after_source_standardization"]
        ),
    ):
        for name in ("raw", "fmt"):
            flat = np.asarray(chunk[name], dtype=np.float32).reshape(-1)
            summary = summaries[name]
            summary["count"] += int(len(flat))
            summary["sum"] += float(np.sum(flat, dtype=np.float64))
            summary["sum_sq"] += float(np.sum(flat.astype(np.float64) ** 2))
            summary["gt5"] += int(np.count_nonzero(np.abs(flat) > 5.0))
            stride = max(1, len(flat) // 50_000)
            summary["samples"].append(np.abs(flat[::stride][:50_000]).astype(np.float32))
    result = {}
    for name, summary in summaries.items():
        count = int(summary["count"])
        sample = np.concatenate(summary.pop("samples"))
        mean = summary["sum"] / count
        result[name] = {
            "scalar_feature_value_count": count,
            "mean_standardized_value": mean,
            "standard_deviation_of_standardized_values": float(
                np.sqrt(max(0.0, summary["sum_sq"] / count - mean * mean))
            ),
            "median_absolute_z": float(np.percentile(sample, 50.0)),
            "p95_absolute_z": float(np.percentile(sample, 95.0)),
            "p99_absolute_z": float(np.percentile(sample, 99.0)),
            "fraction_absolute_z_gt_5": float(summary["gt5"] / count),
            "quantile_estimation": "deterministic within-chunk stride sample; moments and >5 fraction exact",
        }
    return result


def _target_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    vortex_ids: np.ndarray,
    validity_by_id: dict,
) -> dict:
    primary = _metrics(targets, probabilities)
    predicted = np.argmax(probabilities, axis=1)
    true_hairpin = targets >= 2
    predicted_hairpin = predicted >= 2
    hairpin_probability = probabilities[:, 2] + probabilities[:, 3]
    binary_target = true_hairpin.astype(np.int8)
    hairpin_rows = np.flatnonzero(true_hairpin)
    ordinary_rows = np.flatnonzero(~true_hairpin)

    oracle_hairpin_prediction = 2 + np.argmax(probabilities[hairpin_rows, 2:4], axis=1)
    oracle_ordinary_prediction = np.argmax(probabilities[ordinary_rows, 0:2], axis=1)
    per_vortex = {}
    valid_id_error = []
    valid_id_f1 = []
    valid_id_oracle_f1 = []
    coverage_adjusted_id_error = []
    total_hairpin_before = 0
    total_hairpin_invalid = 0
    for key in sorted(validity_by_id, key=int):
        vortex_id = int(key)
        member = vortex_ids == vortex_id
        valid_support = int(np.count_nonzero(member))
        before = int(validity_by_id[key]["before"])
        recorded_valid = int(validity_by_id[key]["valid"])
        if valid_support != recorded_valid or before < valid_support:
            raise RuntimeError(f"target VortexId validity mismatch for {vortex_id}")
        invalid_support = before - valid_support
        total_hairpin_before += before
        total_hairpin_invalid += invalid_support
        valid_ordinary_errors = int(np.count_nonzero(predicted[member] < 2))
        adjusted_error = float((invalid_support + valid_ordinary_errors) / before)
        coverage_adjusted_id_error.append(adjusted_error)
        row = {
            "support_before_streamline_validity": before,
            "valid_support": valid_support,
            "invalid_support": invalid_support,
            "valid_fraction": float(valid_support / before),
            "hairpin_to_ordinary_or_invalid_error_rate": adjusted_error,
        }
        if valid_support:
            valid_error = float(valid_ordinary_errors / valid_support)
            valid_f1 = float(
                f1_score(
                    targets[member], predicted[member], labels=[2, 3],
                    average="macro", zero_division=0,
                )
            )
            valid_oracle_f1 = float(
                f1_score(
                    targets[member],
                    2 + np.argmax(probabilities[member, 2:4], axis=1),
                    labels=[2, 3], average="macro", zero_division=0,
                )
            )
            row.update(
                {
                    "valid_hairpin_to_ordinary_error_rate": valid_error,
                    "valid_four_class_head_limb_macro_f1": valid_f1,
                    "valid_known_hairpin_mask_head_limb_macro_f1": valid_oracle_f1,
                }
            )
            valid_id_error.append(valid_error)
            valid_id_f1.append(valid_f1)
            valid_id_oracle_f1.append(valid_oracle_f1)
        else:
            row.update(
                {
                    "valid_hairpin_to_ordinary_error_rate": None,
                    "valid_four_class_head_limb_macro_f1": None,
                    "valid_known_hairpin_mask_head_limb_macro_f1": None,
                }
            )
        per_vortex[key] = row
    if total_hairpin_before <= 0 or total_hairpin_before - total_hairpin_invalid != len(
        hairpin_rows
    ):
        raise RuntimeError("pooled target hairpin validity count is inconsistent")
    diagnostics = {
        "hairpin_membership_f1": float(
            f1_score(binary_target, predicted_hairpin.astype(np.int8), zero_division=0)
        ),
        "hairpin_membership_average_precision": float(
            average_precision_score(binary_target, hairpin_probability)
        ),
        "valid_hairpin_to_ordinary_error_rate": float(
            np.mean(predicted[hairpin_rows] < 2)
        ),
        # Backward-compatible alias for the pre-audit 2.1 schema. Every row in
        # this pooled quantity has a valid streamline; 2.2 uses the explicit key.
        "hairpin_to_ordinary_error_rate": float(np.mean(predicted[hairpin_rows] < 2)),
        "valid_vortex_id_equal_hairpin_to_ordinary_error_rate": float(
            np.mean(valid_id_error)
        ),
        "hairpin_to_ordinary_or_invalid_error_rate": float(
            (total_hairpin_invalid + np.count_nonzero(predicted[hairpin_rows] < 2))
            / total_hairpin_before
        ),
        "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate": float(
            np.mean(coverage_adjusted_id_error)
        ),
        "hairpin_streamline_valid_fraction": float(
            len(hairpin_rows) / total_hairpin_before
        ),
        "target_vortex_id_count_before_validity": int(len(validity_by_id)),
        "target_vortex_id_count_with_valid_rows": int(len(valid_id_error)),
        "known_hairpin_mask_head_limb_macro_f1": float(
            f1_score(
                targets[hairpin_rows],
                oracle_hairpin_prediction,
                labels=[2, 3],
                average="macro",
                zero_division=0,
            )
        ),
        "known_ordinary_mask_orientation_macro_f1": float(
            f1_score(
                targets[ordinary_rows],
                oracle_ordinary_prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "valid_vortex_id_equal_four_class_head_limb_macro_f1": float(
            np.mean(valid_id_f1)
        ),
        "valid_vortex_id_equal_known_mask_head_limb_macro_f1": float(
            np.mean(valid_id_oracle_f1)
        ),
        "per_vortex_id": per_vortex,
    }
    return {"primary": primary, "diagnostics": diagnostics}


@torch.no_grad()
def _predict_target(
    model: nn.Module,
    variant: str,
    device: torch.device,
    spec: dict,
    manifest: dict,
    manifest_dir: Path,
    normalization: dict,
    sampled_steps: int,
    base_width: int,
):
    targets = []
    probabilities = []
    vortex_ids = []
    seeds = []
    voxel_indices = []
    candidate_indices = []
    model.eval()
    for chunk in _iter_target_chunks(
        manifest,
        manifest_dir,
        normalization,
        sampled_steps=sampled_steps,
        fmt_base_width_value=base_width,
        neighbor_weight=float(
            spec["encoder"]["neighbor_weight_after_source_standardization"]
        ),
    ):
        raw = torch.from_numpy(chunk["raw"]).to(device)
        fmt = torch.from_numpy(chunk["fmt"]).to(device)
        logits = _predict_logits(
            model,
            raw,
            fmt,
            variant=variant,
            batch_size=int(spec["training"]["evaluation_batch_size"]),
        ).numpy()
        shifted = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(shifted)
        prob /= prob.sum(axis=1, keepdims=True)
        targets.append(chunk["labels"])
        probabilities.append(prob.astype(np.float32))
        vortex_ids.append(chunk["vortex_ids"])
        seeds.append(chunk["seeds_xyz"])
        voxel_indices.append(chunk["voxel_indices_xyz"])
        candidate_indices.append(chunk["source_candidate_indices"])
        del raw, fmt
    return {
        "targets": np.concatenate(targets),
        "probabilities": np.concatenate(probabilities),
        "vortex_ids": np.concatenate(vortex_ids),
        "seeds_xyz": np.concatenate(seeds),
        "voxel_indices_xyz": np.concatenate(voxel_indices),
        "source_candidate_indices": np.concatenate(candidate_indices),
    }


def _train_one(
    spec: dict,
    source: dict,
    source_raw: np.ndarray,
    source_fmt: np.ndarray,
    source_meta: dict,
    manifest: dict,
    manifest_dir: Path,
    manifest_hash: str,
    config_hash: str,
    normalization_hash: str,
    normalization: dict,
    base_width: int,
    *,
    variant: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    _set_seed(seed)
    labels_np = source["labels"]
    sample_count = len(labels_np)
    raw = torch.from_numpy(source_raw).to(device)
    fmt = torch.from_numpy(source_fmt).to(device)
    labels = torch.from_numpy(labels_np.astype(np.int64)).to(device)
    model_spec = spec["model"]
    model = PathlineMulticlassClassifier3D(
        variant=variant,
        fmt_dim=source_fmt.shape[1],
        num_classes=4,
        temporal_width=int(model_spec["temporal_width"]),
        embedding_dim=int(model_spec["embedding_dim"]),
        auxiliary_dim=int(model_spec["auxiliary_dim"]),
        dropout=float(model_spec["dropout"]),
    ).to(device)
    parameter_count = trainable_parameter_count(model)
    training = spec["training"]
    criterion = nn.CrossEntropyLoss()
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
    required_streak = int(
        training["stop_gate"]["consecutive_source_zero_error_epochs"]
    )
    exact_streak = 0
    started = time.perf_counter()
    final_source_logits = None
    for epoch in range(1, int(training["max_epochs"]) + 1):
        order = epoch_permutation(sample_count, seed, epoch)
        certificate = permutation_certificate(order, sample_count)
        if certificate["unique_count"] != sample_count or certificate["duplicate_count"]:
            raise RuntimeError("source epoch is not one exact without-replacement pass")
        model.train()
        loss_sum = 0.0
        for start in range(0, sample_count, int(training["batch_size"])):
            batch = torch.from_numpy(order[start : start + int(training["batch_size"])]).to(
                device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                raw[batch],
                fmt[batch] if variant in {"fmt_only", "raw_fmt"} else None,
            )
            loss = criterion(logits, labels[batch])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        final_source_logits = _predict_logits(
            model,
            raw,
            fmt,
            variant=variant,
            batch_size=int(training["evaluation_batch_size"]),
        ).numpy()
        fit = _fit_metrics(labels_np, final_source_logits)
        exact_streak = exact_streak + 1 if _is_exact(fit) else 0
        row = {
            "epoch": epoch,
            "source_cross_entropy": loss_sum / sample_count,
            "source_accuracy": fit["accuracy"],
            "source_error_count": fit["error_count"],
            "source_macro_f1": fit["macro_f1"],
            "minimum_true_logit_margin": fit["minimum_true_logit_margin"],
            "zero_error_streak": exact_streak,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **certificate,
        }
        _write_history_row(history_path, row)
        if epoch <= 5 or epoch % 10 == 0 or fit["error_count"] < 10 or exact_streak:
            print(
                f"{variant} seed={seed} epoch={epoch:04d} "
                f"errors={fit['error_count']} macro_F1={fit['macro_f1']:.7f} "
                f"min_margin={fit['minimum_true_logit_margin']:.6g} "
                f"streak={exact_streak}/{required_streak}",
                flush=True,
            )
        if exact_streak >= required_streak:
            break
        scheduler.step()
    passed = bool(exact_streak >= required_streak and _is_exact(fit))
    if not passed:
        raise RuntimeError(
            f"{variant} seed={seed} failed the channel memorization gate after {epoch} epochs"
        )

    # The target is first touched by the model only after the source gate passes.
    target = _predict_target(
        model,
        variant,
        device,
        spec,
        manifest,
        manifest_dir,
        normalization,
        int(source_meta["sampled_steps"]),
        base_width,
    )
    target_metrics = _target_metrics(
        target["targets"],
        target["probabilities"],
        target["vortex_ids"],
        manifest["streamlines"]["per_vortex_id_validity"],
    )
    shifted = final_source_logits - final_source_logits.max(axis=1, keepdims=True)
    source_probabilities = np.exp(shifted)
    source_probabilities /= source_probabilities.sum(axis=1, keepdims=True)
    prediction_path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        source_row_indices=np.arange(sample_count, dtype=np.int64),
        source_targets=labels_np.astype(np.int8),
        source_probabilities=source_probabilities.astype(np.float32),
        source_predicted_labels=np.argmax(source_probabilities, axis=1).astype(np.int8),
        target_source_candidate_indices=target["source_candidate_indices"],
        target_voxel_indices_xyz=target["voxel_indices_xyz"],
        target_seeds_xyz=target["seeds_xyz"],
        target_vortex_ids=target["vortex_ids"],
        target_targets=target["targets"].astype(np.int8),
        target_probabilities=target["probabilities"],
        target_predicted_labels=np.argmax(target["probabilities"], axis=1).astype(np.int8),
    )
    result = {
        "variant": variant,
        "seed": int(seed),
        "parameter_count": int(parameter_count),
        "source_gate_passed": passed,
        "source_selected_epoch": int(epoch),
        "source_terminal_zero_error_streak": int(exact_streak),
        "source_metrics": fit,
        "target_metrics": target_metrics,
        "target_row_count": int(len(target["targets"])),
        "elapsed_seconds": float(time.perf_counter() - started),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "history_path": str(history_path.resolve()),
        "history_sha256": _sha256(history_path),
        "source_cache_sha256": source_meta["sha256"],
        "target_cache_manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "normalization_sha256": normalization_hash,
        "target_evaluation_count": 1,
        "checkpoint_policy": "model_state_in_memory_only_no_persistent_checkpoint",
    }
    run_path = output_dir / "runs" / f"{variant}_seed{seed}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{variant} seed={seed} TBL macro_F1="
        f"{target_metrics['primary']['macro_f1']:.6f}",
        flush=True,
    )
    del model, raw, fmt, labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate(spec: dict, rows: list[dict]) -> dict:
    by_variant = {}
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
        if not selected:
            continue
        item = {
            "run_count": len(selected),
            "parameter_count": int(selected[0]["parameter_count"]),
            "source_gate_passed_count": int(
                sum(bool(row["source_gate_passed"]) for row in selected)
            ),
            "source_selected_epochs": [int(row["source_selected_epoch"]) for row in selected],
        }
        for name, path in metric_paths.items():
            values = np.asarray(
                [row["target_metrics"][path[0]][path[1]] for row in selected],
                dtype=np.float64,
            )
            item[f"target_{name}_mean"] = float(np.mean(values))
            item[f"target_{name}_std"] = float(
                np.std(values, ddof=1 if len(values) > 1 else 0)
            )
        for class_name in CLASS_NAMES:
            values = np.asarray(
                [
                    row["target_metrics"]["primary"]["per_class_f1"][class_name]
                    for row in selected
                ]
            )
            item[f"target_f1_{class_name}_mean"] = float(np.mean(values))
            item[f"target_f1_{class_name}_std"] = float(
                np.std(values, ddof=1 if len(values) > 1 else 0)
            )
        by_variant[variant] = item

    paired = {}
    lookup = {(row["variant"], int(row["seed"])): row for row in rows}
    lower_is_better = {
        "valid_hairpin_to_ordinary_error_rate",
        "valid_vortex_id_equal_hairpin_to_ordinary_error_rate",
        "hairpin_to_ordinary_or_invalid_error_rate",
        "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate",
    }
    for baseline in ("raw", "raw_wide"):
        for metric, path in metric_paths.items():
            differences = []
            per_seed = {}
            for seed in spec["training"]["seeds"]:
                left = lookup.get(("raw_fmt", int(seed)))
                right = lookup.get((baseline, int(seed)))
                if left is None or right is None:
                    continue
                raw_difference = float(
                    left["target_metrics"][path[0]][path[1]]
                    - right["target_metrics"][path[0]][path[1]]
                )
                improvement = -raw_difference if metric in lower_is_better else raw_difference
                differences.append(improvement)
                per_seed[str(seed)] = improvement
            if differences:
                values = np.asarray(differences)
                paired[f"raw_fmt_vs_{baseline}_{metric}_improvement"] = {
                    "direction": "baseline_minus_raw_fmt" if metric in lower_is_better else "raw_fmt_minus_baseline",
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1 if len(values) > 1 else 0)),
                    "improved_seed_count": int(np.count_nonzero(values > 0.0)),
                    "per_seed": per_seed,
                }
    expected = len(spec["variants"]) * len(spec["training"]["seeds"])
    return {
        "expected_run_count": expected,
        "completed_run_count": len(rows),
        "experiment_complete": len(rows) == expected,
        "all_source_memorization_gates_passed": bool(
            len(rows) == expected and all(row["source_gate_passed"] for row in rows)
        ),
        "variants": by_variant,
        "paired_comparisons": paired,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    flattened = []
    for row in rows:
        primary = row["target_metrics"]["primary"]
        diagnostic = row["target_metrics"]["diagnostics"]
        item = {
            "variant": row["variant"],
            "seed": row["seed"],
            "parameter_count": row["parameter_count"],
            "source_selected_epoch": row["source_selected_epoch"],
            "source_error_count": row["source_metrics"]["error_count"],
            "target_macro_f1": primary["macro_f1"],
            "target_balanced_accuracy": primary["balanced_accuracy"],
            "target_macro_average_precision_ovr": primary["macro_average_precision_ovr"],
            "target_hairpin_membership_f1": diagnostic["hairpin_membership_f1"],
            "target_valid_hairpin_to_ordinary_error_rate": diagnostic[
                "valid_hairpin_to_ordinary_error_rate"
            ],
            "target_valid_vortex_id_equal_hairpin_to_ordinary_error_rate": diagnostic[
                "valid_vortex_id_equal_hairpin_to_ordinary_error_rate"
            ],
            "target_hairpin_to_ordinary_or_invalid_error_rate": diagnostic[
                "hairpin_to_ordinary_or_invalid_error_rate"
            ],
            "target_vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate": diagnostic[
                "vortex_id_equal_hairpin_to_ordinary_or_invalid_error_rate"
            ],
            "target_known_hairpin_mask_head_limb_macro_f1": diagnostic[
                "known_hairpin_mask_head_limb_macro_f1"
            ],
            "target_valid_vortex_id_equal_four_class_head_limb_macro_f1": diagnostic[
                "valid_vortex_id_equal_four_class_head_limb_macro_f1"
            ],
        }
        for class_name in CLASS_NAMES:
            item[f"target_f1_{class_name}"] = primary["per_class_f1"][class_name]
        flattened.append(item)
    if flattened:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    summary_only: bool = False,
) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(config_path)
    source, source_meta = _load_source(spec)
    manifest_path, manifest, manifest_hash = _load_manifest(spec, config_path)
    manifest_dir = manifest_path.parent
    source_raw, source_fmt, normalization, base_width, normalization_path = _normalization(
        spec, source, source_meta, output_dir
    )
    normalization_hash = _sha256(normalization_path)
    shift_path = output_dir / "target_distribution_shift.json"
    shift_identity = {
        "target_cache_manifest_sha256": manifest_hash,
        "normalization_sha256": normalization_hash,
    }
    shift_payload = None
    if shift_path.exists():
        candidate = json.loads(shift_path.read_text(encoding="utf-8"))
        if candidate.get("identity") == shift_identity:
            shift_payload = candidate
    if shift_payload is None:
        shift_statistics = _distribution_shift(
            spec,
            manifest,
            manifest_dir,
            normalization,
            int(source_meta["sampled_steps"]),
            base_width,
        )
        shift_payload = {"identity": shift_identity, "statistics": shift_statistics}
        shift_path.write_text(json.dumps(shift_payload, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not summary_only:
        selected_variants = list(
            dict.fromkeys(variants or [str(value) for value in spec["variants"]])
        )
        selected_seeds = list(
            dict.fromkeys(seeds or [int(value) for value in spec["training"]["seeds"]])
        )
        for variant in selected_variants:
            if variant not in spec["variants"]:
                raise ValueError(f"variant {variant!r} is not in the frozen config")
            for seed in selected_seeds:
                if seed not in spec["training"]["seeds"]:
                    raise ValueError(f"seed {seed} is not in the frozen config")
                _train_one(
                    spec,
                    source,
                    source_raw,
                    source_fmt,
                    source_meta,
                    manifest,
                    manifest_dir,
                    manifest_hash,
                    config_hash,
                    normalization_hash,
                    normalization,
                    base_width,
                    variant=variant,
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                )

    rows = []
    stale_run_paths = []
    for variant in spec["variants"]:
        for seed in spec["training"]["seeds"]:
            path = output_dir / "runs" / f"{variant}_seed{seed}.json"
            if path.is_file():
                row = json.loads(path.read_text(encoding="utf-8"))
                identity_matches = bool(
                    row.get("variant") == variant
                    and int(row.get("seed", -1)) == int(seed)
                    and row.get("config_sha256") == config_hash
                    and row.get("source_cache_sha256") == source_meta["sha256"]
                    and row.get("target_cache_manifest_sha256") == manifest_hash
                    and row.get("normalization_sha256") == normalization_hash
                    and row.get("prediction_sha256")
                    == _sha256(Path(row.get("prediction_path", "__missing__")))
                ) if Path(row.get("prediction_path", "__missing__")).is_file() else False
                if identity_matches:
                    rows.append(row)
                else:
                    stale_run_paths.append(str(path.resolve()))
    aggregate = _aggregate(spec, rows)
    _write_csv(rows, output_dir / "per_run_metrics.csv")
    summary = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "config_sha256": config_hash,
        "source": source_meta,
        "target_cache_manifest": str(manifest_path.resolve()),
        "target_cache_manifest_sha256": manifest_hash,
        "normalization": {
            "path": str(normalization_path.resolve()),
            "sha256": normalization_hash,
            "fit_domain": "channel only",
            "fit_row_count": len(source["labels"]),
        },
        "target_distribution_shift": shift_payload,
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
        },
        "aggregate": aggregate,
        "runs": rows,
        "stale_run_paths_ignored": stale_run_paths,
        "checkpoint_policy": "no persistent model checkpoint",
        "interpretation_boundary": (
            "All network parameters and normalization statistics use channel rows only. "
            "TBL is evaluated once after exact channel memorization. VortexIds manually "
            "define hairpin support/identity, while all four part/type labels remain proxy."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--variant", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.config,
        variants=args.variant,
        seeds=args.seed,
        summary_only=args.summary_only,
    )
