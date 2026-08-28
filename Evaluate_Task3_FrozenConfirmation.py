"""Evaluate frozen Task3 Raw and Raw+FMT classifiers on unseen seed times.

This script is deliberately evaluation-only.  Normalization statistics,
network weights, FMT fusion alpha, and the decision threshold all come from
the old training/validation checkpoints.  Nothing is fitted or selected on
the confirmation slices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D, PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Verify_Task3_FMTClassifier import (
    _classification_metrics, _load_dataset, _loader, _predict, _stack_split,
)
from Verify_Task3_FMTResidual import (
    _apply_raw_pca_transform, _load_raw_model, _predict_components,
    _probabilities,
)


def _find_checkpoint(roots, filename):
    matches = [Path(root) / filename for root in roots]
    matches = [path for path in matches if path.exists()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one checkpoint named {filename}, found {matches}"
        )
    return matches[0]


def _normalise_with_checkpoint(split, checkpoint, require_fmt=True):
    raw, fmt, labels = split
    stats = checkpoint["normalization"]
    raw_mean = np.asarray(stats["raw_mean"], dtype=np.float32)
    raw_std = np.asarray(stats["raw_std"], dtype=np.float32)
    fmt_mean = np.asarray(stats["fmt_mean"], dtype=np.float32)
    fmt_std = np.asarray(stats["fmt_std"], dtype=np.float32)
    raw = ((raw - raw_mean) / raw_std).astype(np.float32)
    if require_fmt:
        if fmt.shape[1] != fmt_mean.shape[1]:
            raise ValueError(
                f"checkpoint expects {fmt_mean.shape[1]} FMT features, "
                f"got {fmt.shape[1]}"
            )
        fmt = ((fmt - fmt_mean) / fmt_std).astype(np.float32)
    else:
        # Raw and Raw-wide never consume this tensor.  Their historical
        # checkpoints may contain statistics for an older FMT representation.
        fmt = fmt.astype(np.float32, copy=False)
    if not np.isfinite(raw).all() or not np.isfinite(fmt).all():
        raise ValueError("non-finite input after frozen checkpoint normalization")
    return raw, fmt, labels.astype(np.float32)


def _load_baseline(path, fmt_dim, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    variant = str(checkpoint["variant"])
    if variant not in {"raw", "raw_wide"}:
        raise ValueError(f"expected Raw baseline checkpoint, got {variant}")
    source = checkpoint["config"]
    model = PathlineBinaryClassifier3D(
        variant=variant,
        fmt_dim=int(fmt_dim),
        temporal_width=source["model"]["temporal_width"],
        embedding_dim=source["model"]["embedding_dim"],
        auxiliary_dim=source["model"]["auxiliary_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def _load_residual(path, fmt_dim, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] not in {"raw_fmt_residual", "raw_pca_residual"}:
        raise ValueError(f"expected residual checkpoint, got {checkpoint['variant']}")
    auxiliary_dim = int(fmt_dim)
    if checkpoint["variant"] == "raw_pca_residual":
        transform = checkpoint.get("auxiliary_transform")
        if not transform or transform.get("kind") != "raw_pca":
            raise ValueError("Raw-PCA residual checkpoint misses its transform")
        auxiliary_dim = int(np.asarray(transform["components"]).shape[0])
    raw_path = Path(checkpoint["raw_checkpoint"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"residual checkpoint references missing Raw checkpoint: {raw_path}"
        )
    raw_model, raw_checkpoint = _load_raw_model(raw_path, auxiliary_dim, device)
    for key in ("raw_mean", "raw_std"):
        if not np.array_equal(
            np.asarray(checkpoint["normalization"][key]),
            np.asarray(raw_checkpoint["normalization"][key]),
        ):
            raise RuntimeError(
                f"Raw and residual checkpoints disagree on frozen {key}"
            )
    model_spec = checkpoint["config"]["model"]
    model = PathlineFMTResidualClassifier3D(
        raw_model,
        fmt_dim=auxiliary_dim,
        **residual_model_kwargs(model_spec),
    ).to(device)
    state = model.state_dict()
    state.update(checkpoint["residual_state_dict"])
    model.load_state_dict(state)
    return model.eval(), checkpoint


def _evaluate_baseline(model, checkpoint, split, batch_size, seed, device):
    normalised = _normalise_with_checkpoint(split, checkpoint, require_fmt=False)
    loader = _loader(
        normalised, batch_size=batch_size, shuffle=False, seed=seed,
        pin_memory=device.type == "cuda",
    )
    targets, probabilities = _predict(model, loader, device)
    return targets, probabilities, _classification_metrics(
        targets, probabilities, checkpoint["threshold"]
    )


def _evaluate_residual(model, checkpoint, split, batch_size, seed, device):
    is_raw_pca = checkpoint["variant"] == "raw_pca_residual"
    normalised = _normalise_with_checkpoint(
        split, checkpoint, require_fmt=not is_raw_pca
    )
    if is_raw_pca:
        raw, _, labels = normalised
        auxiliary = _apply_raw_pca_transform(
            raw, checkpoint["auxiliary_transform"]
        )
        normalised = raw, auxiliary, labels
    loader = _loader(
        normalised, batch_size=batch_size, shuffle=False, seed=seed,
        pin_memory=device.type == "cuda",
    )
    targets, raw_logits, residual_logits = _predict_components(model, loader, device)
    probabilities = _probabilities(
        raw_logits, residual_logits, checkpoint["alpha"]
    )
    return targets, probabilities, _classification_metrics(
        targets, probabilities, checkpoint["threshold"]
    )


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_calibration(path):
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]))
        if key in result:
            raise RuntimeError(f"duplicate calibration row {key}")
        result[key] = {
            "alpha": float(row["selected_alpha"]),
            "threshold": float(row["selected_threshold"]),
        }
    return result


def _summarise(rows, minimum_gain):
    by_key = {(row["dataset"], row["seed"], row["variant"]): row for row in rows}
    comparisons = []
    datasets = sorted({row["dataset"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    for dataset in datasets:
        for seed in seeds:
            raw = by_key[(dataset, seed, "raw")]
            wide = by_key[(dataset, seed, "raw_wide")]
            fmt = by_key[(dataset, seed, "raw_fmt_residual")]
            strong_f1 = max(float(raw["f1"]), float(wide["f1"]))
            strong_ap = max(
                float(raw["average_precision"]),
                float(wide["average_precision"]),
            )
            gain_f1 = float(fmt["f1"]) - strong_f1
            gain_ap = float(fmt["average_precision"]) - strong_ap
            comparisons.append({
                "dataset": dataset,
                "seed": seed,
                "stronger_raw_f1": strong_f1,
                "raw_fmt_f1": float(fmt["f1"]),
                "gain_f1": gain_f1,
                "stronger_raw_average_precision": strong_ap,
                "raw_fmt_average_precision": float(fmt["average_precision"]),
                "gain_average_precision": gain_ap,
                "passes_minimum_gain": int(
                    gain_f1 >= minimum_gain and gain_ap >= minimum_gain
                ),
            })
    aggregate = []
    for dataset in datasets:
        selected = [row for row in comparisons if row["dataset"] == dataset]
        source = [row for row in rows if row["dataset"] == dataset]
        by_variant = {
            variant: [row for row in source if row["variant"] == variant]
            for variant in ("raw", "raw_wide", "raw_fmt_residual")
        }
        item = {"dataset": dataset, "seed_count": len(selected)}
        for metric in ("f1", "average_precision"):
            means = {
                variant: float(np.mean([
                    float(row[metric]) for row in variant_rows
                ]))
                for variant, variant_rows in by_variant.items()
            }
            fmt_values = np.asarray([
                float(row[metric])
                for row in by_variant["raw_fmt_residual"]
            ])
            stronger_method = max(means["raw"], means["raw_wide"])
            item[f"mean_raw_{metric}"] = means["raw"]
            item[f"mean_raw_wide_{metric}"] = means["raw_wide"]
            item[f"mean_stronger_raw_{metric}"] = stronger_method
            item[f"mean_raw_fmt_{metric}"] = means["raw_fmt_residual"]
            item[f"mean_gain_{metric}"] = (
                means["raw_fmt_residual"] - stronger_method
            )
            item[f"std_raw_fmt_{metric}"] = float(fmt_values.std(ddof=0))
            oracle_values = np.asarray([
                float(row[f"stronger_raw_{metric}"]) for row in selected
            ])
            item[f"mean_per_seed_oracle_raw_{metric}"] = float(
                oracle_values.mean()
            )
            item[f"gain_vs_per_seed_oracle_raw_{metric}"] = float(
                fmt_values.mean() - oracle_values.mean()
            )
        item["passes_minimum_gain"] = int(
            item["mean_gain_f1"] >= minimum_gain
            and item["mean_gain_average_precision"] >= minimum_gain
        )
        aggregate.append(item)
    return comparisons, aggregate


def run(config_path):
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "config_snapshot.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if previous != spec:
            raise RuntimeError(
                f"configuration changed in {output_dir}; use a new experiment version"
            )
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    device_name = spec.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    print(f"device={device}", flush=True)
    calibration = _read_calibration(spec.get("calibration_csv"))
    rows = []
    ordinals = list(range(int(spec["expected_slices"])))
    for dataset in spec["datasets"]:
        records = _load_dataset(
            Path(spec["source_cache_root"]) / dataset,
            Path(spec["label_cache_root"]) / dataset,
            spec["sampled_steps"], spec["fmt_subset"],
            required_ordinals=ordinals,
            gram_num_freq=spec.get("fmt_gram_num_freq", 6),
            expected_slices=spec["expected_slices"],
        )
        split = _stack_split(records, ordinals)
        for seed in spec["seeds"]:
            seed = int(seed)
            paths = {
                variant: _find_checkpoint(
                    spec["baseline_checkpoint_roots"],
                    f"{dataset}_{variant}_seed{seed}.pt",
                )
                for variant in ("raw", "raw_wide")
            }
            paths["raw_fmt_residual"] = _find_checkpoint(
                spec["residual_checkpoint_roots"],
                f"{dataset}_raw_fmt_residual_seed{seed}.pt",
            )
            for variant in ("raw", "raw_wide"):
                model, checkpoint = _load_baseline(
                    paths[variant], split[1].shape[1], device
                )
                targets, _, metrics = _evaluate_baseline(
                    model, checkpoint, split, spec["batch_size"], seed, device
                )
                rows.append({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "sample_count": len(targets),
                    "positive_fraction": float(targets.mean()),
                    "frozen_threshold": float(checkpoint["threshold"]),
                    "frozen_alpha": 0.0,
                    **metrics,
                    "checkpoint": str(paths[variant]),
                })
            model, checkpoint = _load_residual(
                paths["raw_fmt_residual"], split[1].shape[1], device
            )
            checkpoint = dict(checkpoint)
            if calibration:
                key = (dataset, seed)
                if key not in calibration:
                    raise RuntimeError(f"missing frozen calibration for {key}")
                checkpoint["alpha"] = calibration[key]["alpha"]
                checkpoint["threshold"] = calibration[key]["threshold"]
            targets, _, metrics = _evaluate_residual(
                model, checkpoint, split, spec["batch_size"], seed, device
            )
            rows.append({
                "dataset": dataset, "seed": seed,
                "variant": "raw_fmt_residual",
                "sample_count": len(targets),
                "positive_fraction": float(targets.mean()),
                "frozen_threshold": float(checkpoint["threshold"]),
                "frozen_alpha": float(checkpoint["alpha"]),
                **metrics,
                "checkpoint": str(paths["raw_fmt_residual"]),
            })
            print(f"evaluated {dataset} seed={seed}", flush=True)
    _write_csv(output_dir / "per_run.csv", rows)
    comparisons, aggregate = _summarise(
        rows, float(spec["minimum_obvious_gain"])
    )
    _write_csv(output_dir / "per_seed_comparison.csv", comparisons)
    _write_csv(output_dir / "per_dataset_summary.csv", aggregate)
    audit = {
        "experiment": spec["experiment"],
        "confirmation_data_was_not_used_for_selection": True,
        "calibration_csv": spec.get("calibration_csv"),
        "minimum_obvious_gain": float(spec["minimum_obvious_gain"]),
        "all_datasets_pass": bool(all(
            row["passes_minimum_gain"] for row in aggregate
        )),
        "dataset_count": len(aggregate),
        "seed_count": len(spec["seeds"]),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    return output_dir / "per_dataset_summary.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run(arguments.config)
