"""Freeze Task3 residual alpha using old validation data only.

For each dataset and seed, select the alpha with maximum Average Precision
among candidates whose validation F1 exceeds the stronger of Raw and Raw-wide
by a configured margin.  The corresponding F1-optimal threshold is frozen for
future confirmation data.  Network weights are never changed here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_baseline, _find_checkpoint, _load_baseline, _load_residual,
    _normalise_with_checkpoint, _summarise,
)
from Verify_Task3_FMTClassifier import (
    _classification_metrics, _load_dataset, _loader, _select_f1_threshold,
    _stack_split,
)
from Verify_Task3_FMTResidual import _predict_components, _probabilities


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_constrained(candidates, minimum_f1, minimum_ap):
    feasible = [
        row for row in candidates
        if row["validation_f1"] >= minimum_f1
    ]
    if not feasible:
        best = max(candidates, key=lambda row: row["validation_f1"])
        raise RuntimeError(
            f"no alpha satisfies validation F1 >= {minimum_f1:.6f}; "
            f"best is {best['validation_f1']:.6f}"
        )
    selected = max(
        feasible,
        key=lambda row: (row["validation_average_precision"], row["validation_f1"]),
    )
    if selected["validation_average_precision"] < minimum_ap:
        raise RuntimeError(
            "constrained alpha does not meet the old-validation Average "
            f"Precision target {minimum_ap:.6f}; got "
            f"{selected['validation_average_precision']:.6f}"
        )
    return selected, len(feasible)


def _select_pareto(candidates, f1_tolerance):
    """Maximise AP without dropping more than tolerance from best F1."""
    maximum_f1 = max(row["validation_f1"] for row in candidates)
    minimum_f1 = maximum_f1 - float(f1_tolerance)
    feasible = [
        row for row in candidates if row["validation_f1"] >= minimum_f1
    ]
    selected = max(
        feasible,
        key=lambda row: (row["validation_average_precision"], row["validation_f1"]),
    )
    return selected, len(feasible), maximum_f1


def run(config_path):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "config_snapshot.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if previous != spec:
            raise RuntimeError(
                f"configuration changed in {output_dir}; use a new version"
            )
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    device_name = spec.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    alpha_grid = np.linspace(
        float(spec["alpha_min"]), float(spec["alpha_max"]),
        int(spec["alpha_steps"]),
    )
    required = sorted(set(spec["train_ordinals"]) | set(spec["validation_ordinals"]))
    rows = []
    evaluation_rows = []
    for dataset in spec["datasets"]:
        records = _load_dataset(
            Path(spec["source_cache_root"]) / dataset,
            Path(spec["label_cache_root"]) / dataset,
            spec["sampled_steps"], spec["fmt_subset"],
            required_ordinals=required,
            gram_num_freq=spec.get("fmt_gram_num_freq", 6),
        )
        validation = _stack_split(records, spec["validation_ordinals"])
        for seed_value in spec["seeds"]:
            seed = int(seed_value)
            baseline_metrics = {}
            for variant in ("raw", "raw_wide"):
                path = _find_checkpoint(
                    spec["baseline_checkpoint_roots"],
                    f"{dataset}_{variant}_seed{seed}.pt",
                )
                model, checkpoint = _load_baseline(
                    path, validation[1].shape[1], device
                )
                _, _, metrics = _evaluate_baseline(
                    model, checkpoint, validation, spec["batch_size"], seed, device
                )
                baseline_metrics[variant] = metrics
                evaluation_rows.append({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    **metrics,
                })
            stronger_f1 = max(
                baseline_metrics["raw"]["f1"],
                baseline_metrics["raw_wide"]["f1"],
            )
            stronger_ap = max(
                baseline_metrics["raw"]["average_precision"],
                baseline_metrics["raw_wide"]["average_precision"],
            )
            residual_path = _find_checkpoint(
                spec["residual_checkpoint_roots"],
                f"{dataset}_raw_fmt_residual_seed{seed}.pt",
            )
            model, checkpoint = _load_residual(
                residual_path, validation[1].shape[1], device
            )
            normalised = _normalise_with_checkpoint(validation, checkpoint)
            loader = _loader(
                normalised, spec["batch_size"], False, seed,
                device.type == "cuda",
            )
            targets, raw_logits, residual_logits = _predict_components(
                model, loader, device
            )
            candidates = []
            for alpha in alpha_grid:
                probabilities = _probabilities(
                    raw_logits, residual_logits, float(alpha)
                )
                threshold = _select_f1_threshold(targets, probabilities)
                metrics = _classification_metrics(
                    targets, probabilities, threshold
                )
                candidates.append({
                    "alpha": float(alpha), "threshold": float(threshold),
                    "validation_f1": metrics["f1"],
                    "validation_average_precision": metrics["average_precision"],
                })
            selection = spec.get("selection", {"mode": "baseline_margin"})
            mode = str(selection.get("mode", "baseline_margin"))
            if mode == "baseline_margin":
                margin = float(spec["minimum_obvious_gain"])
                selected, feasible_count = _select_constrained(
                    candidates,
                    minimum_f1=stronger_f1 + margin,
                    minimum_ap=stronger_ap + margin,
                )
                maximum_f1 = max(row["validation_f1"] for row in candidates)
            elif mode == "pareto_f1_tolerance":
                selected, feasible_count, maximum_f1 = _select_pareto(
                    candidates, float(selection["f1_tolerance"])
                )
            else:
                raise ValueError(f"unknown calibration selection mode {mode!r}")
            row = {
                "dataset": dataset, "seed": seed,
                "selection_mode": mode,
                "selected_alpha": selected["alpha"],
                "selected_threshold": selected["threshold"],
                "maximum_candidate_f1": maximum_f1,
                "selected_f1_drop_from_max": (
                    maximum_f1 - selected["validation_f1"]
                ),
                "validation_stronger_raw_f1": stronger_f1,
                "validation_raw_fmt_f1": selected["validation_f1"],
                "validation_gain_f1": selected["validation_f1"] - stronger_f1,
                "validation_stronger_raw_average_precision": stronger_ap,
                "validation_raw_fmt_average_precision": selected[
                    "validation_average_precision"
                ],
                "validation_gain_average_precision": selected[
                    "validation_average_precision"
                ] - stronger_ap,
                "feasible_alpha_count": feasible_count,
                "residual_checkpoint": str(residual_path),
            }
            rows.append(row)
            evaluation_rows.append({
                "dataset": dataset, "seed": seed,
                "variant": "raw_fmt_residual",
                "f1": selected["validation_f1"],
                "average_precision": selected["validation_average_precision"],
            })
            print(
                f"{dataset:22s} seed={seed} alpha={row['selected_alpha']:.3f} "
                f"gain_F1={row['validation_gain_f1']:+.5f} "
                f"gain_AP={row['validation_gain_average_precision']:+.5f}",
                flush=True,
            )
    _write_csv(output_dir / "calibration.csv", rows)
    _, aggregate = _summarise(
        evaluation_rows, float(spec["minimum_obvious_gain"])
    )
    _write_csv(output_dir / "old_validation_summary.csv", aggregate)
    return output_dir / "calibration.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run(arguments.config)
