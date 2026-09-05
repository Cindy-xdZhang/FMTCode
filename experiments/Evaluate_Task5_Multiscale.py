"""Evaluate Task5 and fixed-scale Task3 transfer on unseen scale tuples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import Evaluate_Task3_MainTable as task3_table
from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_residual,
    _find_checkpoint,
    _load_residual,
    _write_csv,
)
from Verify_Task3_FMTClassifier import _load_dataset, _stack_split


def _load_scale_ids(source_dir, expected_slices):
    paths = sorted(Path(source_dir).glob("slice_*.npz"))
    if len(paths) != int(expected_slices):
        raise RuntimeError(f"expected {expected_slices} scale caches in {source_dir}")
    result = []
    names = None
    for path in paths:
        with np.load(path) as data:
            ids = np.asarray(data["scale_id"], dtype=np.int64)
            metadata = json.loads(str(data["metadata_json"]))
        current = [row["name"] for row in metadata["scale_table"]]
        if names is None:
            names = current
        elif names != current:
            raise RuntimeError(f"confirmation scale table changed in {source_dir}")
        result.append(ids)
    return result, names


def _subset_record(record, mask):
    mask = np.asarray(mask, dtype=bool)
    return record[0][mask], record[1][mask], record[2][mask]


def _mean_std(rows, dataset, variant, metric):
    values = np.asarray([
        float(row[metric]) for row in rows
        if row["dataset"] == dataset and row["variant"] == variant
    ])
    return float(values.mean()), float(values.std(ddof=1))


def _write_comparison_markdown(rows, path):
    lines = [
        "# Task5 3D variable-scale confirmation",
        "",
        "Task3 transfer uses models trained only at the fixed scale. Task5 models "
        "are trained on 18 scale tuples and evaluated on 9 disjoint tuples.",
        "",
        "| Flow | Task3 fixed Raw-PCA transfer F1 | Task3 fixed FMT transfer F1 | Task5 Raw-PCA F1 | Task5 FMT F1 | Task5 FMT−Raw-PCA | Task5−fixed FMT | Task5 Raw-PCA AP | Task5 FMT AP | FMT−Raw-PCA AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["dataset"]} | {row["fixed_raw_pca_f1"]:.4f} | '
            f'{row["fixed_fmt_f1"]:.4f} | {row["task5_raw_pca_f1"]:.4f} | '
            f'{row["task5_fmt_f1"]:.4f} | **{row["fmt_minus_raw_pca_f1"]:+.4f}** | '
            f'{row["task5_minus_fixed_fmt_f1"]:+.4f} | '
            f'{row["task5_raw_pca_average_precision"]:.4f} | '
            f'{row["task5_fmt_average_precision"]:.4f} | '
            f'**{row["fmt_minus_raw_pca_average_precision"]:+.4f}** |'
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path):
    # First produce the complete bias-controlled Task5 Raw/Raw-wide/Raw-PCA/FMT table.
    task3_table.run(config_path)
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"])
    with (output_dir / "paper_table.csv").open(newline="", encoding="utf-8") as handle:
        task5_summary = {row["dataset"]: row for row in csv.DictReader(handle)}

    device_name = str(spec.get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    fixed_rows, scale_rows = [], []
    for group in spec["groups"]:
        ordinals = list(range(int(group["expected_slices"])))
        for dataset in group["datasets"]:
            source_dir = Path(group["source_cache_root"]) / dataset
            records = _load_dataset(
                source_dir,
                Path(group["label_cache_root"]) / dataset,
                spec["sampled_steps"],
                spec["fmt_subset"],
                ordinals,
                spec.get("fmt_gram_num_freq", 6),
                group["expected_slices"],
            )
            pooled = _stack_split(records, ordinals)
            scale_ids, scale_names = _load_scale_ids(source_dir, group["expected_slices"])
            if any(len(ids) != len(record[0]) for ids, record in zip(scale_ids, records)):
                raise RuntimeError(f"scale/sample mismatch for {dataset}")
            for seed_value in spec["seeds"]:
                seed = int(seed_value)
                fixed_paths = {
                    "fixed_raw_pca": _find_checkpoint(
                        group["fixed_scale_raw_pca_checkpoint_roots"],
                        f"{dataset}_raw_pca_residual_seed{seed}.pt",
                    ),
                    "fixed_fmt": _find_checkpoint(
                        group["fixed_scale_fmt_checkpoint_roots"],
                        f"{dataset}_raw_fmt_residual_seed{seed}.pt",
                    ),
                    "task5_raw_pca": _find_checkpoint(
                        group["raw_pca_checkpoint_roots"],
                        f"{dataset}_raw_pca_residual_seed{seed}.pt",
                    ),
                    "task5_fmt": _find_checkpoint(
                        group["fmt_checkpoint_roots"],
                        f"{dataset}_raw_fmt_residual_seed{seed}.pt",
                    ),
                }
                loaded = {}
                for variant, checkpoint_path in fixed_paths.items():
                    model, checkpoint = _load_residual(
                        checkpoint_path, pooled[1].shape[1], device
                    )
                    loaded[variant] = model, checkpoint
                    targets, _, metrics = _evaluate_residual(
                        model, checkpoint, pooled, spec["batch_size"], seed, device
                    )
                    fixed_rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "sample_count": len(targets),
                        **metrics,
                        "checkpoint": str(checkpoint_path),
                    })
                for scale_id, scale_name in enumerate(scale_names):
                    parts = [
                        _subset_record(record, ids == scale_id)
                        for record, ids in zip(records, scale_ids)
                    ]
                    split = tuple(
                        np.concatenate([part[index] for part in parts], axis=0)
                        for index in range(3)
                    )
                    for variant, (model, checkpoint) in loaded.items():
                        targets, _, metrics = _evaluate_residual(
                            model, checkpoint, split,
                            spec["batch_size"], seed, device,
                        )
                        scale_rows.append({
                            "dataset": dataset,
                            "seed": seed,
                            "scale_id": scale_id,
                            "scale_name": scale_name,
                            "variant": variant,
                            "sample_count": len(targets),
                            "positive_fraction": float(targets.mean()),
                            **metrics,
                        })
                print(f"Task5 transfer/scales evaluated {dataset} seed={seed}", flush=True)
    _write_csv(output_dir / "fixed_scale_transfer_per_run.csv", fixed_rows)
    _write_csv(output_dir / "per_scale.csv", scale_rows)

    comparison = []
    for dataset in sorted(task5_summary):
        source = task5_summary[dataset]
        fixed_raw_pca_f1, fixed_raw_pca_f1_std = _mean_std(
            fixed_rows, dataset, "fixed_raw_pca", "f1"
        )
        fixed_fmt_f1, fixed_fmt_f1_std = _mean_std(
            fixed_rows, dataset, "fixed_fmt", "f1"
        )
        fixed_raw_pca_ap, fixed_raw_pca_ap_std = _mean_std(
            fixed_rows, dataset, "fixed_raw_pca", "average_precision"
        )
        fixed_fmt_ap, fixed_fmt_ap_std = _mean_std(
            fixed_rows, dataset, "fixed_fmt", "average_precision"
        )
        task5_raw_pca_f1 = float(source["mean_raw_pca_residual_f1"])
        task5_fmt_f1 = float(source["mean_raw_fmt_residual_f1"])
        task5_raw_pca_ap = float(source["mean_raw_pca_residual_average_precision"])
        task5_fmt_ap = float(source["mean_raw_fmt_residual_average_precision"])
        comparison.append({
            "dataset": dataset,
            "fixed_raw_pca_f1": fixed_raw_pca_f1,
            "fixed_raw_pca_f1_std": fixed_raw_pca_f1_std,
            "fixed_fmt_f1": fixed_fmt_f1,
            "fixed_fmt_f1_std": fixed_fmt_f1_std,
            "task5_raw_pca_f1": task5_raw_pca_f1,
            "task5_fmt_f1": task5_fmt_f1,
            "fmt_minus_raw_pca_f1": task5_fmt_f1 - task5_raw_pca_f1,
            "task5_minus_fixed_fmt_f1": task5_fmt_f1 - fixed_fmt_f1,
            "fixed_raw_pca_average_precision": fixed_raw_pca_ap,
            "fixed_raw_pca_average_precision_std": fixed_raw_pca_ap_std,
            "fixed_fmt_average_precision": fixed_fmt_ap,
            "fixed_fmt_average_precision_std": fixed_fmt_ap_std,
            "task5_raw_pca_average_precision": task5_raw_pca_ap,
            "task5_fmt_average_precision": task5_fmt_ap,
            "fmt_minus_raw_pca_average_precision": task5_fmt_ap - task5_raw_pca_ap,
            "task5_minus_fixed_fmt_average_precision": task5_fmt_ap - fixed_fmt_ap,
        })
    _write_csv(output_dir / "task5_comparison_table.csv", comparison)
    _write_comparison_markdown(comparison, output_dir / "task5_comparison_table.md")
    audit = {
        "experiment": spec["experiment"],
        "training_scale_count": 18,
        "validation_scale_count": 6,
        "confirmation_scale_count": 9,
        "confirmation_scale_tuples_seen_during_training": False,
        "fixed_output_shape": [7, int(spec["sampled_steps"]), 3],
        "dataset_count": len(comparison),
        "positive_f1_vs_raw_pca": sum(
            row["fmt_minus_raw_pca_f1"] > 0 for row in comparison
        ),
        "positive_ap_vs_raw_pca": sum(
            row["fmt_minus_raw_pca_average_precision"] > 0 for row in comparison
        ),
        "positive_f1_vs_fixed_fmt_transfer": sum(
            row["task5_minus_fixed_fmt_f1"] > 0 for row in comparison
        ),
    }
    (output_dir / "task5_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    return output_dir / "task5_comparison_table.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mainExp_Task5_3D_1.1_evaluate.yaml")
    args = parser.parse_args()
    run(args.config)
