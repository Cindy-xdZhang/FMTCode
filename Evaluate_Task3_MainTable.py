"""Evaluate the frozen, bias-controlled Task3 3D paper experiment.

Raw model selection uses development-validation Average Precision only.
Confirmation labels are loaded only after the Raw method has been frozen per
physical family.  The script evaluates Raw, Raw-wide, a structure-matched
Raw-PCA residual, and the FMT residual on fresh confirmation slices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_baseline,
    _evaluate_residual,
    _find_checkpoint,
    _load_baseline,
    _load_residual,
    _write_csv,
)
from Verify_Task3_FMTClassifier import _load_dataset, _stack_split


RAW_VARIANTS = ("raw", "raw_wide", "raw_pca_residual")
ALL_VARIANTS = (*RAW_VARIANTS, "raw_fmt_residual")


def _read_csvs(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _family_map(spec):
    result = {}
    for family, datasets in spec["families"].items():
        for dataset in datasets:
            if dataset in result:
                raise ValueError(f"dataset {dataset} belongs to two families")
            result[dataset] = family
    expected = {
        dataset for group in spec["groups"] for dataset in group["datasets"]
    }
    if set(result) != expected:
        raise ValueError(
            f"family coverage mismatch: missing={expected - set(result)}, "
            f"extra={set(result) - expected}"
        )
    return result


def _select_raw_by_validation(spec, families):
    """Select one deployable Raw-only method per physical family."""
    rows = []
    expected_seeds = {int(value) for value in spec["seeds"]}
    for group in spec["groups"]:
        rows.extend(_read_csvs(group["development_result_csvs"]))
    filtered = [
        row for row in rows
        if row["variant"] in RAW_VARIANTS
        and int(row["seed"]) in expected_seeds
        and row["dataset"] in families
    ]
    expected_keys = {
        (dataset, variant, seed)
        for dataset in families
        for variant in RAW_VARIANTS
        for seed in expected_seeds
    }
    actual_keys = {
        (row["dataset"], row["variant"], int(row["seed"]))
        for row in filtered
    }
    if actual_keys != expected_keys:
        raise RuntimeError(
            "development result coverage mismatch: "
            f"missing={sorted(expected_keys - actual_keys)[:10]}, "
            f"extra={sorted(actual_keys - expected_keys)[:10]}"
        )
    selection_rows = []
    selected = {}
    preference = {name: -index for index, name in enumerate(RAW_VARIANTS)}
    for family in spec["families"]:
        family_rows = [row for row in filtered if families[row["dataset"]] == family]
        scores = {
            variant: float(np.mean([
                float(row["validation_average_precision"])
                for row in family_rows if row["variant"] == variant
            ]))
            for variant in RAW_VARIANTS
        }
        winner = max(RAW_VARIANTS, key=lambda name: (scores[name], preference[name]))
        selected[family] = winner
        selection_rows.append({
            "family": family,
            "selected_raw_variant": winner,
            **{f"validation_ap_{name}": scores[name] for name in RAW_VARIANTS},
            "selection_used_confirmation": 0,
        })
    return selected, selection_rows


def _checkpoint_paths(group, dataset, seed):
    return {
        "raw": _find_checkpoint(
            group["baseline_checkpoint_roots"],
            f"{dataset}_raw_seed{seed}.pt",
        ),
        "raw_wide": _find_checkpoint(
            group["baseline_checkpoint_roots"],
            f"{dataset}_raw_wide_seed{seed}.pt",
        ),
        "raw_pca_residual": _find_checkpoint(
            group["raw_pca_checkpoint_roots"],
            f"{dataset}_raw_pca_residual_seed{seed}.pt",
        ),
        "raw_fmt_residual": _find_checkpoint(
            group["fmt_checkpoint_roots"],
            f"{dataset}_raw_fmt_residual_seed{seed}.pt",
        ),
    }


def _load_variant(variant, path, fmt_dim, device):
    if variant in {"raw", "raw_wide"}:
        return _load_baseline(path, fmt_dim, device)
    return _load_residual(path, fmt_dim, device)


def _evaluate_loaded(variant, model, checkpoint, split, batch_size, seed, device):
    if variant in {"raw", "raw_wide"}:
        targets, _, metrics = _evaluate_baseline(
            model, checkpoint, split, batch_size, seed, device
        )
        alpha = 0.0
    else:
        targets, _, metrics = _evaluate_residual(
            model, checkpoint, split, batch_size, seed, device
        )
        alpha = float(checkpoint["alpha"])
    return targets, metrics, float(checkpoint["threshold"]), alpha


def _bootstrap_ci(values, seed, draws=20000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    samples = values[rng.integers(0, len(values), size=(int(draws), len(values)))]
    means = samples.mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def _summarise(pooled, per_slice, selected_raw, families, bootstrap_seed):
    by_key = {
        (row["dataset"], int(row["seed"]), row["variant"]): row
        for row in pooled
    }
    datasets = sorted({row["dataset"] for row in pooled})
    seeds = sorted({int(row["seed"]) for row in pooled})
    summary = []
    for dataset in datasets:
        family = families[dataset]
        baseline = selected_raw[family]
        item = {
            "dataset": dataset,
            "family": family,
            "selected_raw_variant": baseline,
            "seed_count": len(seeds),
            "confirmation_slice_count": len({
                int(row["ordinal"]) for row in per_slice
                if row["dataset"] == dataset
            }),
        }
        for metric in ("f1", "average_precision"):
            for variant in ALL_VARIANTS:
                values = np.asarray([
                    float(by_key[(dataset, seed, variant)][metric])
                    for seed in seeds
                ])
                item[f"mean_{variant}_{metric}"] = float(values.mean())
                item[f"std_{variant}_{metric}"] = float(values.std(ddof=1))
            gains = np.asarray([
                float(by_key[(dataset, seed, "raw_fmt_residual")][metric])
                - float(by_key[(dataset, seed, baseline)][metric])
                for seed in seeds
            ])
            low, high = _bootstrap_ci(
                gains, int(bootstrap_seed) + datasets.index(dataset) * 13
                + (0 if metric == "f1" else 1),
            )
            item[f"mean_gain_{metric}"] = float(gains.mean())
            item[f"std_gain_{metric}"] = float(gains.std(ddof=1))
            item[f"gain_{metric}_ci95_low"] = low
            item[f"gain_{metric}_ci95_high"] = high
            item[f"positive_seed_count_{metric}"] = int((gains > 0).sum())
            slice_fmt = [
                float(row[metric]) for row in per_slice
                if row["dataset"] == dataset
                and row["variant"] == "raw_fmt_residual"
            ]
            slice_raw = {
                (int(row["seed"]), int(row["ordinal"])): float(row[metric])
                for row in per_slice
                if row["dataset"] == dataset and row["variant"] == baseline
            }
            slice_gains = [
                float(row[metric]) - slice_raw[(int(row["seed"]), int(row["ordinal"]))]
                for row in per_slice
                if row["dataset"] == dataset
                and row["variant"] == "raw_fmt_residual"
            ]
            item[f"slice_macro_raw_fmt_{metric}"] = float(np.mean(slice_fmt))
            item[f"slice_macro_gain_{metric}"] = float(np.mean(slice_gains))
            raw_pca_gains = np.asarray([
                float(by_key[(dataset, seed, "raw_fmt_residual")][metric])
                - float(by_key[(dataset, seed, "raw_pca_residual")][metric])
                for seed in seeds
            ])
            raw_pca_low, raw_pca_high = _bootstrap_ci(
                raw_pca_gains, int(bootstrap_seed) + datasets.index(dataset) * 17
                + (2 if metric == "f1" else 3),
            )
            item[f"mean_fmt_minus_raw_pca_{metric}"] = float(raw_pca_gains.mean())
            item[f"std_fmt_minus_raw_pca_{metric}"] = float(raw_pca_gains.std(ddof=1))
            item[f"fmt_minus_raw_pca_{metric}_ci95_low"] = raw_pca_low
            item[f"fmt_minus_raw_pca_{metric}_ci95_high"] = raw_pca_high
            item[f"positive_seed_count_fmt_minus_raw_pca_{metric}"] = int(
                (raw_pca_gains > 0).sum()
            )
        summary.append(item)
    return summary


def _family_summary(dataset_summary):
    rows = []
    families = sorted({row["family"] for row in dataset_summary})
    for family in families:
        selected = [row for row in dataset_summary if row["family"] == family]
        rows.append({
            "family": family,
            "dataset_count": len(selected),
            "selected_raw_variant": selected[0]["selected_raw_variant"],
            "mean_gain_f1": float(np.mean([
                row["mean_gain_f1"] for row in selected
            ])),
            "mean_gain_average_precision": float(np.mean([
                row["mean_gain_average_precision"] for row in selected
            ])),
            "mean_fmt_minus_raw_pca_f1": float(np.mean([
                row["mean_fmt_minus_raw_pca_f1"] for row in selected
            ])),
            "mean_fmt_minus_raw_pca_average_precision": float(np.mean([
                row["mean_fmt_minus_raw_pca_average_precision"] for row in selected
            ])),
        })
    return rows


def _paper_markdown(summary, family_summary, path):
    lines = [
        "# Task3 3D 论文主表（bias-controlled confirmation）",
        "",
        "主基线在 confirmation 之前按 physical-family 的 development-validation "
        "Average Precision 冻结。Raw+FMT residual 的 epoch 只按自身 validation "
        "Average Precision 选择；residual alpha 固定为 1.0。",
        "",
        "| Flow | 冻结Raw基线 | Raw基线 F1 | Raw+Raw-PCA F1 | Raw+FMT F1 | FMT−Raw-PCA F1 | Raw基线 AP | Raw+Raw-PCA AP | Raw+FMT AP | FMT−Raw-PCA AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        baseline = row["selected_raw_variant"]
        lines.append(
            f'| {row["dataset"]} | {baseline} | '
            f'{row[f"mean_{baseline}_f1"]:.4f}±{row[f"std_{baseline}_f1"]:.4f} | '
            f'{row["mean_raw_pca_residual_f1"]:.4f}±{row["std_raw_pca_residual_f1"]:.4f} | '
            f'{row["mean_raw_fmt_residual_f1"]:.4f}±{row["std_raw_fmt_residual_f1"]:.4f} | '
            f'**{row["mean_fmt_minus_raw_pca_f1"]:+.4f}** | '
            f'{row[f"mean_{baseline}_average_precision"]:.4f}±{row[f"std_{baseline}_average_precision"]:.4f} | '
            f'{row["mean_raw_pca_residual_average_precision"]:.4f}±{row["std_raw_pca_residual_average_precision"]:.4f} | '
            f'{row["mean_raw_fmt_residual_average_precision"]:.4f}±{row["std_raw_fmt_residual_average_precision"]:.4f} | '
            f'**{row["mean_fmt_minus_raw_pca_average_precision"]:+.4f}** |'
        )
    positive_f1 = sum(row["mean_gain_f1"] > 0 for row in summary)
    positive_ap = sum(row["mean_gain_average_precision"] > 0 for row in summary)
    family_f1 = sum(row["mean_gain_f1"] > 0 for row in family_summary)
    family_ap = sum(row["mean_gain_average_precision"] > 0 for row in family_summary)
    raw_pca_positive_f1 = sum(
        row["mean_fmt_minus_raw_pca_f1"] > 0 for row in summary
    )
    raw_pca_positive_ap = sum(
        row["mean_fmt_minus_raw_pca_average_precision"] > 0 for row in summary
    )
    lines.extend([
        "",
        f"条目方向：F1 `{positive_f1}/{len(summary)}`，AP `{positive_ap}/{len(summary)}`；"
        f"physical-family 方向：F1 `{family_f1}/{len(family_summary)}`，"
        f"AP `{family_ap}/{len(family_summary)}`。",
        f"相对 Raw-PCA residual：F1 `{raw_pca_positive_f1}/{len(summary)}`，"
        f"AP `{raw_pca_positive_ap}/{len(summary)}`。",
        "",
        "误差为 5 个训练随机种子的 sample standard deviation。95% confidence "
        "interval 与逐时间片结果见同目录机器表。",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    families = _family_map(spec)
    selected_raw, selection_rows = _select_raw_by_validation(spec, families)
    _write_csv(output_dir / "raw_method_selection.csv", selection_rows)

    device_name = spec.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    pooled_rows, slice_rows = [], []
    for group in spec["groups"]:
        ordinals = list(range(int(group["expected_slices"])))
        for dataset in group["datasets"]:
            records = _load_dataset(
                Path(group["source_cache_root"]) / dataset,
                Path(group["label_cache_root"]) / dataset,
                spec["sampled_steps"], spec["fmt_subset"], ordinals,
                spec.get("fmt_gram_num_freq", 6), group["expected_slices"],
            )
            pooled_split = _stack_split(records, ordinals)
            for seed_value in spec["seeds"]:
                seed = int(seed_value)
                paths = _checkpoint_paths(group, dataset, seed)
                for variant in ALL_VARIANTS:
                    model, checkpoint = _load_variant(
                        variant, paths[variant], pooled_split[1].shape[1], device
                    )
                    targets, metrics, threshold, alpha = _evaluate_loaded(
                        variant, model, checkpoint, pooled_split,
                        spec["batch_size"], seed, device,
                    )
                    pooled_rows.append({
                        "dataset": dataset, "family": families[dataset],
                        "seed": seed, "variant": variant,
                        "sample_count": len(targets),
                        "positive_fraction": float(targets.mean()),
                        "frozen_threshold": threshold, "frozen_alpha": alpha,
                        **metrics, "checkpoint": str(paths[variant]),
                    })
                    for ordinal in ordinals:
                        split = _stack_split(records, [ordinal])
                        slice_targets, slice_metrics, _, _ = _evaluate_loaded(
                            variant, model, checkpoint, split,
                            spec["batch_size"], seed, device,
                        )
                        slice_rows.append({
                            "dataset": dataset, "family": families[dataset],
                            "ordinal": ordinal, "seed": seed, "variant": variant,
                            "sample_count": len(slice_targets),
                            "positive_fraction": float(slice_targets.mean()),
                            **slice_metrics,
                        })
                print(f"evaluated {dataset} seed={seed}", flush=True)
    _write_csv(output_dir / "per_run.csv", pooled_rows)
    _write_csv(output_dir / "per_slice.csv", slice_rows)
    summary = _summarise(
        pooled_rows, slice_rows, selected_raw, families,
        int(spec.get("bootstrap_seed", 7068)),
    )
    family_summary = _family_summary(summary)
    _write_csv(output_dir / "paper_table.csv", summary)
    _write_csv(output_dir / "family_summary.csv", family_summary)
    _paper_markdown(summary, family_summary, output_dir / "paper_table.md")
    audit = {
        "experiment": spec["experiment"],
        "confirmation_data_was_not_used_for_selection": True,
        "raw_selection": "physical-family mean development-validation AP",
        "fmt_checkpoint_selection": "own development-validation AP",
        "residual_alpha": 1.0,
        "training_seed_count": len(spec["seeds"]),
        "dataset_count": len(summary),
        "family_count": len(family_summary),
        "positive_f1_datasets": sum(row["mean_gain_f1"] > 0 for row in summary),
        "positive_ap_datasets": sum(
            row["mean_gain_average_precision"] > 0 for row in summary
        ),
        "positive_f1_datasets_vs_raw_pca": sum(
            row["mean_fmt_minus_raw_pca_f1"] > 0 for row in summary
        ),
        "positive_ap_datasets_vs_raw_pca": sum(
            row["mean_fmt_minus_raw_pca_average_precision"] > 0 for row in summary
        ),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    return output_dir / "paper_table.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
