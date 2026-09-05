"""Summarize paired multi-seed Task3 classifier experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["test_f1", "test_average_precision", "test_roc_auc",
           "test_balanced_accuracy"]
VARIANT_ORDER = ["raw", "raw_wide", "raw_fmt"]


def summarize(input_csv, output_dir=None):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir) if output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(input_csv)
    duplicated = rows.duplicated(["dataset", "variant", "seed"])
    if duplicated.any():
        raise ValueError("duplicate dataset/variant/seed rows")
    groups = rows.groupby(["dataset", "variant"], sort=False)
    summary_parts = []
    for metric in METRICS:
        values = groups[metric].agg(["mean", "std", "min", "max", "count"])
        values.columns = [f"{metric}_{column}" for column in values.columns]
        summary_parts.append(values)
    summary = pd.concat(summary_parts, axis=1).reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)

    pivot = rows.pivot(index=["dataset", "seed"], columns="variant")
    gains = []
    for dataset, seed in pivot.index:
        record = {"dataset": dataset, "seed": seed}
        for metric in METRICS:
            for baseline in ("raw", "raw_wide"):
                record[f"raw_fmt_minus_{baseline}_{metric}"] = float(
                    pivot.loc[(dataset, seed), (metric, "raw_fmt")]
                    - pivot.loc[(dataset, seed), (metric, baseline)]
                )
        gains.append(record)
    gains = pd.DataFrame(gains)
    gains.to_csv(output_dir / "paired_gains.csv", index=False)

    datasets = list(rows["dataset"].drop_duplicates())
    seed_count = int(rows["seed"].nunique())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    labels = {"raw": "Raw", "raw_wide": "Raw-wide", "raw_fmt": "Raw + FMT"}
    colors = {"raw": "#6b7280", "raw_wide": "#9ca3af", "raw_fmt": "#2563eb"}
    for axis, metric, title in zip(
        axes, ("test_f1", "test_average_precision"),
        ("Held-out F1", "Held-out average precision"),
    ):
        x = np.arange(len(datasets), dtype=float)
        width = 0.24
        for offset, variant in enumerate(VARIANT_ORDER):
            subset = rows[rows["variant"] == variant].groupby("dataset")[metric]
            means = subset.mean().reindex(datasets).to_numpy()
            stds = subset.std().reindex(datasets).to_numpy()
            error = stds if seed_count > 1 else None
            axis.bar(
                x + (offset - 1) * width, means, width, yerr=error, capsize=4,
                label=labels[variant], color=colors[variant], edgecolor="black",
                linewidth=0.5,
            )
        axis.set_xticks(x, [name.replace("halfcylinder", "") for name in datasets])
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")
    suffix = (f"mean ± sample standard deviation ({seed_count} seeds)"
              if seed_count > 1 else "single-seed sensitivity run")
    figure.suptitle(f"Task3: temporal test slices, {suffix}")
    figure.savefig(output_dir / "task3_classifier_comparison.png", dpi=220)
    plt.close(figure)
    return summary, gains


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary, gains = summarize(args.input_csv, args.output_dir)
    print(summary.to_string(index=False))
    print("\nPaired gains:\n", gains.to_string(index=False))
