"""Merge and audit the three-seed, all-flow Task3 universality experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("test_f1", "test_average_precision")
VARIANTS = ("raw", "raw_wide", "raw_fmt_residual")
EXPECTED_SEEDS = {20, 21, 22}
FAMILIES = {
    "half-cylinder": {"cylinder3d", "halfcylinderRe640", "halfcylinderRe6400"},
    "Tangaroa": {"tangaroa"},
    "delta-wing": {"deltaWing_resampled", "deltaWing_LBM"},
    "F-22 Raptor": {"f22raptor"},
    "channel + Killing observer": {"channel"},
}


def _read(paths, allowed_variants):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frames.append(frame[frame["variant"].isin(allowed_variants)])
    return pd.concat(frames, ignore_index=True)


def summarize(baseline_csvs, residual_csvs, output_dir):
    baselines = _read(baseline_csvs, {"raw", "raw_wide"})
    residuals = _read(residual_csvs, {"raw_fmt_residual"})
    rows = pd.concat((baselines, residuals), ignore_index=True)
    if rows.duplicated(["dataset", "variant", "seed"]).any():
        raise ValueError("duplicate dataset/variant/seed result")
    datasets = list(rows["dataset"].drop_duplicates())
    expected = {
        (dataset, variant, seed)
        for dataset in datasets for variant in VARIANTS for seed in EXPECTED_SEEDS
    }
    observed = set(zip(rows["dataset"], rows["variant"], rows["seed"]))
    if observed != expected:
        raise ValueError(
            f"incomplete experiment: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    if set().union(*FAMILIES.values()) != set(datasets):
        raise ValueError("flow-family declaration does not cover datasets exactly")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "all_runs.csv", index=False)
    summary = rows.groupby(["dataset", "variant"], sort=False)[list(METRICS)].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)

    pivot = rows.pivot(index=["dataset", "seed"], columns="variant")
    gains = []
    for dataset, seed in pivot.index:
        record = {"dataset": dataset, "seed": int(seed)}
        for metric in METRICS:
            for baseline in ("raw", "raw_wide"):
                record[f"residual_minus_{baseline}_{metric}"] = float(
                    pivot.loc[(dataset, seed), (metric, "raw_fmt_residual")]
                    - pivot.loc[(dataset, seed), (metric, baseline)]
                )
        gains.append(record)
    gains = pd.DataFrame(gains)
    gain_columns = [column for column in gains if column.startswith("residual_minus_")]
    gains["strict_seed_pass"] = (gains[gain_columns] > 0.0).all(axis=1)
    gains.to_csv(output_dir / "paired_gains.csv", index=False)

    audit = []
    for dataset in datasets:
        subset = gains[gains["dataset"] == dataset]
        audit.append({
            "dataset": dataset,
            "seeds": len(subset),
            "all_seed_metric_comparisons_positive": bool(
                (subset[gain_columns] > 0.0).all().all()
            ),
            **{f"min_{column}": float(subset[column].min()) for column in gain_columns},
            **{f"mean_{column}": float(subset[column].mean()) for column in gain_columns},
        })
    audit = pd.DataFrame(audit)
    audit.to_csv(output_dir / "strict_audit.csv", index=False)

    family_rows = []
    for family, members in FAMILIES.items():
        subset = audit[audit["dataset"].isin(members)]
        family_rows.append({
            "flow_family": family,
            "datasets": ", ".join(sorted(members)),
            "dataset_count": len(members),
            "all_datasets_pass": bool(
                subset["all_seed_metric_comparisons_positive"].all()
            ),
        })
    family_audit = pd.DataFrame(family_rows)
    family_audit.to_csv(output_dir / "family_audit.csv", index=False)

    labels = {
        "raw": "Raw", "raw_wide": "Raw-wide",
        "raw_fmt_residual": "Raw + FMT residual",
    }
    colors = {"raw": "#6b7280", "raw_wide": "#a3a3a3",
              "raw_fmt_residual": "#2563eb"}
    display_names = {
        "cylinder3d": "Re160", "halfcylinderRe640": "Re640",
        "halfcylinderRe6400": "Re6400", "tangaroa": "Tangaroa",
        "deltaWing_resampled": "Delta (resampled)",
        "deltaWing_LBM": "Delta (LBM)", "f22raptor": "F-22",
        "channel": "Channel observer",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    x = np.arange(len(datasets), dtype=float)
    width = 0.25
    for axis, metric, title in zip(
        axes, METRICS, ("Held-out F1", "Held-out average precision"),
    ):
        for offset, variant in enumerate(VARIANTS):
            grouped = rows[rows["variant"] == variant].groupby("dataset")[metric]
            means = grouped.mean().reindex(datasets).to_numpy()
            stds = grouped.std().reindex(datasets).to_numpy()
            axis.bar(
                x + (offset - 1) * width, means, width, yerr=stds,
                capsize=3, color=colors[variant], label=labels[variant],
                edgecolor="black", linewidth=0.45,
            )
        axis.set_xticks(x, [display_names[name] for name in datasets],
                        rotation=25, ha="right")
        axis.set_ylim(0.45, 1.0)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Task3 across tested 3D flows: mean ± sample standard deviation (3 seeds)"
    )
    figure.savefig(output_dir / "task3_universality.png", dpi=220)
    plt.close(figure)

    if not audit["all_seed_metric_comparisons_positive"].all():
        failed = audit.loc[
            ~audit["all_seed_metric_comparisons_positive"], "dataset"
        ].tolist()
        raise RuntimeError(f"strict universality criterion failed for {failed}")
    if not family_audit["all_datasets_pass"].all():
        raise RuntimeError("at least one independent flow family failed")
    return summary, gains, audit, family_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", action="append", required=True)
    parser.add_argument("--residual-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = summarize(
        args.baseline_csv, args.residual_csv, args.output_dir
    )
    for title, frame in zip(
        ("Summary", "Paired gains", "Strict audit", "Family audit"), outputs
    ):
        print(f"\n{title}\n{frame.to_string(index=False)}")
