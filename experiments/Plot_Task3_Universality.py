"""Plot the frozen mainExp_Task3Universality_2.2 confirmation gains."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "cylinder3d": "Cylinder Re160",
    "halfcylinderRe640": "Cylinder Re640",
    "halfcylinderRe6400": "Cylinder Re6400",
    "tangaroa": "Tangaroa",
    "deltaWing_resampled": "Delta wing (resampled)",
    "deltaWing_LBM": "Delta wing (original)",
    "f22raptor": "F-22 Raptor",
    "channel": "Channel + observer",
}


def plot_gains(summary_csv: str | Path, output_path: str | Path, minimum_gain: float = 0.02):
    summary_csv = Path(summary_csv)
    output_path = Path(output_path)
    frame = pd.read_csv(summary_csv)
    required = {
        "dataset",
        "mean_gain_f1",
        "mean_gain_average_precision",
        "passes_minimum_gain",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing summary columns: {sorted(missing)}")
    unknown = set(frame["dataset"]) - set(DISPLAY_NAMES)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    if frame["dataset"].duplicated().any():
        raise ValueError("each dataset must occur exactly once")

    frame = frame.copy()
    frame["display_name"] = frame["dataset"].map(DISPLAY_NAMES)
    # Sort by the weaker of the two gains so the boundary cases are explicit.
    frame["minimum_metric_gain"] = frame[
        ["mean_gain_f1", "mean_gain_average_precision"]
    ].min(axis=1)
    frame = frame.sort_values("minimum_metric_gain", ascending=True)

    y = np.arange(len(frame), dtype=float)
    height = 0.36
    figure, axis = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    f1 = frame["mean_gain_f1"].to_numpy()
    ap = frame["mean_gain_average_precision"].to_numpy()
    axis.barh(y - height / 2, f1, height, label="F1 gain", color="#2563eb")
    axis.barh(
        y + height / 2,
        ap,
        height,
        label="Average Precision gain",
        color="#f59e0b",
    )
    axis.axvline(
        minimum_gain,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.4,
        label=f"Frozen minimum gain = {minimum_gain:.02f}",
    )
    axis.set_yticks(y, frame["display_name"])
    axis.set_xlabel("Raw + FMT residual minus stronger Raw baseline (3-seed mean)")
    axis.set_title("Task3 confirmation on new seed times: all tested datasets pass")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")

    x_padding = max(float(max(f1.max(), ap.max())) * 0.012, 0.004)
    for values, offsets in ((f1, y - height / 2), (ap, y + height / 2)):
        for value, y_position in zip(values, offsets):
            axis.text(
                value + x_padding,
                y_position,
                f"+{value:.3f}",
                va="center",
                ha="left",
                fontsize=8.5,
            )
    axis.set_xlim(0.0, max(float(max(f1.max(), ap.max())) * 1.16, minimum_gain * 1.5))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240)
    plt.close(figure)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default=(
            "outputs/mainExp_Task3Universality_2.2/final_confirmation/"
            "per_dataset_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/mainExp_Task3Universality_2.2/final_confirmation/"
            "universality_gains.png"
        ),
    )
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    args = parser.parse_args()
    plotted = plot_gains(args.summary_csv, args.output, args.minimum_gain)
    print(plotted[["dataset", "mean_gain_f1", "mean_gain_average_precision"]].to_string(index=False))
    print(f"Saved {args.output}")
