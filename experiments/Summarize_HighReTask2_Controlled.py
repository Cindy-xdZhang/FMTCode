"""Consolidate the frozen controlled high-Re Task2 comparison and plot it."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ["halfcylinderRe640", "halfcylinderRe6400"]
METHODS = ["Raw+PCA direct", "FMT direct", "Raw+VAE", "FMT+VAE"]


def run(output_dir="outputs/Verify_HighReTask2_Controlled_1.1"):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    direct = pd.read_csv("outputs/Verify_HighReTask2_1.5/direct.csv")
    raw = pd.read_csv(
        "outputs/Verify_HighReVAE_Controlled_1.1_heldout/screening_runs.csv"
    )
    fmt = pd.read_csv("outputs/Verify_HighReTask2_1.5/vae_runs.csv")
    fmt = fmt[fmt["method"] == "FMT+VAE"]

    rows = []
    direct_names = {"Raw+PCA direct": "Raw+PCA direct",
                    "FMT real-neighbor direct": "FMT direct"}
    for _, row in direct.iterrows():
        rows.append({"dataset": row["dataset"], "method": direct_names[row["method"]],
                     "mean_f1": row["f1"], "std_f1": 0.0,
                     "min_f1": row["f1"], "max_f1": row["f1"], "runs": 1})
    for method, frame in (("Raw+VAE", raw), ("FMT+VAE", fmt)):
        for dataset, group in frame.groupby("dataset"):
            values = group.sort_values("training_seed")["f1"].to_numpy()
            rows.append({"dataset": dataset, "method": method,
                         "mean_f1": values.mean(), "std_f1": values.std(ddof=0),
                         "min_f1": values.min(), "max_f1": values.max(),
                         "runs": len(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False)

    gains = []
    for dataset in DATASETS:
        raw_values = raw[raw.dataset == dataset].sort_values("training_seed")["f1"].to_numpy()
        fmt_values = fmt[fmt.dataset == dataset].sort_values("training_seed")["f1"].to_numpy()
        for seed, raw_f1, fmt_f1 in zip([7068, 7069, 7070], raw_values, fmt_values):
            gains.append({"dataset": dataset, "training_seed": seed,
                          "raw_vae_f1": raw_f1, "fmt_vae_f1": fmt_f1,
                          "fmt_gain": fmt_f1 - raw_f1})
    gain_frame = pd.DataFrame(gains); gain_frame.to_csv(output / "paired_gains.csv", index=False)

    payload = {
        "experiment": "Verify_HighReTask2_Controlled_1.1",
        "selection_split": "slices 1-6 train, 7-8 validation",
        "confirmation_split": "slices 1-8 train, 9-10 held-out test",
        "summary": rows,
        "paired_gain": {
            dataset: {
                "values": gain_frame[gain_frame.dataset == dataset]["fmt_gain"].tolist(),
                "mean": float(gain_frame[gain_frame.dataset == dataset]["fmt_gain"].mean()),
                "min": float(gain_frame[gain_frame.dataset == dataset]["fmt_gain"].min()),
            } for dataset in DATASETS
        },
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    colors = ["#8da0cb", "#66c2a5", "#fc8d62", "#1b9e77"]
    for axis, dataset, title in zip(axes, DATASETS, ("Half-cylinder Re=640", "Half-cylinder Re=6400")):
        selected = summary[summary.dataset == dataset].set_index("method").loc[METHODS]
        x = np.arange(len(METHODS))
        axis.bar(x, selected.mean_f1, yerr=selected.std_f1, color=colors,
                 edgecolor="black", linewidth=0.6, capsize=4)
        axis.axhline(0.5, color="#444444", linestyle="--", linewidth=1)
        for index, value in enumerate(selected.mean_f1):
            axis.text(index, value + 0.018, f"{value:.3f}", ha="center", fontsize=9)
        axis.set_xticks(x, ["Raw+PCA\ndirect", "FMT\ndirect", "Raw+VAE", "FMT+VAE"])
        axis.set_title(title); axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Held-out F1")
    axes[0].set_ylim(0.0, 0.82)
    fig.suptitle("Controlled Task2: identical linear latent-2 beta-VAE")
    fig.tight_layout(); fig.savefig(output / "f1_comparison.png", dpi=220)
    plt.close(fig)
    return output


if __name__ == "__main__":
    run()
