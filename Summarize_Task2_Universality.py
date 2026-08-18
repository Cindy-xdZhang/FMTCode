"""Aggregate Task2 universality results with flow-level paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon


def summarize(result_root, output_dir):
    result_root = Path(result_root); output_dir = Path(output_dir)
    summaries = []
    seed_rows = []
    for summary_path in sorted(result_root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(payload)
        with (summary_path.parent / "runs.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        aggregate = [row for row in rows if row["scope"] == "all_test"]
        raw = {row["training_seed"]: float(row["f1"]) for row in aggregate
               if row["variant"] == "raw_vae"}
        fmt = {row["training_seed"]: float(row["f1"]) for row in aggregate
               if row["variant"] == "fmt_vae"}
        if raw.keys() != fmt.keys():
            raise RuntimeError(f"unpaired VAE seeds for {payload['dataset']}")
        for seed in sorted(raw):
            seed_rows.append({"dataset": payload["dataset"], "training_seed": int(seed),
                              "raw_vae_f1": raw[seed], "fmt_vae_f1": fmt[seed],
                              "improvement": fmt[seed] - raw[seed]})
    if not summaries:
        raise RuntimeError(f"no result summaries under {result_root}")
    flow_rows = []
    for payload in summaries:
        summary = payload["summary"]
        flow_rows.append({
            "dataset": payload["dataset"],
            "raw_vae_mean_f1": summary["raw_vae"]["mean_f1"],
            "raw_vae_std_f1": summary["raw_vae"]["std_f1"],
            "fmt_vae_mean_f1": summary["fmt_vae"]["mean_f1"],
            "fmt_vae_std_f1": summary["fmt_vae"]["std_f1"],
            "improvement": summary["fmt_vae_minus_raw_vae"],
            "raw_direct_f1": summary["raw_direct"]["mean_f1"],
            "fmt_direct_f1": summary["fmt_direct"]["mean_f1"],
        })
    improvements = np.asarray([row["improvement"] for row in flow_rows])
    by_name = {row["dataset"]: row["improvement"] for row in flow_rows}
    required = {"cylinder3d", "tangaroa", "deltaWing_LBM", "deltaWing_resampled", "f22raptor"}
    missing = required.difference(by_name)
    if missing:
        raise RuntimeError(f"missing physical-flow results: {sorted(missing)}")
    # The two deltaWing files are related variants of one physical family.  The
    # channel entry is a synthetic objectivity control generated from one steady
    # field, not an independent unsteady flow.  Collapse/exclude them accordingly.
    family_improvements = np.asarray([
        by_name["cylinder3d"],
        by_name["tangaroa"],
        np.mean([by_name["deltaWing_LBM"], by_name["deltaWing_resampled"]]),
        by_name["f22raptor"],
    ])
    statistic, pvalue = wilcoxon(
        family_improvements, alternative="greater", method="exact"
    )
    rng = np.random.default_rng(7068)
    bootstrap = np.asarray([
        rng.choice(improvements, size=len(improvements), replace=True).mean()
        for _ in range(20000)
    ])
    report = {
        "descriptive_dataset_count": len(flow_rows),
        "all_datasets_positive": bool((improvements > 0).all()),
        "mean_improvement": float(improvements.mean()),
        "median_improvement": float(np.median(improvements)),
        "bootstrap_95_ci_for_mean": [float(v) for v in np.percentile(bootstrap, [2.5, 97.5])],
        "conservative_physical_family_test": {
            "independent_unit": "physical flow family",
            "families": ["cylinder3d", "tangaroa", "deltaWing", "f22raptor"],
            "deltaWing_improvement": float(family_improvements[2]),
            "wilcoxon_one_sided_greater": {
                "statistic": float(statistic), "pvalue": float(pvalue)
            },
        },
        "warning": (
            "Six dataset entries are descriptive, not six independent physical flows. "
            "channel is a synthetic unsteady objectivity control generated from a steady VTK field; "
            "the two deltaWing entries are related resolutions/sources and are collapsed into one "
            "family for the conservative Wilcoxon test. Seeds are paired repeats, not independent units."
        ),
        "flows": flow_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "universality_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for filename, table in (("flow_results.csv", flow_rows), ("paired_seed_results.csv", seed_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)

    names = [row["dataset"] for row in flow_rows]
    x = np.arange(len(names)); width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x - width / 2, [r["raw_vae_mean_f1"] for r in flow_rows], width,
                yerr=[r["raw_vae_std_f1"] for r in flow_rows], label="Raw+VAE",
                color="#577590", capsize=3)
    axes[0].bar(x + width / 2, [r["fmt_vae_mean_f1"] for r in flow_rows], width,
                yerr=[r["fmt_vae_std_f1"] for r in flow_rows], label="FMT+VAE",
                color="#43aa8b", capsize=3)
    axes[0].set_xticks(x, names, rotation=18, ha="right")
    axes[0].set(ylabel="Held-out-timeslice F1", ylim=(0, 1), title="Task2 across 3D flows")
    axes[0].legend()
    colors = ["#2a9d8f" if value > 0 else "#e76f51" for value in improvements]
    bars = axes[1].bar(x, improvements, color=colors)
    axes[1].bar_label(bars, fmt="%+.3f", padding=3)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, names, rotation=18, ha="right")
    axes[1].set(ylabel="FMT+VAE minus Raw+VAE F1", title="Per-flow paired improvement")
    fig.tight_layout(); fig.savefig(output_dir / "universality_f1.png", dpi=220); plt.close(fig)
    print(json.dumps(report, indent=2)); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="outputs/Verify_Task2Universality_1.1/results")
    parser.add_argument("--output", default="outputs/Verify_Task2Universality_1.1/summary")
    args = parser.parse_args(); summarize(args.result_root, args.output)
