"""Merge per-dataset OctVAE result directories into one comparison table.

``Run_Task1_OctVAE_3D_1_1.py`` writes one directory per ``--tag`` so that the
datasets can run concurrently without clobbering each other's ``runs.csv``.
This script concatenates those rows and rebuilds the headline comparison, the
per-readout summary and the seed spread from the merged set.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.Run_Task1_OctVAE_3D_1_1 import (  # noqa: E402
    METRIC_KEYS, build_comparison, comparison_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = set(METRIC_KEYS)


def _read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if key in NUMERIC:
                    parsed[key] = float(value)
                elif key == "seed":
                    parsed[key] = int(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def readout_summary(rows, config):
    """Macro means per method/readout, plus the train-seed spread."""
    summary = []
    datasets = [d["id"] for d in config["datasets"]]
    keys = sorted({(r["method"], r["readout"]) for r in rows})
    for method, readout in keys:
        per_dataset, per_seed_spread = [], []
        for dataset in datasets:
            values = [r for r in rows if r["dataset"] == dataset
                      and r["method"] == method and r["readout"] == readout]
            if not values:
                continue
            per_dataset.append(float(np.mean([v["f1"] for v in values])))
            train_seeds = sorted({v["train_seed"] for v in values})
            if len(train_seeds) > 1:
                means = [float(np.mean([v["f1"] for v in values
                                        if v["train_seed"] == seed]))
                         for seed in train_seeds]
                per_seed_spread.append(float(np.std(means)))
        if not per_dataset:
            continue
        entry = {
            "method": method, "readout": readout,
            "datasets": len(per_dataset),
            "macro_f1": float(np.mean(per_dataset)),
            "min_f1": float(np.min(per_dataset)),
            "max_f1": float(np.max(per_dataset)),
        }
        for key in ("ari", "nmi", "iou"):
            values = [float(np.mean([r[key] for r in rows if r["dataset"] == d
                                     and r["method"] == method
                                     and r["readout"] == readout]))
                      for d in datasets
                      if any(r["dataset"] == d and r["method"] == method
                             and r["readout"] == readout for r in rows)]
            entry[f"macro_{key}"] = float(np.mean(values))
        if per_seed_spread:
            entry["mean_train_seed_std"] = float(np.mean(per_seed_spread))
        summary.append(entry)
    return sorted(summary, key=lambda item: -item["macro_f1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pattern", default="outputs/exp_Task1_OctVAE_3D_1.1_*",
                        help="glob of per-dataset result directories")
    parser.add_argument("--output", default=None,
                        help="merged output directory (default: config result_dir)")
    arguments = parser.parse_args()

    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    pattern = Path(arguments.pattern)
    directories = ([Path(x) for x in sorted(glob.glob(arguments.pattern))]
                   if pattern.is_absolute() else sorted(ROOT.glob(arguments.pattern)))
    rows, sources = [], []
    for directory in directories:
        path = directory / "runs.csv"
        if not path.exists():
            continue
        rows.extend(_read_rows(path))
        try:
            sources.append(str(directory.relative_to(ROOT)))
        except ValueError:
            sources.append(str(directory))
    if not rows:
        raise SystemExit(f"no runs.csv found under {arguments.pattern}")

    present = {r["dataset"] for r in rows}
    config["datasets"] = [d for d in config["datasets"] if d["id"] in present]
    missing = present - {d["id"] for d in config["datasets"]}
    if missing:
        raise SystemExit(f"rows reference unconfigured datasets: {sorted(missing)}")

    output = Path(arguments.output or config["output"]["result_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    table = build_comparison(rows, config)
    summary = readout_summary(rows, config)

    _write_csv(output / "runs.csv", rows)
    _write_csv(output / "comparison.csv", table)
    _write_csv(output / "readout_summary.csv", summary)
    markdown = comparison_markdown(table, config)
    lines = ["| method | readout | macro F1 | min | max | macro ARI | macro NMI | train-seed std |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for entry in summary:
        spread = entry.get("mean_train_seed_std")
        lines.append(
            f"| {entry['method']} | {entry['readout']} | {entry['macro_f1']:.4f} | "
            f"{entry['min_f1']:.4f} | {entry['max_f1']:.4f} | {entry['macro_ari']:.4f} | "
            f"{entry['macro_nmi']:.4f} | "
            + ("—" if spread is None else f"{spread:.4f}") + " |")
    readout_markdown = "\n".join(lines)
    (output / "comparison.md").write_text(
        markdown + "\n\n" + readout_markdown + "\n", encoding="utf-8")
    (output / "merge_summary.json").write_text(json.dumps({
        "sources": sources, "row_count": len(rows),
        "datasets": sorted(present), "readout_summary": summary,
    }, indent=2), encoding="utf-8")

    print(markdown)
    print()
    print(readout_markdown)
    print(f"\nmerged {len(rows)} rows from {len(sources)} directories -> {output}")


if __name__ == "__main__":
    main()
