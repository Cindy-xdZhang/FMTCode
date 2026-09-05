"""Summarise frozen Task3 validation CSVs across methods and seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from Evaluate_Task3_FrozenConfirmation import _summarise


def _read(paths, variants):
    rows = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["variant"] not in variants:
                    continue
                rows.append({
                    "dataset": row["dataset"],
                    "seed": int(row["seed"]),
                    "variant": row["variant"],
                    "f1": float(row["validation_f1"]),
                    "average_precision": float(
                        row["validation_average_precision"]
                    ),
                })
    return rows


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(baseline_csvs, residual_csvs, output_dir, minimum_gain):
    rows = _read(baseline_csvs, {"raw", "raw_wide"})
    rows.extend(_read(residual_csvs, {"raw_fmt_residual"}))
    expected = 8 * 3 * 3
    keys = {(row["dataset"], row["seed"], row["variant"]) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(
            f"expected {expected} unique validation rows, got "
            f"{len(rows)} rows/{len(keys)} keys"
        )
    comparisons, summary = _summarise(rows, float(minimum_gain))
    output_dir = Path(output_dir)
    _write(output_dir / "per_seed_comparison.csv", comparisons)
    _write(output_dir / "per_dataset_summary.csv", summary)
    passed = all(row["passes_minimum_gain"] for row in summary)
    print(f"all_datasets_pass={passed}")
    for row in summary:
        print(
            f"{row['dataset']:22s} gain_F1={row['mean_gain_f1']:+.5f} "
            f"gain_AP={row['mean_gain_average_precision']:+.5f} "
            f"pass={row['passes_minimum_gain']}"
        )
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", action="append", required=True)
    parser.add_argument("--residual-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    args = parser.parse_args()
    run(
        args.baseline_csv, args.residual_csv, args.output_dir,
        args.minimum_gain,
    )
