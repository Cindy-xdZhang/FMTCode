"""Combine fresh Task1 and Task2 confirmations and test the requested hierarchy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(task2: str | Path, task1_tables: list[str], output: str | Path) -> Path:
    task2_payload = json.loads(Path(task2).read_text(encoding="utf-8"))
    task2_rows = {row["dataset"]: row for row in task2_payload["paper_table"]}
    task1_rows = {}
    for table in task1_tables:
        task1_rows.update({row["dataset"]: row for row in _read_csv(table)})
    if set(task1_rows) != set(task2_rows):
        raise RuntimeError(
            f"Task1/Task2 datasets differ: Task1-only={set(task1_rows)-set(task2_rows)}, "
            f"Task2-only={set(task2_rows)-set(task1_rows)}"
        )
    rows = []
    for dataset in task2_rows:
        task1_f1 = float(task1_rows[dataset]["fmt_f1_mean"])
        raw_f1 = float(task2_rows[dataset]["raw_f1_mean"])
        fmt_f1 = float(task2_rows[dataset]["fmt_f1_mean"])
        rows.append({
            "dataset": dataset,
            "task1_fmt_kmeans_f1": task1_f1,
            "raw_vae_f1": raw_f1,
            "fmt_vae_f1": fmt_f1,
            "raw_vae_minus_task1": raw_f1 - task1_f1,
            "fmt_vae_minus_raw_vae": fmt_f1 - raw_f1,
        })
    aggregate = {
        "dataset_count": len(rows),
        "task1_fmt_kmeans_f1_mean": sum(
            row["task1_fmt_kmeans_f1"] for row in rows
        ) / len(rows),
        "raw_vae_f1_mean": sum(row["raw_vae_f1"] for row in rows) / len(rows),
        "fmt_vae_f1_mean": sum(row["fmt_vae_f1"] for row in rows) / len(rows),
    }
    aggregate["raw_vae_minus_task1"] = (
        aggregate["raw_vae_f1_mean"] - aggregate["task1_fmt_kmeans_f1_mean"]
    )
    aggregate["fmt_vae_minus_raw_vae"] = (
        aggregate["fmt_vae_f1_mean"] - aggregate["raw_vae_f1_mean"]
    )
    aggregate["hierarchy_satisfied"] = (
        aggregate["raw_vae_minus_task1"] > 0
        and aggregate["fmt_vae_minus_raw_vae"] > 0
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "experiment": task2_payload["experiment"],
        "aggregate": aggregate,
        "paper_table": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    if not aggregate["hierarchy_satisfied"]:
        raise RuntimeError("fresh confirmation did not satisfy the requested hierarchy")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2", required=True)
    parser.add_argument("--task1", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.task2, args.task1, args.output)
