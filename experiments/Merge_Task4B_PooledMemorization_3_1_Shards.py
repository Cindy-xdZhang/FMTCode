"""Merge per-(variant, seed) shard summaries of Verify_Task4B_PooledMemorization_3.1.

The sharded array runs the frozen trainer once per (variant, seed) into a shared
output directory, producing ``summary_<variant>_seed<seed>.json`` files plus the
usual ``runs/``, ``predictions/`` and ``histories/`` artifacts.  This script
assembles the single ``summary.json`` that the sequential trainer would have
written, using the trainer's own ``_aggregate`` so the pass gate is evaluated by
the same code.  It performs no training and changes no run artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.Verify_Task4B_PooledMemorization_3_1 import _aggregate

VARIANTS = ("raw", "fmt_only", "raw_fmt")
SEEDS = (7068, 7069, 7070)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(config_path: Path, output_dir: Path) -> Path:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shards = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = output_dir / f"summary_{variant}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing shard summary: {path}")
            shard = json.loads(path.read_text(encoding="utf-8"))
            runs = shard["aggregate"]["runs"]
            if len(runs) != 1 or runs[0]["variant"] != variant or int(runs[0]["seed"]) != seed:
                raise RuntimeError(f"{path.name}: shard does not contain exactly its own run")
            if shard.get("selected_subset_run") is not True:
                raise RuntimeError(f"{path.name}: shard is not marked as a subset run")
            shards.append((path, shard))

    first = shards[0][1]
    invariant_keys = (
        "experiment", "config", "cache", "cache_sha256", "artifact_identity",
        "fit_equals_evaluation", "holdout", "sample_count", "class_support",
        "interpretation_boundary",
    )
    for path, shard in shards[1:]:
        for key in invariant_keys:
            if json.dumps(shard[key], sort_keys=True) != json.dumps(first[key], sort_keys=True):
                raise RuntimeError(f"{path.name}: shard field {key!r} differs from {shards[0][0].name}")
    if first["experiment"] != spec["experiment"]:
        raise RuntimeError("shard experiment name differs from config")

    normalization_path = output_dir / "normalization_all_samples.npz"
    if not normalization_path.is_file():
        raise FileNotFoundError(normalization_path)
    rows = [shard["aggregate"]["runs"][0] for _, shard in shards]
    for row in rows:
        for key in ("prediction_path", "history_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"{row['variant']} seed {row['seed']}: missing {key} {row[key]}")
    aggregate = _aggregate(spec, rows)
    devices = sorted({json.dumps(shard["device"], sort_keys=True) for _, shard in shards})
    summary = {
        "experiment": first["experiment"],
        "config": first["config"],
        "cache": first["cache"],
        "cache_sha256": first["cache_sha256"],
        "artifact_identity": first["artifact_identity"],
        "normalization_sha256": _sha256(normalization_path),
        "device": first["device"],
        "devices_by_shard": {
            f"{shard['aggregate']['runs'][0]['variant']}_seed{shard['aggregate']['runs'][0]['seed']}": shard["device"]
            for _, shard in shards
        },
        "distinct_device_count": len(devices),
        "fit_equals_evaluation": first["fit_equals_evaluation"],
        "holdout": first["holdout"],
        "sample_count": first["sample_count"],
        "class_support": first["class_support"],
        "selected_subset_run": False,
        "merged_from_shards": [path.name for path, _ in shards],
        "merged_shard_sha256": {path.name: _sha256(path) for path, _ in shards},
        "aggregate": aggregate,
        "interpretation_boundary": first["interpretation_boundary"],
    }
    target = output_dir / "summary.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: aggregate[k] for k in aggregate if k != "runs"}, indent=2))
    print("passed per run:", [(r["variant"], r["seed"], r["passed"], r["selected_epoch"]) for r in rows])
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge(Path(args.config), Path(args.output_dir))
