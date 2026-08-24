"""Validate and merge isolated Task3 residual shards without recomputation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import yaml


MODES = {
    "fmt_residual": "raw_fmt_residual",
    "raw_pca_residual": "raw_pca_residual",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"refusing to overwrite different file: {destination}")
        return
    shutil.copy2(source, destination)


def _write_csv_atomic(path, rows):
    path = Path(path)
    if not rows:
        raise RuntimeError(f"cannot write empty merged CSV: {path}")
    fieldnames = list(rows[0].keys())
    if any(list(row.keys()) != fieldnames for row in rows):
        raise RuntimeError(f"inconsistent CSV schema while merging {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def merge_mode(root, group_name, datasets, seeds, mode):
    expected_variant = MODES[mode]
    shard_root = Path(root) / f"development_{group_name}" / f"{mode}_shards"
    target = Path(root) / f"development_{group_name}" / mode
    rows = []
    sources = []
    for dataset in datasets:
        shard = shard_root / dataset
        csv_path = shard / "per_run.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing shard result: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            dataset_rows = list(csv.DictReader(handle))
        keys = {
            (row["dataset"], int(row["seed"]), row["variant"])
            for row in dataset_rows
        }
        expected = {(dataset, int(seed), expected_variant) for seed in seeds}
        if keys != expected or len(dataset_rows) != len(expected):
            raise RuntimeError(
                f"incomplete or duplicate shard {csv_path}: "
                f"expected={sorted(expected)}, found={sorted(keys)}"
            )
        rows.extend(dataset_rows)
        sources.append(str(shard))
        for folder, pattern in (("checkpoints", "*.pt"), ("histories", "*.csv")):
            files = sorted((shard / folder).glob(pattern))
            expected_count = len(seeds)
            if len(files) != expected_count:
                raise RuntimeError(
                    f"expected {expected_count} files in {shard / folder}, "
                    f"found {len(files)}"
                )
            for source in files:
                _copy_verified(source, target / folder / source.name)
    rows.sort(key=lambda row: (row["dataset"], int(row["seed"])))
    _write_csv_atomic(target / "per_run.csv", rows)
    return {
        "group": group_name,
        "mode": mode,
        "variant": expected_variant,
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "row_count": len(rows),
        "sources": sources,
        "result_csv_sha256": _sha256(target / "per_run.csv"),
    }


def run(config_path, root="outputs/mainExp_Task3_3D_3.1"):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in spec["seeds"]]
    manifests = []
    for group in spec["groups"]:
        for mode in MODES:
            manifests.append(
                merge_mode(root, group["name"], group["datasets"], seeds, mode)
            )
    manifest_path = Path(root) / "residual_shard_merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifests, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"merged {sum(item['row_count'] for item in manifests)} rows")
    print(f"manifest={manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/mainExp_Task3_3D_3.1_evaluate.yaml"
    )
    parser.add_argument("--root", default="outputs/mainExp_Task3_3D_3.1")
    args = parser.parse_args()
    run(args.config, args.root)
