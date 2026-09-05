"""Independent audit for mainExp_Task4B_PooledInstanceSplit_3.1.

This script does not import the training driver or the pooled-split helpers.
It re-derives the split certificates from the cached seeds with its own
KD-tree queries, checks instance purity, recomputes every reported test metric
from the saved predictions with sklearn/numpy, compares them with the formal
summary, and verifies that no model checkpoint was persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)

CLASSES = ("ordinary_streamwise", "ordinary_spanwise", "hairpin_head", "hairpin_limb")
TOLERANCE = 1e-9


class AuditFailure(Exception):
    pass


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrapped(points: np.ndarray, periods) -> np.ndarray:
    if not periods:
        return points
    copies = []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            copies.append(points + np.asarray([sx * periods[0], sy * periods[1], 0.0])[None, :])
    return np.concatenate(copies)


def _metrics(targets, probabilities, ids):
    predicted = probabilities.argmax(axis=1)
    present = np.unique(targets)
    out = {
        "macro_f1": float(f1_score(targets, predicted, labels=present, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_average_precision_ovr": float(np.mean([
            average_precision_score(targets == c, probabilities[:, c]) for c in present
        ])),
    }
    hairpin = targets >= 2
    out["hairpin_to_ordinary_error_rate"] = float(np.mean(predicted[hairpin] < 2)) if hairpin.any() else None
    member = predicted >= 2
    out["hairpin_membership_f1"] = float(f1_score(hairpin, member, zero_division=0))
    known = probabilities[hairpin][:, 2:].argmax(axis=1) + 2
    out["known_hairpin_mask_head_limb_macro_f1"] = float(
        f1_score(targets[hairpin], known, labels=np.unique(targets[hairpin]), average="macro", zero_division=0)
    )
    scores = []
    for instance in np.unique(ids[hairpin]):
        rows = ids == instance
        labels_present = [c for c in (2, 3) if np.any(targets[rows] == c)]
        scores.append(float(f1_score(targets[rows], predicted[rows], labels=labels_present, average="macro", zero_division=0)))
    out["vortex_id_equal_head_limb_macro_f1"] = float(np.mean(scores)) if scores else None
    return out


def audit(config_path: Path, output_dir: Path, write_json: Path | None) -> dict:
    failures: list[str] = []
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary_path = output_dir / "summary.json"
    _require(summary_path.is_file(), "summary.json missing", failures)
    if failures:
        raise AuditFailure(failures)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_path = Path(summary["cache"])
    _require(cache_path.is_file(), "cache file missing", failures)
    _require(_sha256(cache_path) == summary["cache_sha256"], "cache SHA-256 differs from summary", failures)
    recorded_config = Path(summary["config"])
    _require(recorded_config.is_file(), "config file recorded in summary is missing", failures)
    if recorded_config.is_file():
        _require(_sha256(recorded_config) == summary["config_sha256"],
                 "recorded config SHA-256 differs from summary", failures)

    with np.load(cache_path) as data:
        labels = np.asarray(data["labels"], dtype=np.int64)
        split = np.asarray(data["split_codes"], dtype=np.int64)
        volume = np.asarray(data["volume_codes"], dtype=np.int64)
        ids = np.asarray(data["vortex_ids"], dtype=np.int64)
        seeds = np.asarray(data["seeds_xyz"], dtype=np.float64)
        metadata = json.loads(str(data["metadata_json"]))

    # Re-derive the buffer certificates and instance purity with an independent KD-tree.
    certificates = {}
    for name, code in (("channel", 0), ("tbl", 1)):
        rows = volume == code
        meta = metadata["volumes"][name]
        buffer_distance = float(meta["buffer_certificate"]["buffer_distance"])
        stream = meta["streamlines"]
        radius = int(stream["steps_per_direction"]) * float(stream["spatial_step"]) + float(stream["offset"])
        _require(np.isclose(buffer_distance, float(spec["split"]["buffer_radius_multiplier"]) * radius),
                 f"{name}: buffer distance is not the configured multiple of the primitive radius", failures)
        test = rows & (split == 2)
        nontest = rows & (split != 2)
        tree = cKDTree(_wrapped(seeds[test], meta.get("periods_xy")))
        distance, _ = tree.query(seeds[nontest], k=1)
        minimum = float(distance.min())
        _require(minimum >= buffer_distance, f"{name}: kept non-test seed within buffer of a test seed ({minimum} < {buffer_distance})", failures)
        hairpin = rows & (ids > 0)
        _require(np.array_equal(hairpin, rows & (labels >= 2)), f"{name}: hairpin support differs from head/limb labels", failures)
        per_instance_splits = {int(i): np.unique(split[hairpin & (ids == i)]).tolist() for i in np.unique(ids[hairpin])}
        _require(all(len(v) == 1 for v in per_instance_splits.values()), f"{name}: an instance spans several splits", failures)
        assignment = meta["instance_assignment"]
        for split_name, code_value in (("train", 0), ("validation", 1), ("test", 2)):
            cached = sorted(int(i) for i, v in per_instance_splits.items() if v == [code_value])
            declared = sorted(int(i) for i in assignment[split_name])
            _require(set(cached).issubset(declared), f"{name}: cached {split_name} instances not in declared assignment", failures)
        certificates[name] = {"buffer_distance": buffer_distance, "recomputed_minimum_nontest_to_test": minimum,
                              "instances_cached_per_split": {s: int(sum(1 for v in per_instance_splits.values() if v == [c]))
                                                             for s, c in (("train", 0), ("validation", 1), ("test", 2))}}

    # Recompute every reported test metric from predictions.
    runs = summary["runs"]
    _require(len(runs) == len(spec["variants"]) * len(spec["training"]["seeds"]), "run count differs from config", failures)
    expected_test = np.flatnonzero(split == 2)
    recomputed = []
    for row in runs:
        path = Path(row["prediction_path"])
        _require(path.is_file(), f"missing prediction {path.name}", failures)
        if not path.is_file():
            continue
        with np.load(path) as data:
            indices = np.asarray(data["test_source_indices"], dtype=np.int64)
            targets = np.asarray(data["test_targets"], dtype=np.int64)
            probabilities = np.asarray(data["test_probabilities"], dtype=np.float64)
        _require(np.array_equal(np.sort(indices), expected_test), f"{path.name}: test rows differ from cache", failures)
        _require(np.array_equal(targets, labels[indices]), f"{path.name}: targets differ from cache labels", failures)
        _require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), f"{path.name}: probabilities not on the simplex", failures)
        scopes = {"pooled": np.ones(len(indices), dtype=bool), "channel": volume[indices] == 0, "tbl": volume[indices] == 1}
        record = {"variant": row["variant"], "seed": int(row["seed"])}
        for scope, mask in scopes.items():
            metrics = _metrics(targets[mask], probabilities[mask], ids[indices][mask])
            for key, value in metrics.items():
                reported = row.get(f"ext_{scope}_{key}")
                record[f"{scope}_{key}"] = value
                if value is None or reported is None:
                    _require(value is None and reported is None, f"{path.name}: {scope}/{key} None mismatch", failures)
                else:
                    _require(abs(float(reported) - value) <= TOLERANCE, f"{path.name}: {scope}/{key} reported {reported} vs recomputed {value}", failures)
        _require(abs(float(row["test_macro_f1"]) - record["pooled_macro_f1"]) <= TOLERANCE, f"{path.name}: test_macro_f1 differs", failures)
        recomputed.append(record)
    parameter_counts = {r["variant"]: int(r["parameter_count"]) for r in runs}
    if {"raw", "raw_wide", "raw_fmt"} <= set(parameter_counts):
        _require(parameter_counts["raw_wide"] > parameter_counts["raw_fmt"] > parameter_counts["raw"],
                 f"parameter counts do not satisfy raw_wide > raw_fmt > raw: {parameter_counts}", failures)
    checkpoints = [str(p) for p in output_dir.rglob("*") if p.suffix in {".pt", ".pth", ".ckpt", ".safetensors"}]
    _require(not checkpoints, f"persistent checkpoints found: {checkpoints}", failures)

    # Independent paired comparison on pooled macro-F1.
    paired = {}
    by = {(r["variant"], r["seed"]): r for r in recomputed}
    for baseline in ("raw", "raw_wide"):
        diffs = [by[("raw_fmt", s)]["pooled_macro_f1"] - by[(baseline, s)]["pooled_macro_f1"]
                 for s in spec["training"]["seeds"] if ("raw_fmt", s) in by and (baseline, s) in by]
        if diffs:
            paired[f"raw_fmt_minus_{baseline}_pooled_macro_f1"] = {
                "mean": float(np.mean(diffs)), "per_seed": [float(d) for d in diffs],
                "positive_seeds": int(sum(d > 0 for d in diffs))}
    report = {
        "audit": "Audit_Task4B_PooledInstanceSplit_3.1",
        "status": "PASS" if not failures else "FAIL",
        "independent_of_training_driver": True,
        "hard_failures": failures,
        "certificates": certificates,
        "recomputed_runs": recomputed,
        "paired": paired,
        "parameter_counts": parameter_counts,
        "checkpoint_count": len(checkpoints),
    }
    if write_json:
        write_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "hard_failures": failures, "paired": paired}, indent=2))
    if failures:
        raise SystemExit(1)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    audit(Path(args.config), Path(args.output_dir), Path(args.write_json) if args.write_json else None)
