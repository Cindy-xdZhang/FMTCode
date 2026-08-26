"""Adaptive Re160 Task5 selection followed by one sealed fresh-time test.

The adaptive selector may read only the already exposed development ordinals
3--5 from ``Verify_Task5_CylinderHyperparams_1.1``.  It freezes one candidate
and a final-training configuration before the fresh source caches exist.  The
fresh evaluator then opens source starts 85 and 96 exactly once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from Search_Task5_CylinderHyperparams import (
    _candidate,
    _candidate_checkpoint_path,
    _evaluate_baseline,
    _evaluate_residual,
    _load_records,
    _load_spec,
    run_candidate,
)
from Verify_Task3_FMTClassifier import _stack_split


METRICS = ("f1", "average_precision")
BASELINE_METHODS = ("raw", "raw_wide")
RESIDUAL_METHODS = ("raw_pca_residual", "fmt_residual")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, method, ordinal, metric, candidate_id=None):
    selected = [
        float(row[metric])
        for row in rows
        if row["method"] == method
        and int(row["ordinal"]) == int(ordinal)
        and (
            candidate_id is None
            or row.get("candidate_id", "") == candidate_id
        )
    ]
    if not selected:
        raise RuntimeError(
            f"missing rows for method={method}, ordinal={ordinal}, "
            f"candidate={candidate_id}"
        )
    return float(np.mean(selected))


def rank_adaptive_candidates(rows, candidate_ids, ordinals):
    """Rank candidates by their worst old-development gain.

    For every exposed ordinal, seed means are formed first.  The score is the
    minimum gain over F1/AP and over matched Raw-PCA/stronger Raw.  This pure
    helper is intentionally testable without checkpoints.
    """
    leaderboard = []
    for candidate_id in candidate_ids:
        details = {}
        gains = []
        for ordinal in ordinals:
            ordinal = int(ordinal)
            details[str(ordinal)] = {}
            for metric in METRICS:
                fmt = _mean(
                    rows, "fmt_residual", ordinal, metric, candidate_id
                )
                raw_pca = _mean(
                    rows, "raw_pca_residual", ordinal, metric, candidate_id
                )
                raw = _mean(rows, "raw", ordinal, metric)
                raw_wide = _mean(rows, "raw_wide", ordinal, metric)
                strong_raw = max(raw, raw_wide)
                matched_gain = fmt - raw_pca
                strong_gain = fmt - strong_raw
                gains.extend((matched_gain, strong_gain))
                details[str(ordinal)][metric] = {
                    "fmt": fmt,
                    "raw_pca": raw_pca,
                    "strong_raw": strong_raw,
                    "gain_vs_raw_pca": matched_gain,
                    "gain_vs_strong_raw": strong_gain,
                }
        leaderboard.append({
            "candidate_id": candidate_id,
            "worst_adaptive_gain": float(min(gains)),
            "mean_adaptive_gain": float(np.mean(gains)),
            "details_json": json.dumps(details, sort_keys=True),
        })
    return sorted(
        leaderboard,
        key=lambda row: (
            row["worst_adaptive_gain"], row["mean_adaptive_gain"]
        ),
        reverse=True,
    )


def _workflow_spec(config_path):
    return _load_spec(config_path)


def _feature_signature(candidate):
    """Identify candidates that produce exactly the same FMT input."""
    keys = (
        "fmt_recipe", "gram_num_freq", "kinematic_num_freq",
        "gram_subtract_initial", "gram_normalize_initial_scale",
        "kinematic_log_compress", "kinematic_pinv_rtol",
    )
    defaults = {
        "gram_num_freq": 2,
        "kinematic_num_freq": 6,
        "gram_subtract_initial": True,
        "gram_normalize_initial_scale": True,
        "kinematic_log_compress": False,
        "kinematic_pinv_rtol": 1e-6,
    }
    return tuple((key, candidate.get(key, defaults.get(key))) for key in keys)


def _fresh_source_paths(spec):
    return sorted(
        (Path(spec["fresh"]["source_cache_root"]) / spec["dataset"])
        .glob("slice_*.npz")
    )


def _final_training_spec(workflow, search, selection, selection_hash):
    candidate = dict(selection["candidate"])
    return {
        "experiment": workflow["experiment"],
        "source_cache_root": search["source_cache_root"],
        "label_cache_root": search["label_cache_root"],
        "output_root": workflow["output_root"],
        "datasets": [workflow["dataset"]],
        "expected_slices": int(search["expected_slices"]),
        "sampled_steps": int(search["sampled_steps"]),
        "raw_pca_random_state": int(search["raw_pca_random_state"]),
        "raw_wide_parameter_count": int(search["raw_wide_parameter_count"]),
        "screen_seeds": [int(value) for value in workflow["final_seeds"]],
        "screen_split": dict(workflow["final_split"]),
        "training": dict(search["training"]),
        "candidates": [candidate],
        "adaptive_selection_sha256": selection_hash,
        "adaptive_source": {
            "experiment": search["experiment"],
            "ordinals": [int(value) for value in workflow["adaptive_ordinals"]],
            "old_confirmation_read": False,
        },
    }


def adaptive_select(config_path):
    workflow = _workflow_spec(config_path)
    search_path = Path(workflow["previous_search_config"])
    search = _load_spec(search_path)
    output_root = Path(workflow["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "selected_re160_candidate.json"
    if selection_path.exists():
        print(selection_path.read_text(encoding="utf-8"))
        return selection_path
    if _fresh_source_paths(workflow):
        raise RuntimeError(
            "fresh cache already exists before adaptive selection; refusing to "
            "claim a sealed test"
        )

    dataset = str(workflow["dataset"])
    ordinals = [int(value) for value in workflow["adaptive_ordinals"]]
    seeds = [int(value) for value in search["screen_seeds"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(search["training"]["batch_size"])
    rows = []

    # Raw and Raw-wide are candidate-independent and are evaluated once.
    reference_index = next(
        index for index, row in enumerate(search["candidates"])
        if row["fmt_recipe"] == "all"
    )
    reference_candidate = _candidate(search, reference_index)
    reference_records = _load_records(
        search, dataset, reference_candidate, ordinals,
        feature_device=device,
    )
    for ordinal in ordinals:
        split = _stack_split(reference_records, [ordinal])
        for seed in seeds:
            checkpoint_dir = (
                Path(search["output_root"]) / "baselines" / dataset
                / "checkpoints"
            )
            for method in BASELINE_METHODS:
                checkpoint = checkpoint_dir / f"{dataset}_{method}_seed{seed}.pt"
                metrics = _evaluate_baseline(
                    checkpoint, split, batch_size, device
                )
                rows.append({
                    "candidate_id": "", "candidate_index": "",
                    "ordinal": ordinal, "seed": seed, "method": method,
                    **metrics, "checkpoint": str(checkpoint),
                })

    candidate_ids = []
    records_by_signature = {
        _feature_signature(reference_candidate): reference_records
    }
    for index in range(len(search["candidates"])):
        candidate = _candidate(search, index)
        candidate_ids.append(candidate["id"])
        signature = _feature_signature(candidate)
        if signature not in records_by_signature:
            records_by_signature[signature] = _load_records(
                search, dataset, candidate, ordinals,
                feature_device=device,
            )
        records = records_by_signature[signature]
        for ordinal in ordinals:
            split = _stack_split(records, [ordinal])
            for seed in seeds:
                candidate_root = (
                    Path(search["output_root"]) / "candidates"
                    / candidate["id"] / dataset / f"seed{seed}"
                )
                for source, method in (
                    ("raw_pca", "raw_pca_residual"),
                    ("fmt", "fmt_residual"),
                ):
                    checkpoint = _candidate_checkpoint_path(
                        candidate_root, source, dataset, seed
                    )
                    metrics = _evaluate_residual(
                        checkpoint, split, batch_size, device
                    )
                    rows.append({
                        "candidate_id": candidate["id"],
                        "candidate_index": index, "ordinal": ordinal,
                        "seed": seed, "method": method, **metrics,
                        "checkpoint": str(checkpoint),
                    })
        print(f"adaptive evaluated {candidate['id']}", flush=True)

    leaderboard = rank_adaptive_candidates(rows, candidate_ids, ordinals)
    index_by_id = {
        row["id"]: index for index, row in enumerate(search["candidates"])
    }
    for row in leaderboard:
        row["candidate_index"] = index_by_id[row["candidate_id"]]
    _write_csv(output_root / "adaptive_per_run.csv", rows)
    _write_csv(output_root / "adaptive_leaderboard.csv", leaderboard)

    selected = leaderboard[0]
    selected_index = int(selected["candidate_index"])
    selection = {
        **selected,
        "candidate": dict(search["candidates"][selected_index]),
        "selection_rule": (
            "maximize the worst seed-mean gain across exposed development "
            "ordinals 3-5, F1/AP, matched Raw-PCA and stronger Raw; tie by mean"
        ),
        "selection_data": "already exposed development ordinals 3-5",
        "adaptive_ordinals": ordinals,
        "old_task5_confirmation_read": False,
        "fresh_source_cache_read": False,
        "fresh_source_indices": [
            int(value) for value in workflow["fresh"]["source_indices"]
        ],
        "previous_search_config_sha256": _sha256(search_path),
        "fresh_cache_config_sha256": _sha256(
            workflow["fresh"]["cache_config"]
        ),
    }
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    selection_hash = _sha256(selection_path)
    training_spec = _final_training_spec(
        workflow, search, selection, selection_hash
    )
    training_path = output_root / "frozen_candidate_training.yaml"
    training_path.write_text(
        yaml.safe_dump(training_spec, sort_keys=False), encoding="utf-8"
    )
    print(selection_path.read_text(encoding="utf-8"))
    return selection_path


def train_candidate(config_path, seed):
    workflow = _workflow_spec(config_path)
    seed = int(seed)
    allowed = {int(value) for value in workflow["final_seeds"]}
    if seed not in allowed:
        raise ValueError(f"seed {seed} is not pre-registered")
    output_root = Path(workflow["output_root"])
    selection_path = output_root / "selected_re160_candidate.json"
    training_path = output_root / "frozen_candidate_training.yaml"
    if not selection_path.exists() or not training_path.exists():
        raise FileNotFoundError("adaptive selection must finish before training")
    frozen = _load_spec(training_path)
    if frozen["adaptive_selection_sha256"] != _sha256(selection_path):
        raise RuntimeError("frozen training config disagrees with selection")
    return run_candidate(training_path, workflow["dataset"], 0, seed)


def _scale_splits(source_paths, records):
    scale_ids = []
    scale_names = None
    source_indices = []
    for path, record in zip(source_paths, records):
        with np.load(path) as source:
            ids = np.asarray(source["scale_id"], dtype=np.int64)
            metadata = json.loads(str(source["metadata_json"]))
        if len(ids) != len(record[0]):
            raise RuntimeError(f"scale/sample mismatch in {path}")
        names = [row["name"] for row in metadata["scale_table"]]
        if scale_names is None:
            scale_names = names
        elif names != scale_names:
            raise RuntimeError("fresh scale table differs across slices")
        scale_ids.append(ids)
        source_indices.append(int(metadata["source_start_index"]))
    pooled = _stack_split(records, list(range(len(records))))
    per_scale = {}
    for scale_id, name in enumerate(scale_names):
        parts = []
        for record, ids in zip(records, scale_ids):
            mask = ids == scale_id
            parts.append(tuple(value[mask] for value in record[:3]))
        per_scale[name] = tuple(
            np.concatenate([part[index] for part in parts], axis=0)
            for index in range(3)
        )
    return pooled, per_scale, source_indices


def _evaluate_method(method, checkpoint, split, batch_size, device):
    if method in BASELINE_METHODS:
        return _evaluate_baseline(checkpoint, split, batch_size, device)
    return _evaluate_residual(checkpoint, split, batch_size, device)


def summarize_fresh_rows(rows, seeds, gates):
    summary = {}
    for method in (*BASELINE_METHODS, *RESIDUAL_METHODS):
        method_rows = [row for row in rows if row["method"] == method]
        if len(method_rows) != len(seeds):
            raise RuntimeError(f"incomplete fresh rows for {method}")
        summary[method] = {
            metric: float(np.mean([float(row[metric]) for row in method_rows]))
            for metric in METRICS
        }
        summary[method].update({
            f"{metric}_std": float(np.std(
                [float(row[metric]) for row in method_rows], ddof=1
            ))
            for metric in METRICS
        })

    strong = {
        metric: max(summary[method][metric] for method in BASELINE_METHODS)
        for metric in METRICS
    }
    gains = {
        f"{metric}_vs_raw_pca": (
            summary["fmt_residual"][metric]
            - summary["raw_pca_residual"][metric]
        )
        for metric in METRICS
    }
    gains.update({
        f"{metric}_vs_strong_raw": summary["fmt_residual"][metric] - strong[metric]
        for metric in METRICS
    })

    paired = []
    for seed in seeds:
        by_method = {
            row["method"]: row for row in rows if int(row["seed"]) == int(seed)
        }
        row = {"seed": int(seed)}
        for metric in METRICS:
            row[f"{metric}_vs_raw_pca"] = (
                float(by_method["fmt_residual"][metric])
                - float(by_method["raw_pca_residual"][metric])
            )
            row[f"{metric}_vs_strong_raw"] = (
                float(by_method["fmt_residual"][metric])
                - max(float(by_method[method][metric]) for method in BASELINE_METHODS)
            )
        paired.append(row)
    positive = {
        key: sum(float(row[key]) > 0.0 for row in paired)
        for key in (
            "f1_vs_raw_pca", "average_precision_vs_raw_pca",
            "f1_vs_strong_raw", "average_precision_vs_strong_raw",
        )
    }
    primary_pass = (
        gains["f1_vs_raw_pca"] >= float(gates["minimum_matched_gain"])
        and gains["average_precision_vs_raw_pca"]
        >= float(gates["minimum_matched_gain"])
        and gains["f1_vs_strong_raw"] > float(gates["minimum_strong_raw_gain"])
        and gains["average_precision_vs_strong_raw"]
        > float(gates["minimum_strong_raw_gain"])
    )
    robustness_pass = primary_pass and all(
        positive[key] >= int(gates["minimum_positive_seed_count"])
        for key in ("f1_vs_raw_pca", "average_precision_vs_raw_pca")
    )
    return {
        "methods": summary, "strong_raw": strong, "gains": gains,
        "paired_gains": paired, "positive_seed_counts": positive,
        "primary_gate_pass": bool(primary_pass),
        "seed_robustness_gate_pass": bool(robustness_pass),
    }


def evaluate_fresh(config_path):
    workflow = _workflow_spec(config_path)
    output_root = Path(workflow["output_root"])
    selection_path = output_root / "selected_re160_candidate.json"
    training_path = output_root / "frozen_candidate_training.yaml"
    marker = output_root / "fresh_test" / "audit.json"
    if not selection_path.exists() or not training_path.exists():
        raise FileNotFoundError("selection and frozen training must exist")
    identity = {
        "selection_sha256": _sha256(selection_path),
        "training_config_sha256": _sha256(training_path),
        "fresh_cache_config_sha256": _sha256(workflow["fresh"]["cache_config"]),
        "fresh_label_config_sha256": _sha256(workflow["fresh"]["label_config"]),
    }
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError("fresh test was already opened under another identity")
        print(marker.read_text(encoding="utf-8"))
        return marker

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidate = dict(selection["candidate"])
    source_paths = _fresh_source_paths(workflow)
    expected = int(workflow["fresh"]["expected_slices"])
    if len(source_paths) != expected:
        raise RuntimeError(f"expected {expected} fresh slices, found {len(source_paths)}")
    if any(path.stat().st_mtime_ns <= selection_path.stat().st_mtime_ns
           for path in source_paths):
        raise RuntimeError("fresh caches do not post-date frozen selection")
    fresh_spec = {
        "source_cache_root": workflow["fresh"]["source_cache_root"],
        "label_cache_root": workflow["fresh"]["label_cache_root"],
        "expected_slices": expected,
        "sampled_steps": int(workflow["sampled_steps"]),
    }
    records = _load_records(
        fresh_spec, workflow["dataset"], candidate, range(expected)
    )
    pooled, per_scale, source_indices = _scale_splits(source_paths, records)
    if source_indices != [int(value) for value in workflow["fresh"]["source_indices"]]:
        raise RuntimeError(f"unexpected fresh source indices: {source_indices}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(workflow["evaluation_batch_size"])
    seeds = [int(value) for value in workflow["final_seeds"]]
    rows, scale_rows = [], []
    for seed in seeds:
        baseline_dir = output_root / "baselines" / workflow["dataset"] / "checkpoints"
        candidate_root = (
            output_root / "candidates" / candidate["id"]
            / workflow["dataset"] / f"seed{seed}"
        )
        checkpoints = {
            "raw": baseline_dir / f"{workflow['dataset']}_raw_seed{seed}.pt",
            "raw_wide": baseline_dir / f"{workflow['dataset']}_raw_wide_seed{seed}.pt",
            "raw_pca_residual": _candidate_checkpoint_path(
                candidate_root, "raw_pca", workflow["dataset"], seed
            ),
            "fmt_residual": _candidate_checkpoint_path(
                candidate_root, "fmt", workflow["dataset"], seed
            ),
        }
        for method, checkpoint in checkpoints.items():
            metrics = _evaluate_method(
                method, checkpoint, pooled, batch_size, device
            )
            rows.append({
                "dataset": workflow["dataset"], "seed": seed,
                "method": method, "sample_count": len(pooled[2]),
                "positive_fraction": float(pooled[2].mean()),
                **metrics, "checkpoint": str(checkpoint),
            })
            for scale_name, split in per_scale.items():
                scale_metrics = _evaluate_method(
                    method, checkpoint, split, batch_size, device
                )
                scale_rows.append({
                    "dataset": workflow["dataset"], "seed": seed,
                    "scale_name": scale_name, "method": method,
                    "sample_count": len(split[2]),
                    "positive_fraction": float(split[2].mean()),
                    **scale_metrics,
                })
        print(f"fresh evaluated seed={seed}", flush=True)

    fresh_dir = output_root / "fresh_test"
    _write_csv(fresh_dir / "per_run.csv", rows)
    _write_csv(fresh_dir / "per_scale.csv", scale_rows)
    summary = summarize_fresh_rows(rows, seeds, workflow["gates"])
    scale_summary = []
    for scale_name in per_scale:
        selected_rows = [row for row in scale_rows if row["scale_name"] == scale_name]
        for metric in METRICS:
            fmt = float(np.mean([
                row[metric] for row in selected_rows if row["method"] == "fmt_residual"
            ]))
            raw_pca = float(np.mean([
                row[metric] for row in selected_rows
                if row["method"] == "raw_pca_residual"
            ]))
            scale_summary.append({
                "scale_name": scale_name, "metric": metric,
                "fmt": fmt, "raw_pca": raw_pca, "gain": fmt - raw_pca,
            })
    _write_csv(fresh_dir / "per_scale_summary.csv", scale_summary)
    audit = {
        "experiment": workflow["experiment"], **identity,
        "selected_candidate": candidate,
        "fresh_source_indices": source_indices,
        "fresh_scale_names": list(per_scale),
        "fresh_cache_created_after_selection": True,
        "old_task5_confirmation_used_for_selection": False,
        "adaptive_selection_used_only_exposed_development": True,
        "summary": summary,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(marker.read_text(encoding="utf-8"))
    return marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Verify_Task5_Re160FreshTimes_1.1.yaml"
    )
    parser.add_argument(
        "--mode", choices=("adaptive-select", "train-candidate", "evaluate-fresh"),
        required=True,
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "adaptive-select":
        adaptive_select(args.config)
    elif args.mode == "train-candidate":
        if args.seed is None:
            parser.error("train-candidate requires --seed")
        train_candidate(args.config, args.seed)
    else:
        evaluate_fresh(args.config)
