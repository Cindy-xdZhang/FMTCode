"""Development-only Task5 hyperparameter search for Re160 and Re640.

The screen trains on development ordinals 0--2 and selects checkpoints on
ordinal 3.  Candidate selection is based only on ordinal 3.  Ordinals 4--5
remain sealed until ``--mode outer`` evaluates the automatically frozen
candidate.  The previously inspected Task5 confirmation set is deliberately
not read by this program.
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

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
)
from FMT_Utils.Task5FeatureRecipes_3D import task5_fmt_features_from_cache
from Verify_Task3_FMTClassifier import (
    _append_csv,
    _classification_metrics,
    _loader,
    _normalize_train_only,
    _portable_basename,
    _predict,
    _stack_split,
)
from Verify_Task3_FMTResidual import (
    _apply_raw_pca_transform,
    _load_raw_model,
    _predict_components,
    _probabilities,
    _train_one,
)


def _load_spec(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _candidate(spec, index):
    candidates = list(spec["candidates"])
    index = int(index)
    if not 0 <= index < len(candidates):
        raise IndexError(f"candidate index {index} outside [0,{len(candidates)})")
    row = dict(candidates[index])
    row["index"] = index
    return row


def _load_records(spec, dataset, candidate, ordinals, feature_device=None):
    source_dir = Path(spec["source_cache_root"]) / dataset
    label_dir = Path(spec["label_cache_root"]) / dataset
    source_paths = sorted(source_dir.glob("slice_*.npz"))
    if len(source_paths) != int(spec["expected_slices"]):
        raise RuntimeError(
            f"expected {spec['expected_slices']} slices in {source_dir}, "
            f"found {len(source_paths)}"
        )
    required = {int(value) for value in ordinals}
    records = []
    for ordinal, source_path in enumerate(source_paths):
        if ordinal not in required:
            continue
        label_path = label_dir / source_path.name
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        with np.load(source_path) as source, np.load(label_path) as labels:
            raw_flat = np.asarray(source["raw_features"], dtype=np.float32)
            fmt = task5_fmt_features_from_cache(
                source,
                int(spec["sampled_steps"]),
                candidate["fmt_recipe"],
                gram_num_freq=int(candidate.get("gram_num_freq", 2)),
                kinematic_num_freq=int(candidate.get("kinematic_num_freq", 6)),
                gram_subtract_initial=bool(
                    candidate.get("gram_subtract_initial", True)
                ),
                gram_normalize_initial_scale=bool(
                    candidate.get("gram_normalize_initial_scale", True)
                ),
                kinematic_log_compress=bool(
                    candidate.get("kinematic_log_compress", False)
                ),
                kinematic_pinv_rtol=float(
                    candidate.get("kinematic_pinv_rtol", 1e-6)
                ),
                device=feature_device,
            )
            target = np.asarray(labels["labels"], dtype=np.float32)
            metadata = json.loads(str(labels["metadata_json"]))
        if _portable_basename(metadata["source_cache"]) != source_path.name:
            raise ValueError(f"label/source mismatch for {source_path}")
        if len(raw_flat) != len(target) or len(fmt) != len(target):
            raise ValueError(f"feature/label length mismatch in {source_path}")
        raw = raw_flat.reshape(-1, 7, int(spec["sampled_steps"]), 3)
        records.append((raw, fmt, target, ordinal, metadata))
    if len(records) != len(required):
        raise RuntimeError(f"requested {sorted(required)}, loaded {len(records)} slices")
    return records


def _fusion(candidate):
    if "fixed_alpha" in candidate:
        return {
            "fixed_alpha": float(candidate["fixed_alpha"]),
            "selection_metric": str(candidate.get("selection_metric", "average_precision")),
        }
    return {
        "alpha_min": float(candidate.get("alpha_min", 0.0)),
        "alpha_max": float(candidate.get("alpha_max", 3.0)),
        "alpha_steps": int(candidate.get("alpha_steps", 61)),
        "selection_metric": str(candidate.get("selection_metric", "minimum_gain")),
        "minimum_f1_gain": float(candidate.get("minimum_f1_gain", 0.03)),
    }


def _training(spec, candidate, seed):
    result = dict(spec["training"])
    result.update(candidate.get("training", {}))
    result["seeds"] = [int(seed)]
    return result


def _residual_spec(spec, candidate, dataset, seed, source, output_dir, fmt_dim):
    return {
        "experiment": f"{spec['experiment']}_{candidate['id']}_{source}",
        "source_cache_root": spec["source_cache_root"],
        "label_cache_root": spec["label_cache_root"],
        "raw_checkpoint_dir": str(
            Path(spec["output_root"]) / "baselines" / dataset / "checkpoints"
        ),
        "output_dir": str(output_dir),
        "datasets": [dataset],
        "expected_slices": int(spec["expected_slices"]),
        "sampled_steps": int(spec["sampled_steps"]),
        "fmt_subset": candidate["fmt_recipe"],
        "fmt_gram_num_freq": int(candidate.get("gram_num_freq", 2)),
        "fmt_kinematic_num_freq": int(candidate.get("kinematic_num_freq", 6)),
        "auxiliary_source": source,
        "raw_pca_components": int(fmt_dim),
        "raw_pca_random_state": int(spec.get("raw_pca_random_state", 7068)),
        "raw_wide_parameter_count": int(spec["raw_wide_parameter_count"]),
        "split": dict(spec["screen_split"]),
        "evaluation": {"test_enabled": False},
        "model": {
            "embedding_dim": int(candidate.get("embedding_dim", 128)),
            "auxiliary_dim": int(candidate.get("auxiliary_dim", 64)),
            "residual_input": str(candidate.get("residual_input", "geometry_fmt")),
        },
        "fusion": _fusion(candidate),
        "training": _training(spec, candidate, seed),
        "search_candidate": candidate,
    }


def _existing_result(path, dataset, seed):
    if not Path(path).exists():
        return None
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["dataset"] == dataset and int(row["seed"]) == int(seed)]
    if len(rows) > 1:
        raise RuntimeError(f"duplicate results in {path}")
    return rows[0] if rows else None


def run_candidate(config_path, dataset, candidate_index, seed,
                  preloaded_records=None, device=None):
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}")
    if int(seed) not in {int(value) for value in spec["screen_seeds"]}:
        raise ValueError(f"seed {seed} is not pre-registered")
    candidate = _candidate(spec, candidate_index)
    split = spec["screen_split"]
    required = set(split["train_ordinals"]) | set(split["validation_ordinals"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = preloaded_records
    if records is None:
        records = _load_records(
            spec, dataset, candidate, required, feature_device=device
        )
    train = _stack_split(records, split["train_ordinals"])
    validation = _stack_split(records, split["validation_ordinals"])
    train, validation, _, stats = _normalize_train_only(train, validation)
    fmt_dim = int(train[1].shape[1])
    task_root = (
        Path(spec["output_root"]) / "candidates" / candidate["id"]
        / dataset / f"seed{int(seed)}"
    )
    rows = []
    for source in ("fmt", "raw_pca"):
        output_dir = task_root / source
        results_path = output_dir / "per_run.csv"
        existing = _existing_result(results_path, dataset, seed)
        if existing is not None:
            print(f"cached {candidate['id']} {dataset} seed={seed} {source}")
            rows.append(existing)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_spec = _residual_spec(
            spec, candidate, dataset, seed, source, output_dir, fmt_dim
        )
        (output_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(candidate_spec, sort_keys=False), encoding="utf-8"
        )
        row = _train_one(
            candidate_spec, dataset, int(seed),
            (train, validation, None), stats,
            device,
            output_dir,
        )
        row["candidate_id"] = candidate["id"]
        row["candidate_index"] = int(candidate_index)
        row["fmt_recipe"] = candidate["fmt_recipe"]
        row["fmt_dim"] = fmt_dim
        _append_csv(results_path, row)
        rows.append(row)
    print(json.dumps({
        "candidate": candidate["id"], "dataset": dataset, "seed": int(seed),
        "fmt_dim": fmt_dim,
        "validation": {
            row["auxiliary_source"]: {
                "f1": float(row["validation_f1"]),
                "average_precision": float(row["validation_average_precision"]),
            } for row in rows
        },
    }, indent=2))
    return task_root


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _baseline_means(spec):
    result = {}
    for dataset in spec["datasets"]:
        rows = _read_csv(
            Path(spec["output_root"]) / "baselines" / dataset / "per_run.csv"
        )
        result[dataset] = {}
        for variant in ("raw", "raw_wide"):
            selected = [row for row in rows if row["variant"] == variant]
            if len(selected) != len(spec["screen_seeds"]):
                raise RuntimeError(f"incomplete {dataset} {variant} baseline")
            result[dataset][variant] = {
                metric: float(np.mean([float(row[f"validation_{metric}"])
                                       for row in selected]))
                for metric in ("f1", "average_precision")
            }
    return result


def select_candidate(config_path):
    spec = _load_spec(config_path)
    baseline = _baseline_means(spec)
    leaderboard = []
    for index, candidate in enumerate(spec["candidates"]):
        per_dataset = {}
        complete = True
        for dataset in spec["datasets"]:
            source_rows = {}
            for source in ("fmt", "raw_pca"):
                rows = []
                for seed in spec["screen_seeds"]:
                    path = (
                        Path(spec["output_root"]) / "candidates" / candidate["id"]
                        / dataset / f"seed{int(seed)}" / source / "per_run.csv"
                    )
                    if not path.exists():
                        complete = False
                        break
                    rows.extend(_read_csv(path))
                if not complete:
                    break
                source_rows[source] = {
                    metric: float(np.mean([float(row[f"validation_{metric}"])
                                           for row in rows]))
                    for metric in ("f1", "average_precision")
                }
            if not complete:
                break
            stronger = {
                metric: max(baseline[dataset][variant][metric]
                            for variant in ("raw", "raw_wide"))
                for metric in ("f1", "average_precision")
            }
            gains = {}
            for metric in ("f1", "average_precision"):
                gains[f"{metric}_vs_raw_pca"] = (
                    source_rows["fmt"][metric] - source_rows["raw_pca"][metric]
                )
                gains[f"{metric}_vs_strong_raw"] = (
                    source_rows["fmt"][metric] - stronger[metric]
                )
            per_dataset[dataset] = {
                "fmt": source_rows["fmt"], "raw_pca": source_rows["raw_pca"],
                "strong_raw": stronger, "gains": gains,
            }
        if not complete:
            raise RuntimeError(f"candidate {candidate['id']} is incomplete")
        all_gains = [value for data in per_dataset.values()
                     for value in data["gains"].values()]
        leaderboard.append({
            "candidate_index": index,
            "candidate_id": candidate["id"],
            "worst_validation_gain": min(all_gains),
            "mean_validation_gain": float(np.mean(all_gains)),
            "re160_min_gain": min(per_dataset["cylinder3d"]["gains"].values()),
            "re640_min_gain": min(per_dataset["halfcylinderRe640"]["gains"].values()),
            "details_json": json.dumps(per_dataset, sort_keys=True),
        })
    selected = max(
        leaderboard,
        key=lambda row: (row["worst_validation_gain"], row["mean_validation_gain"]),
    )
    output_root = Path(spec["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_root / "validation_leaderboard.csv"
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0]))
        writer.writeheader()
        writer.writerows(sorted(
            leaderboard,
            key=lambda row: (row["worst_validation_gain"], row["mean_validation_gain"]),
            reverse=True,
        ))
    selection = {
        **selected,
        "candidate": spec["candidates"][int(selected["candidate_index"])],
        "selection_data": "development ordinal 3 only",
        "outer_ordinals_were_read": False,
        "selection_rule": "maximize worst Re160/Re640 F1/AP gain versus both matched Raw-PCA and stronger Raw; tie-break by mean gain",
    }
    payload = json.dumps(selection, indent=2, sort_keys=True)
    (output_root / "selected_candidate.json").write_text(payload, encoding="utf-8")
    print(payload)
    return selection


def _normalize_with_checkpoint(split, stats):
    raw, fmt, labels = split
    raw = ((raw - np.asarray(stats["raw_mean"]))
           / np.asarray(stats["raw_std"])).astype(np.float32)
    fmt = ((fmt - np.asarray(stats["fmt_mean"]))
           / np.asarray(stats["fmt_std"])).astype(np.float32)
    return raw, fmt, labels.astype(np.float32)


def _candidate_checkpoint_path(candidate_root, source, dataset, seed):
    """Return the checkpoint path written by the residual trainers."""
    checkpoint_methods = {
        "raw_pca": "raw_pca_residual",
        "fmt": "raw_fmt_residual",
    }
    try:
        checkpoint_method = checkpoint_methods[source]
    except KeyError as exc:
        raise ValueError(f"unknown residual source: {source}") from exc
    return (
        Path(candidate_root) / source / "checkpoints"
        / f"{dataset}_{checkpoint_method}_seed{int(seed)}.pt"
    )


def _evaluate_residual(checkpoint_path, split, batch_size, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalized = _normalize_with_checkpoint(split, checkpoint["normalization"])
    if checkpoint["auxiliary_source"] == "raw_pca":
        raw, _, labels = normalized
        fmt = _apply_raw_pca_transform(raw, checkpoint["auxiliary_transform"])
        normalized = (raw, fmt, labels)
    loader = _loader(normalized, batch_size, False, checkpoint["seed"], device.type == "cuda")
    raw_model, _ = _load_raw_model(
        checkpoint["raw_checkpoint"], normalized[1].shape[1], device
    )
    model_spec = checkpoint["config"]["model"]
    model = PathlineFMTResidualClassifier3D(
        raw_model, fmt_dim=normalized[1].shape[1],
        embedding_dim=model_spec["embedding_dim"],
        auxiliary_dim=model_spec["auxiliary_dim"],
        residual_input=model_spec["residual_input"],
    ).to(device)
    state = model.state_dict()
    state.update(checkpoint["residual_state_dict"])
    model.load_state_dict(state)
    targets, raw_logits, residual_logits = _predict_components(model, loader, device)
    probabilities = _probabilities(
        raw_logits, residual_logits, float(checkpoint["alpha"])
    )
    return _classification_metrics(targets, probabilities, checkpoint["threshold"])


def _evaluate_baseline(checkpoint_path, split, batch_size, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = checkpoint["normalization"]
    raw, _, labels = split
    raw = ((raw - np.asarray(stats["raw_mean"]))
           / np.asarray(stats["raw_std"])).astype(np.float32)
    dummy = np.zeros((len(raw), 1), dtype=np.float32)
    loader = _loader((raw, dummy, labels), batch_size, False,
                     checkpoint["seed"], device.type == "cuda")
    source = checkpoint["config"]
    model = PathlineBinaryClassifier3D(
        variant=checkpoint["variant"], fmt_dim=1,
        temporal_width=source["model"]["temporal_width"],
        embedding_dim=source["model"]["embedding_dim"],
        auxiliary_dim=source["model"]["auxiliary_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    targets, probabilities = _predict(model.eval(), loader, device)
    return _classification_metrics(targets, probabilities, checkpoint["threshold"])


def evaluate_outer(config_path):
    spec = _load_spec(config_path)
    output_root = Path(spec["output_root"])
    selection_path = output_root / "selected_candidate.json"
    if not selection_path.exists():
        raise FileNotFoundError("select a candidate before opening outer ordinals")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidate = dict(selection["candidate"])
    candidate["index"] = int(selection["candidate_index"])
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    outer_dir = output_root / "outer_development_holdout"
    marker = outer_dir / "audit.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["selection_sha256"] != selection_hash:
            raise RuntimeError("outer holdout was already opened for another candidate")
        print(marker.read_text(encoding="utf-8"))
        return marker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in spec["datasets"]:
        records = _load_records(spec, dataset, candidate, spec["outer_ordinals"])
        split = _stack_split(records, spec["outer_ordinals"])
        for seed in spec["screen_seeds"]:
            baseline_dir = Path(spec["output_root"]) / "baselines" / dataset / "checkpoints"
            for variant in ("raw", "raw_wide"):
                metrics = _evaluate_baseline(
                    baseline_dir / f"{dataset}_{variant}_seed{int(seed)}.pt",
                    split, spec["training"]["batch_size"], device,
                )
                rows.append({"dataset": dataset, "seed": int(seed),
                             "method": variant, **metrics})
            candidate_root = (
                output_root / "candidates" / candidate["id"] / dataset
                / f"seed{int(seed)}"
            )
            for source, method in (("raw_pca", "raw_pca_residual"),
                                   ("fmt", "fmt_residual")):
                checkpoint = _candidate_checkpoint_path(
                    candidate_root, source, dataset, seed
                )
                metrics = _evaluate_residual(
                    checkpoint, split, spec["training"]["batch_size"], device
                )
                rows.append({"dataset": dataset, "seed": int(seed),
                             "method": method, **metrics})
    outer_dir.mkdir(parents=True, exist_ok=True)
    per_run = outer_dir / "per_run.csv"
    with per_run.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for dataset in spec["datasets"]:
        summary[dataset] = {}
        selected_rows = [row for row in rows if row["dataset"] == dataset]
        for method in ("raw", "raw_wide", "raw_pca_residual", "fmt_residual"):
            method_rows = [row for row in selected_rows if row["method"] == method]
            summary[dataset][method] = {
                metric: float(np.mean([row[metric] for row in method_rows]))
                for metric in ("f1", "average_precision")
            }
        strong = {
            metric: max(summary[dataset][method][metric]
                        for method in ("raw", "raw_wide"))
            for metric in ("f1", "average_precision")
        }
        summary[dataset]["gains"] = {
            f"{metric}_vs_raw_pca": (
                summary[dataset]["fmt_residual"][metric]
                - summary[dataset]["raw_pca_residual"][metric]
            )
            for metric in ("f1", "average_precision")
        }
        summary[dataset]["gains"].update({
            f"{metric}_vs_strong_raw": (
                summary[dataset]["fmt_residual"][metric] - strong[metric]
            )
            for metric in ("f1", "average_precision")
        })
    audit = {
        "experiment": spec["experiment"],
        "selection_sha256": selection_hash,
        "selected_candidate": candidate,
        "selection_used_only": "development ordinal 3",
        "outer_ordinals": list(spec["outer_ordinals"]),
        "old_task5_confirmation_read": False,
        "summary": summary,
    }
    marker.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(marker.read_text(encoding="utf-8"))
    return marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("candidate", "select", "outer"), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "candidate":
        if args.dataset is None or args.candidate_index is None or args.seed is None:
            parser.error("candidate mode requires --dataset, --candidate-index, and --seed")
        run_candidate(args.config, args.dataset, args.candidate_index, args.seed)
    elif args.mode == "select":
        select_candidate(args.config)
    else:
        evaluate_outer(args.config)
