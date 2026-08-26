"""Controlled supervised Task3 comparison: Raw, Raw-wide, and Raw+FMT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from FMT_Utils.DFT_FMT_3D import (
    fmt_feature_indices_3d, pathline_velocity_gradient_dft_features_3d,
    time_local_gram_dft_features_3d,
)
from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D, trainable_parameter_count,
)


def _portable_basename(path_value):
    """Return a cache basename written with either Windows or POSIX separators."""
    return Path(str(path_value).replace("\\", "/")).name


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _config_without_seeds(spec):
    """Return the config identity; adding replicate seeds is the only safe edit."""
    copied = json.loads(json.dumps(spec))
    copied["training"].pop("seeds", None)
    return copied


def _validate_config_snapshot(spec, snapshot_path):
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.exists():
        return
    previous = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if _config_without_seeds(previous) != _config_without_seeds(spec):
        raise RuntimeError(
            f"configuration changed for existing output {snapshot_path.parent}; "
            "use a new experiment version/output directory"
        )


def _load_dataset(source_dir, label_dir, sampled_steps, fmt_subset,
                  required_ordinals=None, gram_num_freq=6,
                  expected_slices=10):
    source_paths = sorted(Path(source_dir).glob("slice_*.npz"))
    if len(source_paths) != int(expected_slices):
        raise RuntimeError(
            f"expected {expected_slices} source slices in {source_dir}, "
            f"found {len(source_paths)}"
        )
    required = None if required_ordinals is None else {
        int(value) for value in required_ordinals
    }
    extended = {
        "time_local_gram", "all_plus_time_local_gram",
        "all_plus_kinematic", "all_plus_gram_kinematic",
    }
    if fmt_subset in extended:
        fmt_indices = fmt_feature_indices_3d("all")
    else:
        fmt_indices = fmt_feature_indices_3d(fmt_subset)
    records = []
    for ordinal, source_path in enumerate(source_paths):
        if required is not None and ordinal not in required:
            continue
        label_path = Path(label_dir) / source_path.name
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        with np.load(source_path) as source, np.load(label_path) as labels:
            raw = np.asarray(source["raw_features"], dtype=np.float32)
            cached_fmt = np.asarray(source["fmt_features"], dtype=np.float32)[:, fmt_indices]
            target = np.asarray(labels["labels"], dtype=np.float32)
            metadata = json.loads(str(labels["metadata_json"]))
        if raw.shape != (len(target), 7 * int(sampled_steps) * 3):
            raise ValueError(f"unexpected raw shape {raw.shape} in {source_path}")
        if fmt_subset in extended:
            primitives = raw.reshape(-1, 7, int(sampled_steps), 3)
            parts = [] if fmt_subset == "time_local_gram" else [cached_fmt]
            if "gram" in fmt_subset or fmt_subset == "time_local_gram":
                parts.append(time_local_gram_dft_features_3d(
                    primitives, num_freq=int(gram_num_freq),
                ).astype(np.float32))
            if "kinematic" in fmt_subset:
                parts.append(pathline_velocity_gradient_dft_features_3d(
                    primitives, num_freq=6,
                ).astype(np.float32))
            fmt = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        else:
            fmt = cached_fmt
        if fmt.shape[0] != len(target):
            raise ValueError(f"FMT/label length mismatch in {source_path}")
        # Label caches may be generated on Windows and consumed on Ibex/Linux.
        # pathlib.Path on POSIX does not treat a backslash as a separator.
        cached_source_name = _portable_basename(metadata["source_cache"])
        if cached_source_name != source_path.name:
            raise ValueError(f"label/source mismatch for {source_path}")
        records.append((
            raw.reshape(-1, 7, int(sampled_steps), 3), fmt, target, ordinal,
            metadata,
        ))
    return records


def _stack_split(records, ordinals):
    selected = [record for record in records if record[3] in set(ordinals)]
    if len(selected) != len(ordinals):
        raise ValueError(f"split {ordinals} selected {len(selected)} slices")
    return tuple(np.concatenate([record[index] for record in selected], axis=0)
                 for index in range(3))


def _normalize_train_only(train, validation, test=None):
    raw_train, fmt_train, _ = train
    raw_mean = raw_train.mean(axis=(0, 1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    raw_std = raw_train.std(axis=(0, 1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    raw_std = np.maximum(raw_std, 1e-6)
    fmt_mean = fmt_train.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    fmt_std = fmt_train.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    fmt_std = np.maximum(fmt_std, 1e-6)

    def transform(split):
        raw, fmt, labels = split
        raw = ((raw - raw_mean) / raw_std).astype(np.float32)
        fmt = ((fmt - fmt_mean) / fmt_std).astype(np.float32)
        if not np.isfinite(raw).all() or not np.isfinite(fmt).all():
            raise ValueError("non-finite normalized classifier input")
        return raw, fmt, labels.astype(np.float32)

    stats = {
        "raw_mean": raw_mean, "raw_std": raw_std,
        "fmt_mean": fmt_mean, "fmt_std": fmt_std,
    }
    transformed_test = None if test is None else transform(test)
    return transform(train), transform(validation), transformed_test, stats


def _loader(split, batch_size, shuffle, seed, pin_memory):
    raw, fmt, labels = split
    dataset = TensorDataset(
        torch.from_numpy(raw), torch.from_numpy(fmt), torch.from_numpy(labels)
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=0, pin_memory=bool(pin_memory), generator=generator,
    )


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    for raw, fmt, labels in loader:
        raw = raw.to(device, non_blocking=True)
        fmt = fmt.to(device, non_blocking=True)
        logits = model(raw, fmt if model.variant == "raw_fmt" else None)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(targets).astype(bool), np.concatenate(probabilities)


def _ranking_metrics(targets, probabilities):
    return {
        "average_precision": float(average_precision_score(targets, probabilities)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
    }


def _select_f1_threshold(targets, probabilities):
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _classification_metrics(targets, probabilities, threshold):
    predicted = probabilities >= float(threshold)
    return {
        **_ranking_metrics(targets, probabilities),
        "f1": float(f1_score(targets, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "precision": float(precision_score(targets, predicted, zero_division=0)),
        "recall": float(recall_score(targets, predicted, zero_division=0)),
        "predicted_positive_fraction": float(predicted.mean()),
    }


def _train_one(spec, dataset, variant, seed, splits, stats, device, output_dir):
    _set_seed(seed)
    train, validation, test = splits
    pin = device.type == "cuda"
    train_loader = _loader(train, spec["training"]["batch_size"], True, seed, pin)
    validation_loader = _loader(validation, spec["training"]["batch_size"], False, seed, pin)
    test_loader = None if test is None else _loader(
        test, spec["training"]["batch_size"], False, seed, pin
    )
    model = PathlineBinaryClassifier3D(
        variant=variant, fmt_dim=train[1].shape[1],
        temporal_width=spec["model"]["temporal_width"],
        embedding_dim=spec["model"]["embedding_dim"],
        auxiliary_dim=spec["model"]["auxiliary_dim"],
    ).to(device)
    parameters = trainable_parameter_count(model)
    positive = float(train[2].sum())
    negative = float(len(train[2]) - positive)
    if positive == 0 or negative == 0:
        raise RuntimeError("training labels contain only one class")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["training"]["learning_rate"]),
        weight_decay=float(spec["training"]["weight_decay"]),
    )
    best_score = -np.inf
    best_epoch = -1
    best_state = None
    stale = 0
    started = time.perf_counter()
    history_path = output_dir / "histories" / f"{dataset}_{variant}_seed{seed}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()
    for epoch in range(int(spec["training"]["max_epochs"])):
        model.train()
        total_loss = 0.0
        count = 0
        for raw, fmt, labels in train_loader:
            raw = raw.to(device, non_blocking=True)
            fmt = fmt.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(raw, fmt if variant == "raw_fmt" else None)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
        val_targets, val_probabilities = _predict(model, validation_loader, device)
        val_score = _ranking_metrics(val_targets, val_probabilities)["average_precision"]
        improved = val_score > best_score + float(spec["training"]["min_delta"])
        if improved:
            best_score = val_score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        _append_csv(history_path, {
            "epoch": epoch + 1, "train_loss": total_loss / count,
            "validation_average_precision": val_score,
            "is_best": int(improved), "stale_epochs": stale,
        })
        print(
            f"{dataset} {variant} seed={seed} epoch={epoch + 1:02d} "
            f"loss={total_loss / count:.5f} val_AP={val_score:.5f}", flush=True,
        )
        if stale >= int(spec["training"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    val_targets, val_probabilities = _predict(model, validation_loader, device)
    threshold = _select_f1_threshold(val_targets, val_probabilities)
    val_metrics = _classification_metrics(val_targets, val_probabilities, threshold)
    if test_loader is None:
        test_metrics = None
    else:
        test_targets, test_probabilities = _predict(model, test_loader, device)
        test_metrics = _classification_metrics(
            test_targets, test_probabilities, threshold
        )
    checkpoint = output_dir / "checkpoints" / f"{dataset}_{variant}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "variant": variant, "dataset": dataset,
        "seed": seed, "threshold": threshold, "best_epoch": best_epoch,
        "parameter_count": parameters, "normalization": stats,
        "config": spec,
    }, checkpoint)
    result = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "parameter_count": parameters, "best_epoch": best_epoch,
        "validation_threshold": threshold,
        "train_positive_fraction": float(train[2].mean()),
        "validation_positive_fraction": float(validation[2].mean()),
        **{f"validation_{key}": value for key, value in val_metrics.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }
    if test_metrics is not None:
        result["test_positive_fraction"] = float(test[2].mean())
        result.update({f"test_{key}": value for key, value in test_metrics.items()})
    return result


def _completion_message(key, row):
    if "test_f1" in row:
        return (
            f"DONE {key}: test F1={row['test_f1']:.5f}, "
            f"AP={row['test_average_precision']:.5f}"
        )
    return (
        f"DONE {key}: validation F1={row['validation_f1']:.5f}, "
        f"AP={row['validation_average_precision']:.5f}; test disabled"
    )


def run(config_path):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.yaml"
    _validate_config_snapshot(spec, snapshot_path)
    snapshot_path.write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    results_path = output_dir / "per_run.csv"
    completed = set()
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as handle:
            completed = {
                (row["dataset"], row["variant"], int(row["seed"]))
                for row in csv.DictReader(handle)
            }
    device_name = spec["training"].get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    print(f"device={device}", flush=True)
    for dataset in spec["datasets"]:
        test_enabled = bool(spec.get("evaluation", {}).get("test_enabled", True))
        required_ordinals = set(spec["split"]["train_ordinals"]) | set(
            spec["split"]["validation_ordinals"]
        )
        if test_enabled:
            required_ordinals |= set(spec["split"]["test_ordinals"])
        records = _load_dataset(
            Path(spec["source_cache_root"]) / dataset,
            Path(spec["label_cache_root"]) / dataset,
            spec["sampled_steps"], spec["fmt_subset"], required_ordinals,
            gram_num_freq=spec.get("fmt_gram_num_freq", 6),
            expected_slices=spec.get("expected_slices", 10),
        )
        train = _stack_split(records, spec["split"]["train_ordinals"])
        validation = _stack_split(records, spec["split"]["validation_ordinals"])
        test = _stack_split(
            records, spec["split"]["test_ordinals"]
        ) if test_enabled else None
        train, validation, test, stats = _normalize_train_only(train, validation, test)
        splits = (train, validation, test)
        for seed in spec["training"]["seeds"]:
            for variant in spec["variants"]:
                key = (dataset, variant, int(seed))
                if key in completed:
                    print(f"cached result: {key}", flush=True)
                    continue
                row = _train_one(
                    spec, dataset, variant, int(seed), splits, stats, device, output_dir
                )
                _append_csv(results_path, row)
                completed.add(key)
                print(_completion_message(key, row), flush=True)
    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
