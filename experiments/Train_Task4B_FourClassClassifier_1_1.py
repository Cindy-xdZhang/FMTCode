"""Train Raw/FMT four-class channel-flow baselines for Task4-b."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset
import yaml

from FMT_Utils.Task4A_PerVortexClustering_3D import fmt_base_feature_width
from FMT_Utils.Task4B_Classifier_3D import (
    PathlineMulticlassClassifier3D,
    trainable_parameter_count,
)
from FMT_Utils.Task4B_ProxyLabels_3D import CLASS_NAMES


DEFAULT_CONFIG = Path("config/mainExp_Task4B_1.2.yaml")
SPLIT_NAMES = ("train", "validation", "test")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cache_sha256(path: Path, expected_sha256: str) -> str:
    """Reject a cache that is not the frozen Task4-b training artifact."""

    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("cache_sha256 must be a 64-character hexadecimal digest")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            "training cache SHA-256 differs from the frozen config: "
            f"expected={expected}, actual={actual}"
        )
    return actual


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    exists = path.exists()
    fields = list(rows[0])
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class ExactBalancedMulticlassBatchSampler(Sampler[list[int]]):
    """Emit deterministic batches with equal quotas from every class."""

    def __init__(
        self,
        labels: np.ndarray,
        *,
        batch_size: int,
        num_classes: int,
        seed: int,
    ):
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size % self.num_classes != 0:
            raise ValueError("balanced batch size must be divisible by class count")
        self.quota = self.batch_size // self.num_classes
        self.by_class = [
            np.flatnonzero(self.labels == class_id)
            for class_id in range(self.num_classes)
        ]
        if any(len(indices) == 0 for indices in self.by_class):
            raise ValueError("balanced sampler requires every class")
        self.batch_count = int(
            np.ceil(max(len(indices) for indices in self.by_class) / self.quota)
        )

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        needed = self.batch_count * self.quota
        streams = []
        for indices in self.by_class:
            cycles = []
            while sum(len(cycle) for cycle in cycles) < needed:
                cycles.append(rng.permutation(indices))
            streams.append(np.concatenate(cycles)[:needed])
        self.epoch += 1
        for batch_index in range(self.batch_count):
            start = batch_index * self.quota
            stop = start + self.quota
            batch = np.concatenate([stream[start:stop] for stream in streams])
            rng.shuffle(batch)
            yield batch.tolist()


def _normalize_train_only(
    raw_features: np.ndarray,
    fmt_features: np.ndarray,
    split_codes: np.ndarray,
    *,
    sampled_steps: int,
    encoder_spec: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.asarray(raw_features, dtype=np.float32).reshape(
        -1, 7, int(sampled_steps), 3
    )
    fmt = np.asarray(fmt_features, dtype=np.float32)
    train = np.asarray(split_codes) == 0
    raw_mean = raw[train].mean(
        axis=(0, 1, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    raw_std = raw[train].std(
        axis=(0, 1, 2), keepdims=True, dtype=np.float64
    ).astype(np.float32)
    raw_std = np.maximum(raw_std, 1e-6)
    fmt_mean = fmt[train].mean(axis=0, keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    fmt_std = fmt[train].std(axis=0, keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    fmt_std = np.maximum(fmt_std, 1e-6)
    raw = ((raw - raw_mean) / raw_std).astype(np.float32)
    fmt = ((fmt - fmt_mean) / fmt_std).astype(np.float32)
    base_width = fmt_base_feature_width(
        int(encoder_spec["num_freq"]),
        str(encoder_spec["mode"]),
        bool(encoder_spec["include_chirality"]),
    )
    fmt[:, base_width:] *= float(
        encoder_spec["neighbor_weight_after_train_standardization"]
    )
    if not np.isfinite(raw).all() or not np.isfinite(fmt).all():
        raise RuntimeError("normalization produced non-finite features")
    return raw, fmt, {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "fmt_mean": fmt_mean,
        "fmt_std": fmt_std,
        "fmt_base_width": base_width,
    }


def _loader(
    raw: np.ndarray,
    fmt: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    balanced: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(raw[indices]),
        torch.from_numpy(fmt[indices]),
        torch.from_numpy(labels[indices].astype(np.int64)),
        torch.from_numpy(indices.astype(np.int64)),
    )
    common = {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    if balanced:
        sampler = ExactBalancedMulticlassBatchSampler(
            labels[indices],
            batch_size=int(batch_size),
            num_classes=4,
            seed=int(seed),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **common,
    )


@torch.no_grad()
def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device, variant: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    targets, probabilities, source_indices = [], [], []
    for raw, fmt, labels, indices in loader:
        raw = raw.to(device, non_blocking=True)
        fmt = fmt.to(device, non_blocking=True)
        logits = model(
            raw,
            fmt if variant in {"fmt_only", "raw_fmt"} else None,
        )
        targets.append(labels.numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        source_indices.append(indices.numpy())
    return (
        np.concatenate(targets),
        np.concatenate(probabilities),
        np.concatenate(source_indices),
    )


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = np.argmax(probabilities, axis=1)
    labels = np.arange(4)
    per_f1 = f1_score(
        targets, predicted, labels=labels, average=None, zero_division=0
    )
    per_precision = precision_score(
        targets, predicted, labels=labels, average=None, zero_division=0
    )
    per_recall = recall_score(
        targets, predicted, labels=labels, average=None, zero_division=0
    )
    one_hot = np.eye(4, dtype=np.float64)[targets]
    per_ap = average_precision_score(one_hot, probabilities, average=None)
    return {
        "sample_count": int(len(targets)),
        "macro_f1": float(np.mean(per_f1)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_average_precision_ovr": float(np.mean(per_ap)),
        "per_class_f1": {
            name: float(per_f1[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_precision": {
            name: float(per_precision[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_recall": {
            name: float(per_recall[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "per_class_average_precision_ovr": {
            name: float(per_ap[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "support": {
            name: int(np.count_nonzero(targets == index))
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(
            targets, predicted, labels=labels
        ).tolist(),
    }


def _flatten_run_metrics(prefix: str, metrics: dict, row: dict) -> None:
    for key in ("sample_count", "macro_f1", "balanced_accuracy", "macro_average_precision_ovr"):
        row[f"{prefix}_{key}"] = metrics[key]
    for class_name in CLASS_NAMES:
        row[f"{prefix}_f1_{class_name}"] = metrics["per_class_f1"][class_name]
        row[f"{prefix}_precision_{class_name}"] = metrics[
            "per_class_precision"
        ][class_name]
        row[f"{prefix}_recall_{class_name}"] = metrics["per_class_recall"][
            class_name
        ]
        row[f"{prefix}_ap_{class_name}"] = metrics[
            "per_class_average_precision_ovr"
        ][class_name]
        row[f"{prefix}_support_{class_name}"] = metrics["support"][class_name]
    row[f"{prefix}_confusion_matrix_true_rows_predicted_columns"] = metrics[
        "confusion_matrix_true_rows_predicted_columns"
    ]


def _train_one(
    spec: dict,
    raw: np.ndarray,
    fmt: np.ndarray,
    labels: np.ndarray,
    split_codes: np.ndarray,
    *,
    variant: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    _set_seed(seed)
    training = spec["training"]
    model_spec = spec["model"]
    train_indices = np.flatnonzero(split_codes == 0)
    validation_indices = np.flatnonzero(split_codes == 1)
    test_indices = np.flatnonzero(split_codes == 2)
    train_loader = _loader(
        raw,
        fmt,
        labels,
        train_indices,
        batch_size=int(training["batch_size"]),
        device=device,
        balanced=True,
        seed=seed,
    )
    validation_loader = _loader(
        raw,
        fmt,
        labels,
        validation_indices,
        batch_size=int(training["evaluation_batch_size"]),
        device=device,
        balanced=False,
        seed=seed,
    )
    test_loader = _loader(
        raw,
        fmt,
        labels,
        test_indices,
        batch_size=int(training["evaluation_batch_size"]),
        device=device,
        balanced=False,
        seed=seed,
    )
    model = PathlineMulticlassClassifier3D(
        variant=variant,
        fmt_dim=fmt.shape[1],
        num_classes=4,
        temporal_width=int(model_spec["temporal_width"]),
        embedding_dim=int(model_spec["embedding_dim"]),
        auxiliary_dim=int(model_spec["auxiliary_dim"]),
        dropout=float(model_spec["dropout"]),
    ).to(device)
    parameter_count = trainable_parameter_count(model)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training["label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history_path = output_dir / "histories" / f"{variant}_seed{seed}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_raw, batch_fmt, batch_labels, _ in train_loader:
            batch_raw = batch_raw.to(device, non_blocking=True)
            batch_fmt = batch_fmt.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch_raw,
                batch_fmt if variant in {"fmt_only", "raw_fmt"} else None,
            )
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            sample_count += len(batch_labels)
        val_targets, val_probabilities, _ = _predict(
            model, validation_loader, device, variant
        )
        val_metrics = _metrics(val_targets, val_probabilities)
        score = float(val_metrics["macro_f1"])
        improved = score > best_score + float(training["min_delta"])
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        _write_csv(
            history_path,
            [
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / sample_count,
                    "validation_macro_f1": score,
                    "validation_balanced_accuracy": val_metrics[
                        "balanced_accuracy"
                    ],
                    "validation_macro_average_precision_ovr": val_metrics[
                        "macro_average_precision_ovr"
                    ],
                    "is_best": int(improved),
                    "stale_epochs": stale,
                }
            ],
        )
        print(
            f"{variant} seed={seed} epoch={epoch:03d} "
            f"loss={loss_sum / sample_count:.5f} val_macro_F1={score:.5f}",
            flush=True,
        )
        if stale >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError("training did not produce an in-memory best state")
    model.load_state_dict(best_state)
    val_targets, val_probabilities, val_source = _predict(
        model, validation_loader, device, variant
    )
    test_targets, test_probabilities, test_source = _predict(
        model, test_loader, device, variant
    )
    validation_metrics = _metrics(val_targets, val_probabilities)
    test_metrics = _metrics(test_targets, test_probabilities)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{variant}_seed{seed}.npz"
    np.savez_compressed(
        prediction_path,
        validation_source_indices=val_source,
        validation_targets=val_targets,
        validation_probabilities=val_probabilities,
        test_source_indices=test_source,
        test_targets=test_targets,
        test_probabilities=test_probabilities,
    )
    row = {
        "variant": variant,
        "seed": int(seed),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_path": str(prediction_path.resolve()),
        "checkpoint_policy": "best_state_in_memory_only_no_model_file",
    }
    _flatten_run_metrics("validation", validation_metrics, row)
    _flatten_run_metrics("test", test_metrics, row)
    del best_state, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _summarize(spec: dict, rows: list[dict]) -> dict:
    variants = [str(value) for value in spec["variants"]]
    summary = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        item = {
            "run_count": len(selected),
            "parameter_count": int(selected[0]["parameter_count"]),
        }
        for metric in (
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_macro_average_precision_ovr",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(
                np.std(values, ddof=1 if len(values) > 1 else 0)
            )
        for class_name in CLASS_NAMES:
            values = np.asarray(
                [row[f"test_f1_{class_name}"] for row in selected],
                dtype=np.float64,
            )
            item[f"test_f1_{class_name}_mean"] = float(np.mean(values))
        summary[variant] = item
    paired = {}
    if "raw" in variants and "raw_fmt" in variants:
        raw = {int(row["seed"]): row for row in rows if row["variant"] == "raw"}
        raw_fmt = {
            int(row["seed"]): row
            for row in rows
            if row["variant"] == "raw_fmt"
        }
        common = sorted(set(raw) & set(raw_fmt))
        for metric in (
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_macro_average_precision_ovr",
        ):
            differences = np.asarray(
                [raw_fmt[seed][metric] - raw[seed][metric] for seed in common]
            )
            paired[f"raw_fmt_minus_raw_{metric}"] = {
                "mean": float(np.mean(differences)),
                "std": float(
                    np.std(differences, ddof=1 if len(differences) > 1 else 0)
                ),
                "per_seed": {
                    str(seed): float(raw_fmt[seed][metric] - raw[seed][metric])
                    for seed in common
                },
            }
    return {"variants": summary, "paired_comparisons": paired}


def run(
    config_path: str | Path,
    *,
    output_override: str | None = None,
    smoke_test: bool = False,
) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if smoke_test:
        spec = copy.deepcopy(spec)
        spec["experiment"] = "Verify_Task4B_ClassifierSmoke_1.2"
        spec["variants"] = ["raw", "raw_wide", "fmt_only", "raw_fmt"]
        spec["training"]["seeds"] = [7068]
        spec["training"]["max_epochs"] = 2
        spec["training"]["patience"] = 2
        spec["output_dir"] = "outputs/Verify_Task4B_ClassifierSmoke_1.2/channel_GTs"
        spec["runtime_mode"] = "smoke_test_not_research_result"
    output_dir = Path(output_override or spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "config_snapshot.yaml"
    if snapshot.exists():
        old = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if json.loads(json.dumps(old, sort_keys=True)) != json.loads(
            json.dumps(spec, sort_keys=True)
        ):
            raise RuntimeError("training config changed; use a new experiment version")
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    cache_path = Path(spec["cache"])
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache_sha256 = _validate_cache_sha256(cache_path, spec.get("cache_sha256", ""))
    with np.load(cache_path) as cache:
        raw_features = np.asarray(cache["raw_features"], dtype=np.float32)
        fmt_features = np.asarray(cache["fmt_features"], dtype=np.float32)
        labels = np.asarray(cache["labels"], dtype=np.int8)
        split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
        metadata = json.loads(str(cache["metadata_json"]))
    split_certificate = metadata.get("split_nonoverlap_certificate")
    if not isinstance(split_certificate, dict):
        raise RuntimeError("cache lacks a split non-overlap certificate")
    minimum_gap = float(split_certificate.get("minimum_cyclic_x_gap", np.nan))
    required_gap = float(split_certificate.get("required_minimum_gap", np.nan))
    if (
        split_certificate.get("certified") is not True
        or not np.isfinite(minimum_gap)
        or not np.isfinite(required_gap)
        or minimum_gap <= required_gap
    ):
        raise RuntimeError(
            "cache split non-overlap certificate is absent or invalid: "
            f"{split_certificate}"
        )
    cache_encoder = metadata.get("encoder", {})
    for key in (
        "num_freq",
        "mode",
        "include_chirality",
        "neighbor_pool",
        "neighbor_scale",
        "neighbor_weight_after_train_standardization",
    ):
        if key not in spec["encoder"] or cache_encoder.get(key) != spec["encoder"][key]:
            raise RuntimeError(
                f"supervised encoder contract differs from cache for {key}: "
                f"config={spec['encoder'].get(key)!r}, cache={cache_encoder.get(key)!r}"
            )
    expected_fmt_dim = int(spec["encoder"]["expected_feature_dim"])
    if fmt_features.ndim != 2 or fmt_features.shape[1] != expected_fmt_dim:
        raise RuntimeError(
            f"FMT cache width {fmt_features.shape} != expected {expected_fmt_dim}"
        )
    sampled_steps = int(metadata["streamlines"]["sampled_steps"])
    raw, fmt, normalization = _normalize_train_only(
        raw_features,
        fmt_features,
        split_codes,
        sampled_steps=sampled_steps,
        encoder_spec=spec["encoder"],
    )
    np.savez_compressed(
        output_dir / "normalization_train_only.npz", **normalization
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    results_path = output_dir / "per_run_metrics.csv"
    if results_path.exists():
        results_path.unlink()
    for variant in spec["variants"]:
        for seed in spec["training"]["seeds"]:
            row = _train_one(
                spec,
                raw,
                fmt,
                labels,
                split_codes,
                variant=str(variant),
                seed=int(seed),
                device=device,
                output_dir=output_dir,
            )
            rows.append(row)
            _write_csv(results_path, [row])
    aggregate = _summarize(spec, rows)
    payload = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_sha256,
        "cache_split_nonoverlap_certificate": split_certificate,
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
        },
        "taxonomy": {
            "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
            "non_vortex": "excluded before four-class training",
        },
        "split_counts": {
            name: {
                CLASS_NAMES[class_id]: int(
                    np.count_nonzero((split_codes == split_code) & (labels == class_id))
                )
                for class_id in range(4)
            }
            for split_code, name in enumerate(SPLIT_NAMES)
        },
        "selection": "validation macro-F1; test never selects epoch or hyperparameters",
        "checkpoint_policy": (
            "best state retained in memory only; no .pt/.pth/.ckpt is written"
        ),
        "runs": rows,
        "aggregate": aggregate,
        "interpretation_boundary": (
            "All targets are deterministic proxy labels from one steady channel "
            "volume. Metrics quantify proxy reproduction on a fixed stratified "
            "sample of spatially buffered slabs, not whole-field segmentation, "
            "manual anatomy or cross-time generalization."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.config,
        output_override=args.output_dir,
        smoke_test=args.smoke_test,
    )
