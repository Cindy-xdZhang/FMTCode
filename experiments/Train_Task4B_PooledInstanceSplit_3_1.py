"""Train the pooled channel+TBL four-class experiment (mainExp_Task4B_PooledInstanceSplit_3.1).

The training loop, exact class-balanced batches, validation-selected in-memory
best state and single test evaluation are reused unchanged from the frozen 1.2
trainer.  This driver validates the pooled cache (buffer certificates, instance
purity, encoder contract), normalizes on training rows only, runs every variant
and seed, and adds per-volume, hairpin-membership and per-instance diagnostics
computed from the saved predictions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from experiments.Train_Task4B_FourClassClassifier_1_1 import (
    _normalize_train_only,
    _summarize,
    _train_one,
    _write_csv,
)
from FMT_Utils.Task4B_PooledSplit_3D import SPLIT_CODES, SPLIT_NAMES
from FMT_Utils.Task4B_ProxyLabels_3D import CLASS_NAMES

DEFAULT_CONFIG = Path("config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml")
VOLUME_NAMES = {0: "channel", 1: "tbl"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _confusion(targets: np.ndarray, predicted: np.ndarray, k: int = 4) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (targets.astype(np.int64), predicted.astype(np.int64)), 1)
    return matrix


def _f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0)


def _basic_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    matrix = _confusion(targets, predicted)
    f1 = _f1_from_confusion(matrix)
    recall = np.where(matrix.sum(axis=1) > 0, np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1), 0.0)
    present = np.flatnonzero(matrix.sum(axis=1) > 0)
    ap = []
    for class_id in present:
        ap.append(float(average_precision_score(targets == class_id, probabilities[:, class_id])))
    return {
        "row_count": int(len(targets)),
        "macro_f1": float(f1[present].mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_average_precision_ovr": float(np.mean(ap)),
        "per_class_f1": {CLASS_NAMES[c]: float(f1[c]) for c in range(4)},
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
    }


def _hairpin_diagnostics(targets, probabilities, vortex_ids) -> dict:
    predicted = probabilities.argmax(axis=1)
    hairpin = targets >= 2
    to_ordinary = float(np.mean(predicted[hairpin] < 2)) if hairpin.any() else None
    ordinary_to_hairpin = float(np.mean(predicted[~hairpin] >= 2)) if (~hairpin).any() else None
    # membership F1: head/limb versus ordinary
    member_pred = predicted >= 2
    tp = np.count_nonzero(member_pred & hairpin)
    fp = np.count_nonzero(member_pred & ~hairpin)
    fn = np.count_nonzero(~member_pred & hairpin)
    membership_f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
    # known-mask head/limb: restrict argmax to classes 2,3 inside hairpin rows
    known = probabilities[hairpin][:, 2:].argmax(axis=1) + 2
    known_matrix = _confusion(targets[hairpin] - 2, known - 2, k=2)
    known_f1 = _f1_from_confusion(known_matrix)
    known_present = np.flatnonzero(known_matrix.sum(axis=1) > 0)
    # VortexId-equal head/limb macro-F1 with the full four-class prediction
    per_instance = []
    for instance in np.unique(vortex_ids[hairpin]):
        rows = vortex_ids == instance
        matrix = _confusion(targets[rows], predicted[rows])
        f1 = _f1_from_confusion(matrix)
        present = [c for c in (2, 3) if matrix[c].sum() > 0]
        per_instance.append(float(np.mean([f1[c] for c in present])))
    return {
        "hairpin_to_ordinary_error_rate": to_ordinary,
        "ordinary_to_hairpin_error_rate": ordinary_to_hairpin,
        "hairpin_membership_f1": membership_f1,
        "known_hairpin_mask_head_limb_macro_f1": float(known_f1[known_present].mean()),
        "vortex_id_equal_head_limb_macro_f1": float(np.mean(per_instance)) if per_instance else None,
        "test_instance_count": int(len(per_instance)),
    }


def _extended_row_metrics(prediction_path: Path, cache: dict) -> dict:
    with np.load(prediction_path) as data:
        indices = np.asarray(data["test_source_indices"], dtype=np.int64)
        targets = np.asarray(data["test_targets"], dtype=np.int64)
        probabilities = np.asarray(data["test_probabilities"], dtype=np.float64)
    expected = np.flatnonzero(cache["split_codes"] == SPLIT_CODES["test"])
    if not np.array_equal(np.sort(indices), expected):
        raise RuntimeError(f"{prediction_path.name}: test indices differ from cache test rows")
    if not np.array_equal(targets, cache["labels"][indices]):
        raise RuntimeError(f"{prediction_path.name}: test targets differ from cache labels")
    ids = cache["vortex_ids"][indices]
    volumes = cache["volume_codes"][indices]
    out = {"pooled": {**_basic_metrics(targets, probabilities), **_hairpin_diagnostics(targets, probabilities, ids)}}
    for code, name in VOLUME_NAMES.items():
        mask = volumes == code
        out[name] = {
            **_basic_metrics(targets[mask], probabilities[mask]),
            **_hairpin_diagnostics(targets[mask], probabilities[mask], ids[mask]),
        }
    return out


def _validate_cache(spec: dict, cache_path: Path) -> tuple[dict, dict, str]:
    sha = _sha256(cache_path)
    expected = spec.get("cache_sha256")
    if expected and str(expected).lower() != sha:
        raise RuntimeError("pooled cache SHA-256 differs from config")
    with np.load(cache_path) as data:
        cache = {
            "raw_features": np.asarray(data["raw_features"], dtype=np.float32),
            "fmt_features": np.asarray(data["fmt_features"], dtype=np.float32),
            "labels": np.asarray(data["labels"], dtype=np.int8),
            "split_codes": np.asarray(data["split_codes"], dtype=np.int8),
            "volume_codes": np.asarray(data["volume_codes"], dtype=np.int8),
            "vortex_ids": np.asarray(data["vortex_ids"], dtype=np.int32),
            "seeds_xyz": np.asarray(data["seeds_xyz"], dtype=np.float64),
            "distance_to_test": np.asarray(data["distance_to_test"], dtype=np.float64),
        }
        metadata = json.loads(str(data["metadata_json"]))
    certificates = metadata["buffer_certificates"]
    if certificates.get("all_certified") is not True:
        raise RuntimeError(f"pooled cache buffer certificates are not all certified: {certificates}")
    for volume_name, code in (("channel", 0), ("tbl", 1)):
        certificate = certificates[volume_name]
        rows = cache["volume_codes"] == code
        nontest = rows & (cache["split_codes"] != SPLIT_CODES["test"])
        if nontest.any() and cache["distance_to_test"][nontest].min() < float(certificate["buffer_distance"]):
            raise RuntimeError(f"{volume_name}: a cached non-test row violates the buffer distance")
        hairpin = rows & (cache["vortex_ids"] > 0)
        for instance in np.unique(cache["vortex_ids"][hairpin]):
            codes = np.unique(cache["split_codes"][hairpin & (cache["vortex_ids"] == instance)])
            if len(codes) != 1:
                raise RuntimeError(f"{volume_name}: instance {instance} spans several splits")
    for key in ("num_freq", "mode", "include_chirality", "neighbor_pool", "neighbor_scale",
                "neighbor_weight_after_train_standardization"):
        if metadata["encoder"].get(key) != spec["encoder"][key]:
            raise RuntimeError(f"encoder contract differs from cache for {key}")
    if cache["fmt_features"].shape[1] != int(spec["encoder"]["expected_feature_dim"]):
        raise RuntimeError("FMT width differs from config")
    return cache, metadata, sha


def run(config_path: str | Path, *, output_override: str | None = None,
        cache_override: str | None = None, smoke: bool = False) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if smoke:
        spec = copy.deepcopy(spec)
        spec["experiment"] = spec["experiment"] + "_smoke"
        spec["training"]["seeds"] = [7068]
        spec["training"]["max_epochs"] = 2
        spec["training"]["patience"] = 2
        spec["output_dir"] = "outputs/mainExp_Task4B_PooledInstanceSplit_3.1/smoke"
        spec["cache"]["path"] = "outputs/mainExp_Task4B_PooledInstanceSplit_3.1/smoke/cache/task4b_pooled_cache.npz"
        spec["runtime_mode"] = "smoke_not_research_result"
    output_dir = Path(output_override or spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "config_snapshot.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if json.dumps(previous, sort_keys=True) != json.dumps(spec, sort_keys=True):
            raise RuntimeError("training config changed; use a new experiment version")
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    cache_path = Path(cache_override or spec["cache"]["path"])
    cache, metadata, cache_sha = _validate_cache(spec, cache_path)
    raw, fmt, normalization = _normalize_train_only(
        cache["raw_features"], cache["fmt_features"], cache["split_codes"],
        sampled_steps=int(metadata["streamlines"]["sampled_steps"]), encoder_spec=spec["encoder"],
    )
    np.savez_compressed(output_dir / "normalization_train_only.npz", **normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    results_path = output_dir / "per_run_metrics.csv"
    if results_path.exists():
        results_path.unlink()
    extended = {}
    for variant in spec["variants"]:
        for seed in spec["training"]["seeds"]:
            row = _train_one(spec, raw, fmt, cache["labels"], cache["split_codes"],
                             variant=str(variant), seed=int(seed), device=device, output_dir=output_dir)
            ext = _extended_row_metrics(Path(row["prediction_path"]), cache)
            for scope, values in ext.items():
                for key in ("macro_f1", "balanced_accuracy", "macro_average_precision_ovr",
                            "hairpin_to_ordinary_error_rate", "hairpin_membership_f1",
                            "known_hairpin_mask_head_limb_macro_f1", "vortex_id_equal_head_limb_macro_f1"):
                    row[f"ext_{scope}_{key}"] = values[key]
            extended[f"{variant}_seed{seed}"] = ext
            rows.append(row)
            _write_csv(results_path, [row])

    aggregate = _summarize(spec, rows)
    aggregate["extended"] = _aggregate_extended(spec, rows)
    payload = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_sha,
        "cache_buffer_certificates": metadata["buffer_certificates"],
        "cache_split_class_counts": metadata["split_class_counts"],
        "instance_assignment": {v: metadata["volumes"][v]["instance_assignment"] for v in ("channel", "tbl")},
        "device": {"type": device.type,
                   "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                   "torch_version": torch.__version__},
        "taxonomy": {"classes": {str(i): n for i, n in enumerate(CLASS_NAMES)}},
        "selection": "validation macro-F1 on held-out validation instances; test never selects epoch or hyperparameters",
        "checkpoint_policy": "best state retained in memory only; no .pt/.pth/.ckpt is written",
        "runs": rows,
        "extended_test_metrics": extended,
        "aggregate": aggregate,
        "interpretation_boundary": (
            "Targets are deterministic proxy labels from one channel and one TBL volume. "
            "Test rows belong to hairpin instances (and their Voronoi neighbourhoods) that never "
            "entered fitting or validation; a one-radius buffer removed nearby fitting cubes. "
            "Results quantify proxy reproduction on held-out instances of these two volumes only."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    return output_dir


def _aggregate_extended(spec: dict, rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k.startswith("ext_")]
    by_variant = {}
    for variant in spec["variants"]:
        selected = [r for r in rows if r["variant"] == variant]
        item = {}
        for key in keys:
            values = np.asarray([r[key] for r in selected if r[key] is not None], dtype=np.float64)
            if len(values):
                item[key] = {"mean": float(values.mean()),
                             "std": float(values.std(ddof=1) if len(values) > 1 else 0.0)}
        by_variant[variant] = item
    paired = {}
    for baseline in ("raw", "raw_wide"):
        if baseline not in spec["variants"] or "raw_fmt" not in spec["variants"]:
            continue
        base = {int(r["seed"]): r for r in rows if r["variant"] == baseline}
        target = {int(r["seed"]): r for r in rows if r["variant"] == "raw_fmt"}
        common = sorted(set(base) & set(target))
        for key in keys:
            diffs = [target[s][key] - base[s][key] for s in common
                     if target[s][key] is not None and base[s][key] is not None]
            if diffs:
                diffs = np.asarray(diffs, dtype=np.float64)
                paired[f"raw_fmt_minus_{baseline}_{key}"] = {
                    "mean": float(diffs.mean()),
                    "std": float(diffs.std(ddof=1) if len(diffs) > 1 else 0.0),
                    "positive_seeds": int(np.count_nonzero(diffs > 0)),
                    "seed_count": int(len(diffs)),
                }
    return {"variants": by_variant, "paired_comparisons": paired}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--cache")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.config, output_override=args.output_dir, cache_override=args.cache, smoke=args.smoke)
