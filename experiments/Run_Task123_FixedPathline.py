"""Evaluate fixed 3D pathline settings on Task1, Task2, and Task3.

Only development ordinals 0--7 are opened by the three worker modes.  Every
variant is evaluated on the intersection of seeds whose seven pathlines are
valid for all registered variants.  Task-level feature and network recipes
remain frozen; only ``dt_scale``, integration steps, and sampled steps differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import feature_matrix, stack_features, stack_reference
from experiments.Run_Task1_3D_Main import _fit_score
from experiments.Run_Task2_3D_Main import _latent_score, _prepare_inputs
from experiments.Verify_HighReVAE import _train as train_vae
from experiments.Verify_Task3_FMTClassifier import (
    _normalize_train_only,
    _train_one as train_raw_backbone,
)
from experiments.Verify_Task3_FMTResidual import _train_one as train_residual


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_spec(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "cache_root", "cache_sources",
        "datasets", "families", "variants", "splits", "task1", "task2",
        "task3", "selection",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"fixed-pathline config misses keys: {missing}")
    if len(spec["datasets"]) != len(set(spec["datasets"])):
        raise ValueError("datasets must be unique")
    variant_ids = [str(row["id"]) for row in spec["variants"]]
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("variant ids must be unique")
    opened = sorted(int(value) for value in spec["selection"]["opened_ordinals"])
    authorized = sorted(
        {int(value) for key in ("train", "validation")
         for value in spec["splits"][key]}
    )
    if opened != authorized:
        raise ValueError(
            f"opened development ordinals {opened} differ from train/validation "
            f"union {authorized}"
        )
    confirmation = {int(value) for value in spec["splits"]["confirmation"]}
    if confirmation & set(opened):
        raise ValueError("development and confirmation ordinals overlap")
    if bool(spec["selection"].get("confirmation_opened", True)):
        raise ValueError("development search must declare confirmation_opened=false")
    return spec, path


def _variant(spec: dict, variant_id: str) -> dict:
    matches = [row for row in spec["variants"] if str(row["id"]) == variant_id]
    if len(matches) != 1:
        raise ValueError(f"variant {variant_id!r} matched {len(matches)} entries")
    return dict(matches[0])


def _cache_dir(spec: dict, variant_id: str, dataset: str) -> Path:
    staged = set(spec["cache_sources"].get("staged_new2", []))
    if dataset in staged:
        root = Path(spec["cache_root"])
    else:
        root = Path(spec["cache_sources"]["imported_old8"])
    return root / variant_id / dataset


def _source_index(metadata: dict) -> int:
    return int(metadata.get(
        "original_source_start_index", metadata["source_start_index"]
    ))


def load_common_records(
    spec: dict,
    variant_id: str,
    dataset: str,
    ordinals: Iterable[int],
) -> list[dict]:
    """Load one variant on the all-variant common-valid population."""
    variant_ids = [str(row["id"]) for row in spec["variants"]]
    expected = int(spec.get("expected_slices", 10))
    paths = {
        value: sorted(_cache_dir(spec, value, dataset).glob("slice_*.npz"))
        for value in variant_ids
    }
    for value, matches in paths.items():
        if len(matches) != expected:
            raise RuntimeError(
                f"expected {expected} cache slices for {value}/{dataset}, "
                f"found {len(matches)} in {_cache_dir(spec, value, dataset)}"
            )
    requested = [int(value) for value in ordinals]
    if len(requested) != len(set(requested)):
        raise ValueError(f"duplicate ordinals requested: {requested}")
    if any(value < 0 or value >= expected for value in requested):
        raise IndexError(f"ordinals outside [0,{expected}): {requested}")

    records = []
    for ordinal in requested:
        masks, references, metadata_by_variant = {}, {}, {}
        for value in variant_ids:
            with np.load(paths[value][ordinal], allow_pickle=False) as data:
                masks[value] = np.asarray(data["valid_mask"], dtype=bool)
                references[value] = np.asarray(data["reference"], dtype=bool)
                metadata_by_variant[value] = json.loads(
                    str(data["metadata_json"].item())
                )
                if int(masks[value].sum()) != len(references[value]):
                    raise RuntimeError(
                        f"valid/reference count mismatch in {paths[value][ordinal]}"
                    )
        shapes = {mask.shape for mask in masks.values()}
        if len(shapes) != 1:
            raise RuntimeError(
                f"valid-mask shape differs for {dataset} ordinal {ordinal}: {shapes}"
            )
        indices = {_source_index(value) for value in metadata_by_variant.values()}
        if len(indices) != 1:
            raise RuntimeError(
                f"source index differs for {dataset} ordinal {ordinal}: {indices}"
            )
        common = np.logical_and.reduce(list(masks.values()))
        current_mask = masks[variant_id]
        selector = common[current_mask]
        if int(selector.sum()) < 100:
            raise RuntimeError(
                f"only {int(selector.sum())} common seeds for "
                f"{variant_id}/{dataset}/ordinal{ordinal}"
            )
        common_references = [
            references[value][common[masks[value]]] for value in variant_ids
        ]
        if not all(np.array_equal(common_references[0], value)
                   for value in common_references[1:]):
            raise RuntimeError(
                f"IVD reference differs across variants for {dataset} "
                f"ordinal {ordinal}"
            )
        current_path = paths[variant_id][ordinal]
        with np.load(current_path, allow_pickle=False) as data:
            raw = np.asarray(data["raw_features"], dtype=np.float32)
            fmt = np.asarray(data["fmt_features"], dtype=np.float32)
            reference = np.asarray(data["reference"], dtype=bool)
        if not (len(raw) == len(fmt) == len(reference) == int(current_mask.sum())):
            raise RuntimeError(f"cache row count mismatch in {current_path}")
        metadata = dict(metadata_by_variant[variant_id])
        metadata.update({
            "common_valid_primitives": int(selector.sum()),
            "common_valid_fraction": float(common.mean()),
            "original_source_start_index": next(iter(indices)),
        })
        records.append({
            "path": current_path,
            "ordinal": ordinal,
            "raw": raw[selector],
            "fmt": fmt[selector],
            "reference": reference[selector],
            "metadata": metadata,
            "features": {},
        })
    return records


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _identity(spec: dict, config_path: Path, task: str, variant: str,
              dataset: str, seed: int | None) -> dict:
    return {
        "experiment": spec["experiment"],
        "config_sha256": _sha256(config_path),
        "task": task,
        "variant": variant,
        "dataset": dataset,
        "family": spec["families"][dataset],
        "seed": seed,
        "opened_ordinals": sorted(
            int(value) for value in spec["selection"]["opened_ordinals"]
        ),
        "confirmation_opened": False,
    }


def _existing(path: Path, identity: dict) -> Path | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = {key: payload.get(key) for key in identity}
    if observed != identity:
        raise RuntimeError(f"stale result identity in {path}")
    print(f"cached {path}", flush=True)
    return path


def _decode_task1(spec: dict, job_index: int) -> tuple[str, str]:
    variants = [str(row["id"]) for row in spec["variants"]]
    count = len(variants) * len(spec["datasets"])
    if not 0 <= int(job_index) < count:
        raise IndexError(f"Task1 job index {job_index} outside [0,{count})")
    variant_index, dataset_index = divmod(int(job_index), len(spec["datasets"]))
    return variants[variant_index], spec["datasets"][dataset_index]


def _decode_seeded(spec: dict, task: str, job_index: int) -> tuple[str, str, int]:
    seeds = [int(value) for value in spec[task]["development_training_seeds"]]
    variants = [str(row["id"]) for row in spec["variants"]]
    per_variant = len(spec["datasets"]) * len(seeds)
    count = len(variants) * per_variant
    if not 0 <= int(job_index) < count:
        raise IndexError(f"{task} job index {job_index} outside [0,{count})")
    variant_index, remainder = divmod(int(job_index), per_variant)
    dataset_index, seed_index = divmod(remainder, len(seeds))
    return variants[variant_index], spec["datasets"][dataset_index], seeds[seed_index]


def run_task1(config_path: str | Path, job_index: int) -> Path:
    spec, path = _load_spec(config_path)
    variant_id, dataset = _decode_task1(spec, job_index)
    identity = _identity(spec, path, "Task1", variant_id, dataset, None)
    target = Path(spec["output_root"]) / "development" / "task1" / variant_id / f"{dataset}.json"
    if _existing(target, identity):
        return target
    train = load_common_records(spec, variant_id, dataset, spec["splits"]["train"])
    validation = load_common_records(
        spec, variant_id, dataset, spec["splits"]["validation"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    settings = spec["task1"]
    for seed in settings["development_kmeans_seeds"]:
        for arm, feature, pca_dim in (
            ("raw", "raw", int(settings["raw_pca_dim"])),
            ("fmt", str(settings["fmt_feature"]), int(settings["fmt_pca_dim"])),
        ):
            score = _fit_score(
                train, validation, feature, pca_dim, int(seed),
                int(settings["kmeans_n_init"]), device,
            )
            rows.append({
                "arm": arm, "feature": feature, "pca_dim": pca_dim,
                "kmeans_seed": int(seed), **score,
            })
    result = {
        **identity,
        "pathline": _variant(spec, variant_id),
        "train_samples": int(sum(len(row["reference"]) for row in train)),
        "validation_samples": int(sum(len(row["reference"]) for row in validation)),
        "rows": rows,
    }
    print(f"DONE Task1 {variant_id}/{dataset}", flush=True)
    return _atomic_json(target, result)


def _task2_source(spec: dict) -> EasyConfig:
    source_path = Path(spec["task2"]["source_config"])
    return EasyConfig(str(source_path))


def run_task2(config_path: str | Path, job_index: int) -> Path:
    spec, path = _load_spec(config_path)
    variant_id, dataset, seed = _decode_seeded(spec, "task2", job_index)
    identity = _identity(spec, path, "Task2", variant_id, dataset, seed)
    target = (
        Path(spec["output_root"]) / "development" / "task2" / variant_id
        / dataset / f"seed{seed}.json"
    )
    if _existing(target, identity):
        return target
    train = load_common_records(spec, variant_id, dataset, spec["splits"]["train"])
    validation = load_common_records(
        spec, variant_id, dataset, spec["splits"]["validation"]
    )
    reference = stack_reference(validation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = dict(spec["task2"]["architecture"])
    source = _task2_source(spec)
    rows = []
    for arm in ("raw", "fmt"):
        train_x, validation_x = _prepare_inputs(
            train, validation, arm, str(spec["task2"]["fmt_feature"]), device
        )
        train_mu, validation_mu, losses = train_vae(
            train_x, validation_x, settings, source, seed, device
        )
        score = _latent_score(
            train_mu, validation_mu, reference,
            int(spec["task2"]["kmeans_seed"]),
            int(spec["task2"]["kmeans_n_init"]),
        )
        rows.append({
            "arm": arm,
            "method": "Raw+VAE" if arm == "raw" else "FMT+VAE",
            "feature": "center_relative" if arm == "raw" else spec["task2"]["fmt_feature"],
            "architecture": settings["id"],
            **score, **losses,
        })
        del train_x, validation_x, train_mu, validation_mu
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        **identity,
        "pathline": _variant(spec, variant_id),
        "train_samples": int(sum(len(row["reference"]) for row in train)),
        "validation_samples": int(len(reference)),
        "rows": rows,
    }
    print(f"DONE Task2 {variant_id}/{dataset}/seed{seed}", flush=True)
    return _atomic_json(target, result)


def _task3_split(records: list[dict], feature: str, device) -> tuple:
    raw = np.concatenate([
        row["raw"].reshape(
            -1, 7, row["raw"].shape[1] // (7 * 3), 3
        ) for row in records
    ], axis=0)
    fmt = np.concatenate([
        feature_matrix(row, feature, device) for row in records
    ], axis=0)
    labels = np.concatenate([row["reference"] for row in records]).astype(np.float32)
    return raw, fmt, labels


def _task3_backbone_spec(spec: dict, output: Path, seed: int) -> dict:
    task = spec["task3"]
    return {
        "experiment": f"{spec['experiment']}_Task3_RawBackbone",
        "model": dict(task["backbone_model"]),
        "training": {**dict(task["training"]), "seeds": [int(seed)]},
        "output_dir": str(output),
    }


def _task3_residual_spec(spec: dict, output: Path, checkpoint_dir: Path,
                         seed: int, auxiliary_source: str, fmt_dim: int) -> dict:
    task = spec["task3"]
    return {
        "experiment": f"{spec['experiment']}_Task3_{auxiliary_source}",
        "raw_checkpoint_dir": str(checkpoint_dir),
        "raw_wide_parameter_count": int(task["raw_wide_parameter_count"]),
        "raw_pca_components": int(fmt_dim),
        "raw_pca_random_state": int(task["raw_pca_random_state"]),
        "auxiliary_source": auxiliary_source,
        "model": dict(task["residual_model"]),
        "fusion": dict(task["fusion"]),
        "training": {**dict(task["training"]), "seeds": [int(seed)]},
        "output_dir": str(output),
    }


def run_task3(config_path: str | Path, job_index: int) -> Path:
    spec, path = _load_spec(config_path)
    variant_id, dataset, seed = _decode_seeded(spec, "task3", job_index)
    identity = _identity(spec, path, "Task3", variant_id, dataset, seed)
    job_root = (
        Path(spec["output_root"]) / "development" / "task3" / variant_id
        / dataset / f"seed{seed}"
    )
    target = job_root / "result.json"
    if _existing(target, identity):
        return target
    train_records = load_common_records(
        spec, variant_id, dataset, spec["splits"]["train"]
    )
    validation_records = load_common_records(
        spec, variant_id, dataset, spec["splits"]["validation"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = str(spec["task3"]["fmt_feature"])
    train = _task3_split(train_records, feature, device)
    validation = _task3_split(validation_records, feature, device)
    train, validation, _, stats = _normalize_train_only(train, validation)
    fmt_dim = int(train[1].shape[1])

    backbone_root = job_root / "backbone"
    backbone_spec = _task3_backbone_spec(spec, backbone_root, seed)
    raw_row = train_raw_backbone(
        backbone_spec, dataset, "raw", seed,
        (train, validation, None), stats, device, backbone_root,
    )
    raw_wide_row = train_raw_backbone(
        backbone_spec, dataset, "raw_wide", seed,
        (train, validation, None), stats, device, backbone_root,
    )
    if int(raw_wide_row["parameter_count"]) != int(
        spec["task3"]["raw_wide_parameter_count"]
    ):
        raise RuntimeError(
            "Task3 Raw-wide parameter count changed: "
            f"{raw_wide_row['parameter_count']} != "
            f"{spec['task3']['raw_wide_parameter_count']}"
        )
    checkpoint_dir = backbone_root / "checkpoints"
    rows = [
        {"arm": "raw_backbone", **raw_row},
        {"arm": "raw_wide_guard", **raw_wide_row},
    ]
    for source in ("raw_pca", "fmt"):
        output = job_root / source
        residual_spec = _task3_residual_spec(
            spec, output, checkpoint_dir, seed, source, fmt_dim
        )
        row = train_residual(
            residual_spec, dataset, seed, (train, validation, None),
            stats, device, output,
        )
        rows.append({"arm": source, **row})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    residual_counts = {
        int(row["trainable_residual_parameter_count"])
        for row in rows if row["arm"] in {"raw_pca", "fmt"}
    }
    if len(residual_counts) != 1:
        raise RuntimeError(
            f"Task3 paired residual capacities differ: {residual_counts}"
        )
    result = {
        **identity,
        "pathline": _variant(spec, variant_id),
        "fmt_feature": feature,
        "fmt_dim": fmt_dim,
        "train_samples": int(len(train[2])),
        "validation_samples": int(len(validation[2])),
        "rows": rows,
    }
    print(f"DONE Task3 {variant_id}/{dataset}/seed{seed}", flush=True)
    return _atomic_json(target, result)


def _collect(spec: dict, task: str) -> list[dict]:
    root = Path(spec["output_root"]) / "development" / task.lower()
    paths = sorted(root.glob("**/*.json"))
    expected = (
        len(spec["variants"]) * len(spec["datasets"])
        if task == "Task1" else
        len(spec["variants"]) * len(spec["datasets"])
        * len(spec[task.lower()]["development_training_seeds"])
    )
    if len(paths) != expected:
        raise RuntimeError(
            f"{task} expected {expected} result files, found {len(paths)}"
        )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("task") != task for row in payloads):
        raise RuntimeError(f"{task} result directory contains another task")
    return payloads


def _paired_summary(payloads: list[dict], task: str, variants: list[str],
                    datasets: list[str]) -> list[dict]:
    result = []
    arm_names = {
        "Task1": ("raw", "fmt"),
        "Task2": ("raw", "fmt"),
        "Task3": ("raw_pca", "fmt"),
    }[task]
    for variant in variants:
        dataset_rows = []
        for dataset in datasets:
            matches = [row for row in payloads
                       if row["variant"] == variant and row["dataset"] == dataset]
            arm_values = {arm: [] for arm in arm_names}
            arm_ap = {arm: [] for arm in arm_names}
            for payload in matches:
                for row in payload["rows"]:
                    arm = row["arm"]
                    if arm not in arm_values:
                        continue
                    arm_values[arm].append(float(row["validation_f1"] if task == "Task3" else row["f1"]))
                    if task == "Task3":
                        arm_ap[arm].append(float(row["validation_average_precision"]))
            if any(not arm_values[arm] for arm in arm_names):
                raise RuntimeError(f"incomplete {task} pair for {variant}/{dataset}")
            raw_f1 = float(np.mean(arm_values[arm_names[0]]))
            fmt_f1 = float(np.mean(arm_values[arm_names[1]]))
            item = {
                "task": task, "variant": variant, "dataset": dataset,
                "raw_f1": raw_f1, "fmt_f1": fmt_f1,
                "f1_gain": fmt_f1 - raw_f1,
            }
            if task == "Task3":
                raw_ap = float(np.mean(arm_ap[arm_names[0]]))
                fmt_ap = float(np.mean(arm_ap[arm_names[1]]))
                item.update({
                    "raw_ap": raw_ap, "fmt_ap": fmt_ap,
                    "ap_gain": fmt_ap - raw_ap,
                })
            dataset_rows.append(item)
            result.append(item)
        if len(dataset_rows) != len(datasets):
            raise RuntimeError(f"incomplete dataset coverage for {task}/{variant}")
    return result


def select(config_path: str | Path) -> Path:
    spec, path = _load_spec(config_path)
    output = Path(spec["output_root"]) / "selection"
    output.mkdir(parents=True, exist_ok=True)
    variants = [str(row["id"]) for row in spec["variants"]]
    per_dataset = []
    for task in ("Task1", "Task2", "Task3"):
        per_dataset.extend(_paired_summary(
            _collect(spec, task), task, variants, list(spec["datasets"])
        ))
    leaderboard = []
    for variant in variants:
        task_rows = {}
        for task in ("Task1", "Task2", "Task3"):
            values = [row for row in per_dataset
                      if row["variant"] == variant and row["task"] == task]
            task_rows[task] = {
                "raw_f1": float(np.mean([row["raw_f1"] for row in values])),
                "fmt_f1": float(np.mean([row["fmt_f1"] for row in values])),
                "f1_gain": float(np.mean([row["f1_gain"] for row in values])),
            }
            if task == "Task3":
                task_rows[task]["ap_gain"] = float(np.mean([
                    row["ap_gain"] for row in values
                ]))
        gains = [task_rows[task]["f1_gain"] for task in ("Task1", "Task2", "Task3")]
        fmt_f1 = [task_rows[task]["fmt_f1"] for task in ("Task1", "Task2", "Task3")]
        leaderboard.append({
            "variant": variant,
            "equal_task_mean_f1_gain": float(np.mean(gains)),
            "minimum_task_f1_gain": float(np.min(gains)),
            "task3_ap_gain": task_rows["Task3"]["ap_gain"],
            "mean_absolute_fmt_f1": float(np.mean(fmt_f1)),
            "tasks": task_rows,
        })
    leaderboard.sort(key=lambda row: (
        row["equal_task_mean_f1_gain"], row["minimum_task_f1_gain"],
        row["task3_ap_gain"], row["mean_absolute_fmt_f1"], row["variant"],
    ), reverse=True)
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank
    selection = {
        "experiment": spec["experiment"],
        "config_sha256": _sha256(path),
        "status": "development_winner_frozen_before_confirmation",
        "opened_ordinals": sorted(spec["selection"]["opened_ordinals"]),
        "confirmation_opened": False,
        "selection_rule": spec["selection"],
        "winner": leaderboard[0],
        "baseline": next(
            row for row in leaderboard
            if row["variant"] == spec["selection"]["baseline_variant"]
        ),
        "leaderboard": leaderboard,
        "per_dataset": per_dataset,
    }
    _atomic_json(output / "selection.json", selection)
    print(
        f"WINNER {leaderboard[0]['variant']}: equal-task mean F1 gain="
        f"{leaderboard[0]['equal_task_mean_f1_gain']:+.6f}", flush=True,
    )
    return output / "selection.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Verify_Task123_FixedPathline_1.1.yaml"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=("task1", "task2", "task3", "select"),
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "select":
        select(args.config)
        return
    if args.job_index is None:
        parser.error(f"--job-index is required for --mode {args.mode}")
    {"task1": run_task1, "task2": run_task2, "task3": run_task3}[args.mode](
        args.config, args.job_index
    )


if __name__ == "__main__":
    main()
