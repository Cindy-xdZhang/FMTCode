"""Evaluate clean-trained 3D Task1--Task3 models under pathline corruption."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.PathlineClassifier_3D import (
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from FMT_Utils.RawPathline_3D import raw_pathline_representation
from FMT_Utils.RobustnessFeatures_3D import (
    corrupt_pathline_primitives_3d,
    reshape_cached_primitives,
)
from FMT_Utils.Task12Data_3D import (
    feature_matrix,
    load_cache_records,
    stack_features,
    stack_reference,
)
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
    fit_kmeans_transform,
)
from experiments.Verify_HighReVAE import _train
from experiments.Verify_Task3_FMTClassifier import (
    _classification_metrics,
    _loader,
)
from experiments.Verify_Task3_FMTResidual import (
    _apply_raw_pca_transform,
    _load_raw_model,
    _predict_components,
    _probabilities,
)
from experiments import Search_Task3_FMTResidual_3D as task3_search


def _read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "corruptions", "randomization",
        "task1", "task2", "task3",
    }
    if required - set(spec):
        raise ValueError(f"noise config misses {sorted(required - set(spec))}")
    ids = [row["id"] for row in spec["corruptions"]]
    if len(ids) != len(set(ids)) or ids[0] != "clean":
        raise ValueError("corruptions must be unique and begin with clean")
    return spec


def _stable_seed(base: int, *parts) -> int:
    payload = "|".join([str(int(base)), *(str(value) for value in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def _primitive_scale(raw: np.ndarray) -> float:
    primitives = reshape_cached_primitives(raw)
    offsets = np.linalg.norm(
        primitives[:, 1:, 0] - primitives[:, :1, 0], axis=-1
    )
    positive = offsets[np.isfinite(offsets) & (offsets > 0)]
    if not len(positive):
        raise ValueError("cannot infer a positive primitive offset")
    return float(np.median(positive))


def _recompute_record(
    record: dict,
    condition: dict,
    *,
    repeat_seed: int,
    base_seed: int,
    device,
    need_cached_fmt: bool,
) -> dict:
    primitives = corrupt_pathline_primitives_3d(
        record["raw"], condition["kind"], float(condition["level"]),
        spatial_scale=_primitive_scale(record["raw"]),
        random_state=_stable_seed(
            base_seed, record["metadata"]["dataset"], record["ordinal"],
            condition["id"], repeat_seed,
        ),
    )
    if need_cached_fmt:
        fmt = pathline_dft_features_3d(
            torch.from_numpy(primitives).to(device), num_freq=6,
            neighbor_weight=1.0, neighbor_scale=1.0,
            neighbor_pool="sort", mode="gram", include_chirality=True,
        ).astype(np.float32)
        if condition["id"] == "clean":
            difference = np.asarray(fmt - record["fmt"], dtype=np.float64)
            max_absolute = float(np.max(np.abs(difference)))
            relative_l2 = float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(record["fmt"].astype(np.float64)), 1e-12)
            )
            # Original caches were generated on CUDA whereas this replay may
            # run on a different GPU or CPU.  A norm check admits harmless
            # FFT reduction-order noise while still rejecting a changed FMT
            # recipe or primitive layout by several orders of magnitude.
            if max_absolute > 1e-3 or relative_l2 > 1e-4:
                raise RuntimeError(
                    "clean FMT recomputation differs from cache: "
                    f"max_absolute={max_absolute}, relative_l2={relative_l2}"
                )
    else:
        fmt = np.zeros((len(primitives), 161), dtype=np.float32)
    return {
        "path": record["path"], "ordinal": record["ordinal"],
        "raw": primitives.reshape(len(primitives), -1), "fmt": fmt,
        "reference": record["reference"], "metadata": record["metadata"],
        "features": {},
    }


def _corrupted_records(
    records, condition, repeat_seed, base_seed, device, need_cached_fmt=True
):
    return [
        _recompute_record(
            record, condition, repeat_seed=repeat_seed, base_seed=base_seed,
            device=device, need_cached_fmt=need_cached_fmt,
        )
        for record in records
    ]


def _task1_source_for_dataset(source: dict, dataset: str) -> dict:
    matches = [row for row in source["sources"].values() if dataset in row["datasets"]]
    if len(matches) != 1:
        raise ValueError(f"Task1 dataset {dataset} matched {len(matches)} sources")
    return matches[0]


def _by_ordinal(records):
    result = {int(row["ordinal"]): row for row in records}
    if len(result) != len(records):
        raise RuntimeError("duplicate cache ordinal")
    return result


def run_task1(config_path: str | Path, job_index: int) -> Path:
    spec = _load_spec(config_path)
    task = spec["task1"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    selected = json.loads(Path(task["selected_config"]).read_text(encoding="utf-8"))
    expected_choices = {
        "raw": (selected["raw"]["feature"], str(selected["raw"]["pca_dim"])),
        "fmt": (selected["fmt"]["feature"], str(selected["fmt"]["pca_dim"])),
    }
    configured_choices = {
        row["id"]: (row["feature"], str(row["pca_dim"])) for row in task["arms"]
    }
    if expected_choices != configured_choices:
        raise RuntimeError(
            f"Task1 noise recipes differ from frozen main: {configured_choices} "
            f"!= {expected_choices}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    required = sorted(set(task["final_train"]) | set(task["cluster_calibration"]))
    base_seed = int(spec["randomization"]["corruption_seed"])
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets):
        raise IndexError("Task1 noise array index out of range")
    dataset = datasets[index]
    item = _task1_source_for_dataset(source, dataset)
    development = _by_ordinal(load_cache_records(
        Path(item["development_cache"]) / dataset,
        expected_count=int(source["development_count"]), ordinals=required,
    ))
    confirmation = load_cache_records(
        Path(item["confirmation_cache"]) / dataset,
        expected_count=int(source["confirmation_count"]),
    )
    train = [development[int(value)] for value in task["final_train"]]
    calibration = [development[int(value)] for value in task["cluster_calibration"]]
    calibration_reference = stack_reference(calibration)
    confirmation_reference = stack_reference(confirmation)
    clean_inputs = {
        arm["id"]: (
            stack_features(train, arm["feature"], device),
            stack_features(calibration, arm["feature"], device),
        )
        for arm in task["arms"]
    }
    for seed_value in task["kmeans_seeds"]:
        seed = int(seed_value)
        noisy_by_condition = {
            condition["id"]: _corrupted_records(
                confirmation, condition, seed, base_seed, device,
                need_cached_fmt=True,
            )
            for condition in spec["corruptions"]
        }
        for arm in task["arms"]:
            train_x, calibration_x = clean_inputs[arm["id"]]
            fitted = fit_kmeans_transform(
                train_x, int(arm["pca_dim"]), seed, int(task["kmeans_n_init"])
            )
            cluster = calibrate_vortex_cluster(
                calibration_reference, fitted.predict(calibration_x)
            )
            for condition in spec["corruptions"]:
                evaluate_x = stack_features(
                    noisy_by_condition[condition["id"]], arm["feature"], device
                )
                score = binary_cluster_metrics(
                    confirmation_reference, fitted.predict(evaluate_x), cluster
                )
                rows.append({
                    "experiment": spec["experiment"], "task": "Task1",
                    "dataset": dataset, "arm": arm["id"], "seed": seed,
                    "condition": condition["id"], "kind": condition["kind"],
                    "level": float(condition["level"]),
                    "cluster_as_vortex": int(cluster), **score,
                })
    target = Path(spec["output_root"]) / "task1" / "shards" / f"{dataset}.csv"
    _write_csv(target, rows)
    return target


def merge_task1(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    source = yaml.safe_load(Path(
        spec["task1"]["source_config"]
    ).read_text(encoding="utf-8"))
    rows = []
    for dataset in source["datasets"]:
        rows.extend(_read_csv(
            Path(spec["output_root"]) / "task1" / "shards" / f"{dataset}.csv"
        ))
    expected = (
        len(source["datasets"]) * len(spec["task1"]["arms"])
        * len(spec["task1"]["kmeans_seeds"]) * len(spec["corruptions"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"incomplete Task1 noise output: {len(rows)} != {expected}")
    target = Path(spec["output_root"]) / "task1" / "per_run.csv"
    _write_csv(target, rows)
    return target


def _task2_group(source: dict, dataset: str) -> dict:
    matches = [row for row in source["groups"].values() if dataset in row["datasets"]]
    if len(matches) != 1:
        raise ValueError(f"Task2 dataset {dataset} matched {len(matches)} groups")
    return matches[0]


def _rms_rows(values):
    scale = np.sqrt(np.square(values).mean(axis=1, keepdims=True)).clip(1e-8)
    return values / scale


def _fit_task2_transform(records, arm: str, device):
    if arm == "raw":
        values = np.concatenate([
            raw_pathline_representation(row["raw"], "center_relative") for row in records
        ])
        sampled_steps = records[0]["raw"].shape[1] // (7 * 3)
        split = sampled_steps * 3
        values = np.concatenate(
            (_rms_rows(values[:, :split]), _rms_rows(values[:, split:])), axis=1
        )
    elif arm == "fmt":
        values = stack_features(records, "fmt_all+kin4", device)
    else:
        raise ValueError(f"unknown Task2 arm {arm}")
    scaler = StandardScaler().fit(values)

    def transform(target_records):
        if arm == "raw":
            target = np.concatenate([
                raw_pathline_representation(row["raw"], "center_relative")
                for row in target_records
            ])
            target = np.concatenate(
                (_rms_rows(target[:, :split]), _rms_rows(target[:, split:])), axis=1
            )
        else:
            target = stack_features(target_records, "fmt_all+kin4", device)
        return scaler.transform(target).astype(np.float32)

    return scaler.transform(values).astype(np.float32), transform


@torch.no_grad()
def _vae_encode(model, values: np.ndarray, device, batch_size=8192) -> np.ndarray:
    model.eval()
    result = []
    for begin in range(0, len(values), int(batch_size)):
        mu, _ = model.encode(torch.from_numpy(values[begin:begin + batch_size]).to(device))
        result.append(mu.cpu().numpy())
    return np.concatenate(result)


def run_task2(config_path: str | Path, job_index: int) -> Path:
    spec = _load_spec(config_path)
    task = spec["task2"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    selection = json.loads(Path(task["selected_config"]).read_text(encoding="utf-8"))
    winner = selection["winner"]
    if winner["fmt_feature"] != "fmt_all+kin4" or winner["architecture"] != "l512_l64_b1e-6":
        raise RuntimeError("Task2 frozen main recipe changed")
    datasets, seeds = list(source["datasets"]), [int(v) for v in task["training_seeds"]]
    index = int(job_index)
    if not 0 <= index < len(datasets) * len(seeds):
        raise IndexError("Task2 noise array index out of range")
    dataset_index, seed_index = divmod(index, len(seeds))
    dataset, seed = datasets[dataset_index], seeds[seed_index]
    group = _task2_group(source, dataset)
    required = sorted({
        int(value) for key in ("train", "cluster_calibration", "confirmation")
        for value in task[key]
    })
    records = _by_ordinal(load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(source["expected_slices"]), ordinals=required,
    ))
    train = [records[int(value)] for value in task["train"]]
    calibration = [records[int(value)] for value in task["cluster_calibration"]]
    confirmation = [records[int(value)] for value in task["confirmation"]]
    calibration_reference = stack_reference(calibration)
    confirmation_reference = stack_reference(confirmation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = next(
        dict(row) for row in source["architectures"]
        if row["id"] == winner["architecture"]
    )
    source_config = EasyConfig(group["source_config"])
    base_seed = int(spec["randomization"]["corruption_seed"])
    noisy = {
        condition["id"]: _corrupted_records(
            confirmation, condition, seed, base_seed, device, need_cached_fmt=True
        )
        for condition in spec["corruptions"]
    }
    target = Path(spec["output_root"]) / "task2" / "shards" / f"{dataset}_seed{seed}.csv"
    rows = _read_csv(target)
    completed = {(row["arm"], row["condition"]) for row in rows}
    for arm in task["arms"]:
        train_x, transform = _fit_task2_transform(train, arm, device)
        calibration_x = transform(calibration)
        train_mu, calibration_mu, losses, model = _train(
            train_x, calibration_x, architecture, source_config, seed, device,
            return_model=True,
        )
        kmeans = KMeans(
            n_clusters=2, random_state=int(task["kmeans_seed"]),
            n_init=int(task["kmeans_n_init"]),
        ).fit(train_mu)
        cluster = calibrate_vortex_cluster(
            calibration_reference, kmeans.predict(calibration_mu)
        )
        for condition in spec["corruptions"]:
            key = (arm, condition["id"])
            if key in completed:
                continue
            noisy_x = transform(noisy[condition["id"]])
            noisy_mu = _vae_encode(model, noisy_x, device)
            score = binary_cluster_metrics(
                confirmation_reference, kmeans.predict(noisy_mu), cluster
            )
            rows.append({
                "experiment": spec["experiment"], "task": "Task2",
                "dataset": dataset, "arm": arm, "training_seed": seed,
                "condition": condition["id"], "kind": condition["kind"],
                "level": float(condition["level"]),
                "cluster_as_vortex": int(cluster), **score, **losses,
            })
            _write_csv(target, rows)
            completed.add(key)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return target


def _load_residual_model(checkpoint_path: Path, fmt_dim: int, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_model, raw_checkpoint = _load_raw_model(
        checkpoint["raw_checkpoint"], fmt_dim, device
    )
    for key in ("raw_mean", "raw_std"):
        if not np.array_equal(
            np.asarray(checkpoint["normalization"][key]),
            np.asarray(raw_checkpoint["normalization"][key]),
        ):
            raise RuntimeError(f"{checkpoint_path}: Raw normalization mismatch")
    model = PathlineFMTResidualClassifier3D(
        raw_model, fmt_dim=fmt_dim,
        **residual_model_kwargs(checkpoint["config"]["model"]),
    ).to(device)
    state = model.state_dict()
    state.update(checkpoint["residual_state_dict"])
    model.load_state_dict(state)
    return model.eval(), checkpoint


def _task3_noisy_split(records, candidate, condition, seed, base_seed, device):
    changed = _corrupted_records(
        records, condition, seed, base_seed, device, need_cached_fmt=False
    )
    chunks = []
    for record in changed:
        fmt = feature_matrix(record, candidate["fmt_feature"], device)
        raw = reshape_cached_primitives(record["raw"])
        chunks.append((raw, fmt, record["reference"].astype(np.float32)))
    return tuple(np.concatenate([row[index] for row in chunks]) for index in range(3))


def run_task3(config_path: str | Path, job_index: int) -> Path:
    spec = _load_spec(config_path)
    task = spec["task3"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    selection = json.loads(Path(task["selected_config"]).read_text(encoding="utf-8"))
    winner = selection["winner"]
    if winner["candidate_id"] != "g08_aivd1w3_dft":
        raise RuntimeError("Task3 frozen main recipe changed")
    candidate = next(
        dict(row) for row in source["candidates"] if row["id"] == winner["candidate_id"]
    )
    datasets, seeds = list(source["datasets"]), [int(v) for v in task["training_seeds"]]
    index = int(job_index)
    if not 0 <= index < len(datasets) * len(seeds):
        raise IndexError("Task3 noise array index out of range")
    dataset_index, seed_index = divmod(index, len(seeds))
    dataset, seed = datasets[dataset_index], seeds[seed_index]
    _, group = task3_search._group_for_dataset(source, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    required = sorted(set(task["train"]) | set(task["validation"]) | set(task["confirmation"]))
    records = task3_search._load_records(source, dataset, candidate, device, ordinals=required)
    train = task3_search._stack_split(records, task["train"])
    validation = task3_search._stack_split(records, task["validation"])
    confirmation = task3_search._stack_split(records, task["confirmation"])
    raw_stats = task3_search._frozen_raw_normalization(
        group, dataset, int(task["training_seeds"][0])
    )
    train, validation, confirmation_clean, stats = task3_search._normalize_train_only(
        train, validation, confirmation, raw_stats=raw_stats
    )
    fmt_dim = int(train[1].shape[1])
    confirmation_records = [row for row in records if row[3] in set(task["confirmation"])]
    # Convert Task3 tuples back to the minimal cache-record contract used by
    # the common corruption helper.  Cached FMT is unnecessary for aivd1w3.
    base_records = []
    for raw, _, labels, ordinal, metadata in confirmation_records:
        base_records.append({
            "path": Path(metadata["source_cache"]), "ordinal": int(ordinal),
            "raw": raw.reshape(len(raw), -1),
            "fmt": np.zeros((len(raw), 161), dtype=np.float32),
            "reference": labels.astype(bool), "metadata": {
                **metadata, "dataset": dataset,
            }, "features": {},
        })
    base_seed = int(spec["randomization"]["corruption_seed"])
    noisy_raw = {
        condition["id"]: _task3_noisy_split(
            base_records, candidate, condition, seed, base_seed, device
        )
        for condition in spec["corruptions"]
    }
    target = Path(spec["output_root"]) / "task3" / "shards" / f"{dataset}_seed{seed}.csv"
    rows = _read_csv(target)
    completed = {(row["arm"], row["condition"]) for row in rows}
    for arm in task["arms"]:
        output_dir = (
            Path(spec["output_root"]) / "task3" / "temporary_training"
            / dataset / f"seed{seed}" / arm
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        run_spec = task3_search._candidate_spec(
            source, group, candidate, dataset, seed, arm, output_dir, fmt_dim
        )
        run_spec["experiment"] = spec["experiment"]
        run_spec["split"] = {
            "train_ordinals": list(task["train"]),
            "validation_ordinals": list(task["validation"]),
            "test_ordinals": list(task["confirmation"]),
        }
        run_spec["evaluation"] = {"test_enabled": True}
        clean_result = task3_search._train_one(
            run_spec, dataset, seed, (train, validation, confirmation_clean),
            stats, device, output_dir,
        )
        checkpoint_path = Path(clean_result["checkpoint"])
        model, checkpoint = _load_residual_model(checkpoint_path, fmt_dim, device)
        for condition in spec["corruptions"]:
            key = (arm, condition["id"])
            if key in completed:
                continue
            raw, fmt, labels = noisy_raw[condition["id"]]
            raw = (
                (raw - np.asarray(checkpoint["normalization"]["raw_mean"]))
                / np.asarray(checkpoint["normalization"]["raw_std"])
            ).astype(np.float32)
            if arm == "fmt":
                auxiliary = (
                    (fmt - np.asarray(checkpoint["normalization"]["fmt_mean"]))
                    / np.asarray(checkpoint["normalization"]["fmt_std"])
                ).astype(np.float32)
            else:
                auxiliary = _apply_raw_pca_transform(
                    raw, checkpoint["auxiliary_transform"]
                )
            split = (raw, auxiliary, labels.astype(np.float32))
            loader = _loader(split, 1024, False, seed, device.type == "cuda")
            targets, raw_logits, residual_logits = _predict_components(model, loader, device)
            probabilities = _probabilities(
                raw_logits, residual_logits, float(checkpoint["alpha"]),
                checkpoint["config"]["model"],
            )
            score = _classification_metrics(
                targets, probabilities, float(checkpoint["threshold"])
            )
            if condition["id"] == "clean":
                if abs(score["f1"] - float(clean_result["test_f1"])) > 1e-7:
                    raise RuntimeError("Task3 clean replay does not match training result")
            rows.append({
                "experiment": spec["experiment"], "task": "Task3",
                "dataset": dataset, "arm": arm, "training_seed": seed,
                "condition": condition["id"], "kind": condition["kind"],
                "level": float(condition["level"]),
                "clean_best_epoch": int(checkpoint["best_epoch"]),
                "clean_alpha": float(checkpoint["alpha"]),
                "clean_threshold": float(checkpoint["threshold"]), **score,
            })
            _write_csv(target, rows)
            completed.add(key)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return target


def _collect_task_rows(spec: dict, task_name: str) -> list[dict]:
    output = Path(spec["output_root"]) / task_name.lower()
    if task_name == "Task1":
        return _read_csv(output / "per_run.csv")
    source = yaml.safe_load(
        Path(spec[task_name.lower()]["source_config"]).read_text(encoding="utf-8")
    )
    rows = []
    for dataset in source["datasets"]:
        for seed in spec[task_name.lower()]["training_seeds"]:
            rows.extend(_read_csv(output / "shards" / f"{dataset}_seed{int(seed)}.csv"))
    return rows


def summarize(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    source1 = yaml.safe_load(Path(spec["task1"]["source_config"]).read_text(encoding="utf-8"))
    expected = {
        "Task1": len(source1["datasets"]) * len(spec["task1"]["arms"])
        * len(spec["task1"]["kmeans_seeds"]) * len(spec["corruptions"]),
        "Task2": len(source1["datasets"]) * len(spec["task2"]["arms"])
        * len(spec["task2"]["training_seeds"]) * len(spec["corruptions"]),
        "Task3": len(source1["datasets"]) * len(spec["task3"]["arms"])
        * len(spec["task3"]["training_seeds"]) * len(spec["corruptions"]),
    }
    all_rows, observed = {}, {}
    for task_name in ("Task1", "Task2", "Task3"):
        rows = _collect_task_rows(spec, task_name)
        all_rows[task_name] = rows
        observed[task_name] = len(rows)
    if observed != expected:
        raise RuntimeError(f"incomplete noise outputs: {observed} != {expected}")
    table = []
    for task_name, rows in all_rows.items():
        for condition in spec["corruptions"]:
            for arm in spec[task_name.lower()]["arms"]:
                arm_name = arm if isinstance(arm, str) else arm["id"]
                selected = [
                    row for row in rows
                    if row["condition"] == condition["id"] and row["arm"] == arm_name
                ]
                dataset_values = []
                dataset_ap = []
                for dataset in sorted({row["dataset"] for row in selected}):
                    subset = [row for row in selected if row["dataset"] == dataset]
                    dataset_values.append(np.mean([float(row["f1"]) for row in subset]))
                    if task_name == "Task3":
                        dataset_ap.append(np.mean([
                            float(row["average_precision"]) for row in subset
                        ]))
                row = {
                    "task": task_name, "condition": condition["id"],
                    "kind": condition["kind"], "level": float(condition["level"]),
                    "arm": arm_name, "dataset_macro_f1": float(np.mean(dataset_values)),
                    "dataset_count": len(dataset_values), "run_count": len(selected),
                }
                if dataset_ap:
                    row["dataset_macro_average_precision"] = float(np.mean(dataset_ap))
                table.append(row)
    clean_lookup = {
        (row["task"], row["arm"]): row for row in table if row["condition"] == "clean"
    }
    for row in table:
        clean = clean_lookup[(row["task"], row["arm"])]
        row["f1_change_from_clean"] = row["dataset_macro_f1"] - clean["dataset_macro_f1"]
        row["f1_retention"] = (
            row["dataset_macro_f1"] / clean["dataset_macro_f1"]
            if clean["dataset_macro_f1"] > 0 else float("nan")
        )
        if "dataset_macro_average_precision" in row:
            row["average_precision_change_from_clean"] = (
                row["dataset_macro_average_precision"]
                - clean["dataset_macro_average_precision"]
            )
    gains = []
    for task_name in ("Task1", "Task2", "Task3"):
        raw_arm = "raw" if task_name in {"Task1", "Task2"} else "raw_pca"
        for condition in spec["corruptions"]:
            raw = next(row for row in table if row["task"] == task_name and row["condition"] == condition["id"] and row["arm"] == raw_arm)
            fmt = next(row for row in table if row["task"] == task_name and row["condition"] == condition["id"] and row["arm"] == "fmt")
            item = {
                "task": task_name, "condition": condition["id"],
                "kind": condition["kind"], "level": float(condition["level"]),
                "raw_f1": raw["dataset_macro_f1"], "fmt_f1": fmt["dataset_macro_f1"],
                "fmt_minus_raw_f1": fmt["dataset_macro_f1"] - raw["dataset_macro_f1"],
            }
            if task_name == "Task3":
                item.update({
                    "raw_average_precision": raw["dataset_macro_average_precision"],
                    "fmt_average_precision": fmt["dataset_macro_average_precision"],
                    "fmt_minus_raw_average_precision": (
                        fmt["dataset_macro_average_precision"]
                        - raw["dataset_macro_average_precision"]
                    ),
                })
            gains.append(item)
    output = Path(spec["output_root"])
    _write_csv(output / "robustness_table.csv", table)
    _write_csv(output / "paired_gain_table.csv", gains)
    summary = {
        "schema": 1, "experiment": spec["experiment"],
        "record_counts": observed, "table": table, "paired_gains": gains,
        "interpretation": (
            "All trainable and clustering models were fitted on clean data; "
            "only confirmation pathlines were corrupted."
        ),
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gains, indent=2))
    return target


def cleanup(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    audit_path = Path(spec["output_root"]) / "independent_audit.json"
    if not audit_path.exists() or json.loads(
        audit_path.read_text(encoding="utf-8")
    ).get("status") != "PASS":
        raise RuntimeError("refusing cleanup before independent audit PASS")
    root = (Path(spec["output_root"]) / "task3" / "temporary_training").resolve()
    expected_parent = (Path(spec["output_root"]) / "task3").resolve()
    root.relative_to(expected_parent)
    count = len(list(root.rglob("*.pt"))) if root.exists() else 0
    if root.exists():
        shutil.rmtree(root)
    target = Path(spec["output_root"]) / "cleanup.json"
    target.write_text(json.dumps({
        "status": "PASS", "deleted_checkpoint_count": count,
        "temporary_training_exists": root.exists(),
        "independent_audit_sha256": hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_Task123_NoiseRobustness_1.1.yaml")
    parser.add_argument(
        "--mode", required=True,
        choices=("task1", "task1-merge", "task2", "task3", "summarize", "cleanup"),
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "task1":
        if args.job_index is None:
            parser.error("task1 requires --job-index")
        run_task1(args.config, args.job_index)
    elif args.mode == "task1-merge":
        merge_task1(args.config)
    elif args.mode == "task2":
        if args.job_index is None:
            parser.error("task2 requires --job-index")
        run_task2(args.config, args.job_index)
    elif args.mode == "task3":
        if args.job_index is None:
            parser.error("task3 requires --job-index")
        run_task3(args.config, args.job_index)
    elif args.mode == "summarize":
        summarize(args.config)
    else:
        cleanup(args.config)


if __name__ == "__main__":
    main()
