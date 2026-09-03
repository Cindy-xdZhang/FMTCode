"""Corruption robustness of the stronger non-FMT baselines (Task1--Task3).

Extends ``Verify_Task123_NoiseRobustness_1.1`` (FMT vs. the original Raw
baselines) with the stronger baselines frozen by
``Verify_Task123_StrongBaselines_1.1/1.2``.  The corruption seed, conditions,
training seeds and split contract are identical to 1.1, so every corrupted
confirmation pathline realization is bitwise the same as the one already
scored for the FMT and Raw arms; those 1.1 per-run rows are merged into the
combined table after their SHA-256 digests are checked against the 1.1
independent audit.  Nothing here retrains or re-selects a recipe: all
representations, PCA widths and architectures are read from frozen manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans

from DeepUtils.utils import EasyConfig
from FMT_Utils.RobustnessFeatures_3D import (
    non_fmt_feature_matrix,
    reshape_cached_primitives,
)
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
    fit_kmeans_transform,
)
from experiments.Run_Task123_NoiseRobustness_1_1 import (
    _by_ordinal,
    _corrupted_records,
    _load_spec as _load_noise_spec,
    _read_csv,
    _task1_source_for_dataset,
    _task2_group,
    _vae_encode,
    _write_csv,
)
from experiments.Run_Task123_StrongBaselines_1_1 import (
    _task2_architecture,
    _task2_inputs,
    _task3_wide_model,
)
from experiments.Verify_HighReVAE import _train
from experiments.Verify_Task3_FMTClassifier import (
    _classification_metrics,
    _loader,
    _predict,
)
from experiments.Verify_Task3_FMTResidual import _load_raw_model
from experiments import Search_Task3_FMTResidual_3D as task3_search


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_spec(path: str | Path) -> dict:
    spec = _load_noise_spec(path)
    if "reference_experiment" not in spec:
        raise ValueError("noise 1.2 config requires reference_experiment")
    return spec


def _frozen_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ Task1
def _check_task1_arm(arm: dict) -> None:
    frozen = _frozen_json(arm["frozen_from"])
    winner = frozen["winners"][arm["family"]]
    if frozen.get("confirmation_opened", True):
        raise RuntimeError(f"{arm['frozen_from']} was frozen after opening confirmation")
    if winner["representation"] != arm["representation"] or str(winner["pca_dim"]) != str(arm["pca_dim"]):
        raise RuntimeError(
            f"Task1 arm {arm['id']} ({arm['representation']}/{arm['pca_dim']}) differs from "
            f"frozen winner {winner['representation']}/{winner['pca_dim']}"
        )


def _task1_matrix(records, representation: str, num_freq: int) -> np.ndarray:
    return np.concatenate([
        non_fmt_feature_matrix(record["raw"], representation, num_freq=num_freq)
        for record in records
    ])


def run_task1(config_path: str | Path, job_index: int) -> Path:
    spec = _load_spec(config_path)
    task = spec["task1"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    for arm in task["arms"]:
        _check_task1_arm(arm)
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets):
        raise IndexError("Task1 noise array index out of range")
    dataset = datasets[index]
    item = _task1_source_for_dataset(source, dataset)
    required = sorted(set(task["final_train"]) | set(task["cluster_calibration"]))
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
    num_freq = int(task["num_freq"])
    base_seed = int(spec["randomization"]["corruption_seed"])
    device = torch.device("cpu")
    clean_inputs = {
        arm["id"]: (
            _task1_matrix(train, arm["representation"], num_freq),
            _task1_matrix(calibration, arm["representation"], num_freq),
        )
        for arm in task["arms"]
    }
    rows = []
    for seed_value in task["kmeans_seeds"]:
        seed = int(seed_value)
        noisy_by_condition = {
            condition["id"]: _corrupted_records(
                confirmation, condition, seed, base_seed, device, need_cached_fmt=False
            )
            for condition in spec["corruptions"]
        }
        for arm in task["arms"]:
            train_x, calibration_x = clean_inputs[arm["id"]]
            pca_dim = None if str(arm["pca_dim"]) == "none" else int(arm["pca_dim"])
            fitted = fit_kmeans_transform(train_x, pca_dim, seed, int(task["kmeans_n_init"]))
            cluster = calibrate_vortex_cluster(
                calibration_reference, fitted.predict(calibration_x)
            )
            for condition in spec["corruptions"]:
                evaluate_x = _task1_matrix(
                    noisy_by_condition[condition["id"]], arm["representation"], num_freq
                )
                score = binary_cluster_metrics(
                    confirmation_reference, fitted.predict(evaluate_x), cluster
                )
                rows.append({
                    "experiment": spec["experiment"], "task": "Task1",
                    "dataset": dataset, "arm": arm["id"], "seed": seed,
                    "representation": arm["representation"],
                    "pca_dim": str(arm["pca_dim"]),
                    "condition": condition["id"], "kind": condition["kind"],
                    "level": float(condition["level"]),
                    "cluster_as_vortex": int(cluster), **score,
                })
    target = Path(spec["output_root"]) / "task1" / "shards" / f"{dataset}.csv"
    _write_csv(target, rows)
    return target


def merge_task1(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    source = yaml.safe_load(Path(spec["task1"]["source_config"]).read_text(encoding="utf-8"))
    rows = []
    for dataset in source["datasets"]:
        rows.extend(_read_csv(Path(spec["output_root"]) / "task1" / "shards" / f"{dataset}.csv"))
    expected = (
        len(source["datasets"]) * len(spec["task1"]["arms"])
        * len(spec["task1"]["kmeans_seeds"]) * len(spec["corruptions"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"incomplete Task1 noise output: {len(rows)} != {expected}")
    target = Path(spec["output_root"]) / "task1" / "per_run.csv"
    _write_csv(target, rows)
    return target


# ------------------------------------------------------------------ Task2
def _check_task2_arms(task: dict) -> None:
    frozen = _frozen_json(task["frozen_from"])
    if frozen.get("confirmation_opened", True) or not frozen.get("main_comparison_unchanged", False):
        raise RuntimeError("Task2 strong-baseline manifest is not a pre-confirmation freeze")
    by_id = {row["id"]: row for row in frozen["arms"]}
    for arm in task["arms"]:
        if arm["id"] not in by_id:
            raise RuntimeError(f"Task2 arm {arm['id']} is not a frozen strong baseline")
        for key in ("representation", "architecture"):
            if by_id[arm["id"]][key] != arm[key]:
                raise RuntimeError(f"Task2 arm {arm['id']} {key} differs from frozen manifest")
        if arm["id"] == "raw_tuned" and frozen["strongest_raw_architecture"] != arm["architecture"]:
            raise RuntimeError("raw_tuned architecture is not the frozen strongest Raw VAE")


def run_task2(config_path: str | Path, job_index: int) -> Path:
    spec = _load_spec(config_path)
    task = spec["task2"]
    _check_task2_arms(task)
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    datasets = list(source["datasets"])
    seeds = [int(value) for value in task["training_seeds"]]
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
    source_config = EasyConfig(group["source_config"])
    base_seed = int(spec["randomization"]["corruption_seed"])
    num_freq = int(task["num_freq"])
    pca_random_state = int(task.get("pca_random_state", 7068))
    noisy = {
        condition["id"]: _corrupted_records(
            confirmation, condition, seed, base_seed, device, need_cached_fmt=False
        )
        for condition in spec["corruptions"]
    }
    target = Path(spec["output_root"]) / "task2" / "shards" / f"{dataset}_seed{seed}.csv"
    rows = _read_csv(target)
    completed = {(row["arm"], row["condition"]) for row in rows}
    for arm in task["arms"]:
        if all((arm["id"], condition["id"]) in completed for condition in spec["corruptions"]):
            continue
        # Train-only fitting of every transform (RMS grouping, PCA, scaler) and
        # of the VAE on clean data; calibration decides the vortex cluster name.
        train_x, calibration_x = _task2_inputs(
            train, calibration, arm["representation"], num_freq, device,
            pca_random_state=pca_random_state,
        )
        architecture = _task2_architecture(source, arm["architecture"])
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
            key = (arm["id"], condition["id"])
            if key in completed:
                continue
            replay_train_x, noisy_x = _task2_inputs(
                train, noisy[condition["id"]], arm["representation"], num_freq, device,
                pca_random_state=pca_random_state,
            )
            if not np.allclose(replay_train_x, train_x, rtol=0.0, atol=1e-5):
                raise RuntimeError("train-only transform is not deterministic across conditions")
            noisy_mu = _vae_encode(model, noisy_x, device)
            score = binary_cluster_metrics(
                confirmation_reference, kmeans.predict(noisy_mu), cluster
            )
            rows.append({
                "experiment": spec["experiment"], "task": "Task2",
                "dataset": dataset, "arm": arm["id"], "training_seed": seed,
                "representation": arm["representation"],
                "architecture": arm["architecture"], "input_dim": int(train_x.shape[1]),
                "condition": condition["id"], "kind": condition["kind"],
                "level": float(condition["level"]),
                "cluster_as_vortex": int(cluster), **score, **losses,
            })
            _write_csv(target, rows)
            completed.add(key)
            print(f"{dataset}/{arm['id']}/seed{seed}/{condition['id']}: F1={score['f1']:.5f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return target


# ------------------------------------------------------------------ Task3
def run_task3(config_path: str | Path, job_index: int) -> Path:
    spec = _load_spec(config_path)
    task = spec["task3"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    selection = _frozen_json(task["selected_config"])
    if selection["winner"]["candidate_id"] != "g08_aivd1w3_dft":
        raise RuntimeError("Task3 frozen main recipe changed")
    candidate = next(
        dict(row) for row in source["candidates"]
        if row["id"] == selection["winner"]["candidate_id"]
    )
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets):
        raise IndexError("Task3 noise array index out of range")
    dataset = datasets[index]
    _, group = task3_search._group_for_dataset(source, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = task3_search._load_records(
        source, dataset, candidate, device, ordinals=task["confirmation"]
    )
    confirmation_ordinals = {int(value) for value in task["confirmation"]}
    base_records = []
    for raw, _, labels, ordinal, metadata in records:
        if int(ordinal) not in confirmation_ordinals:
            continue
        base_records.append({
            "path": Path(metadata["source_cache"]), "ordinal": int(ordinal),
            "raw": np.asarray(raw).reshape(len(raw), -1),
            "fmt": np.zeros((len(raw), 161), dtype=np.float32),
            "reference": np.asarray(labels).astype(bool),
            "metadata": {**metadata, "dataset": dataset}, "features": {},
        })
    base_seed = int(spec["randomization"]["corruption_seed"])
    targets_written = []
    for seed_value in task["training_seeds"]:
        seed = int(seed_value)
        noisy = {}
        for condition in spec["corruptions"]:
            changed = _corrupted_records(
                base_records, condition, seed, base_seed, device, need_cached_fmt=False
            )
            raw = np.concatenate([reshape_cached_primitives(row["raw"]) for row in changed])
            labels = np.concatenate([row["reference"] for row in changed]).astype(np.float32)
            noisy[condition["id"]] = (raw, labels)
        raw_path = Path(group["raw_checkpoint_dir"]) / f"{dataset}_raw_seed{seed}.pt"
        raw_model, raw_checkpoint = _load_raw_model(raw_path, 1, device)
        wide_path = Path(group["raw_checkpoint_dir"]) / f"{dataset}_raw_wide_seed{seed}.pt"
        models = {
            "raw": (raw_model, raw_checkpoint),
            "raw_wide": _task3_wide_model(wide_path, 1, device),
        }
        raw_mean = np.asarray(raw_checkpoint["normalization"]["raw_mean"], dtype=np.float32)
        raw_std = np.asarray(raw_checkpoint["normalization"]["raw_std"], dtype=np.float32)
        target = Path(spec["output_root"]) / "task3" / "shards" / f"{dataset}_seed{seed}.csv"
        rows = _read_csv(target)
        completed = {(row["arm"], row["condition"]) for row in rows}
        for arm in task["arms"]:
            model, checkpoint = models[arm]
            for key in ("raw_mean", "raw_std"):
                if not np.array_equal(
                    np.asarray(checkpoint["normalization"][key]),
                    np.asarray(raw_checkpoint["normalization"][key]),
                ):
                    raise RuntimeError(f"{dataset}/{arm} normalization changed")
            for condition in spec["corruptions"]:
                if (arm, condition["id"]) in completed:
                    continue
                raw, labels = noisy[condition["id"]]
                normalized = ((raw - raw_mean) / raw_std).astype(np.float32)
                dummy = np.zeros((len(normalized), 1), dtype=np.float32)
                loader = _loader((normalized, dummy, labels), 1024, False, seed, device.type == "cuda")
                targets, probabilities = _predict(model, loader, device)
                score = _classification_metrics(
                    targets, probabilities, float(checkpoint["threshold"])
                )
                rows.append({
                    "experiment": spec["experiment"], "task": "Task3",
                    "dataset": dataset, "arm": arm, "training_seed": seed,
                    "condition": condition["id"], "kind": condition["kind"],
                    "level": float(condition["level"]),
                    "clean_threshold": float(checkpoint["threshold"]),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    **score,
                })
                _write_csv(target, rows)
                completed.add((arm, condition["id"]))
        targets_written.append(target)
        print(f"{dataset}/seed{seed}: {len(rows)} rows", flush=True)
    return targets_written[-1]


# --------------------------------------------------------------- summary
def _reference_rows(spec: dict) -> dict[str, list[dict]]:
    """Load 1.1 per-run rows after checking them against the 1.1 audit."""
    root = Path(spec["reference_experiment"])
    audit = _frozen_json(root / "independent_audit.json")
    if audit.get("status") != "PASS":
        raise RuntimeError("reference noise experiment lacks an independent PASS audit")
    evidence = {}
    for remote, digest in audit["evidence_sha256"].items():
        marker = "/outputs/"
        local = Path("outputs") / remote.split(marker, 1)[1] if marker in remote else Path(remote)
        evidence[local.resolve()] = digest
    source = yaml.safe_load(Path(spec["task1"]["source_config"]).read_text(encoding="utf-8"))
    reference_spec = yaml.safe_load((root.parent.parent / "config" / f"{root.name}.yaml").read_text(encoding="utf-8")) if (root.parent.parent / "config" / f"{root.name}.yaml").exists() else None
    paths = {"Task1": [root / "task1" / "per_run.csv"], "Task2": [], "Task3": []}
    seeds = {
        "Task2": [int(v) for v in spec["task2"]["training_seeds"]],
        "Task3": [int(v) for v in spec["task3"]["training_seeds"]],
    }
    for task_name in ("Task2", "Task3"):
        for dataset in source["datasets"]:
            for seed in seeds[task_name]:
                paths[task_name].append(root / task_name.lower() / "shards" / f"{dataset}_seed{seed}.csv")
    rows = {}
    for task_name, task_paths in paths.items():
        rows[task_name] = []
        for path in task_paths:
            digest = evidence.get(path.resolve())
            if digest is None:
                raise RuntimeError(f"{path} is not listed in the reference audit evidence")
            if _sha256(path) != digest:
                raise RuntimeError(f"{path} differs from the audited reference evidence")
            rows[task_name].extend(_read_csv(path))
    conditions = [row["id"] for row in spec["corruptions"]]
    for task_name, values in rows.items():
        observed = sorted({row["condition"] for row in values})
        if observed != sorted(conditions):
            raise RuntimeError(f"reference {task_name} conditions differ: {observed}")
    if reference_spec is not None:
        if reference_spec["randomization"] != spec["randomization"]:
            raise RuntimeError("corruption randomization differs from the reference experiment")
        if [row["id"] for row in reference_spec["corruptions"]] != conditions:
            raise RuntimeError("corruption conditions differ from the reference experiment")
    return rows


def _own_rows(spec: dict) -> dict[str, list[dict]]:
    output = Path(spec["output_root"])
    source = yaml.safe_load(Path(spec["task1"]["source_config"]).read_text(encoding="utf-8"))
    rows = {"Task1": _read_csv(output / "task1" / "per_run.csv"), "Task2": [], "Task3": []}
    for task_name in ("Task2", "Task3"):
        for dataset in source["datasets"]:
            for seed in spec[task_name.lower()]["training_seeds"]:
                path = output / task_name.lower() / "shards" / f"{dataset}_seed{int(seed)}.csv"
                rows[task_name].extend(_read_csv(path))
    return rows


def _arm_ids(spec: dict, task_name: str) -> list[str]:
    arms = spec[task_name.lower()]["arms"]
    return [arm if isinstance(arm, str) else arm["id"] for arm in arms]


def _dataset_macro(rows: list[dict], metric: str) -> float:
    means = []
    for dataset in sorted({row["dataset"] for row in rows}):
        means.append(np.mean([float(row[metric]) for row in rows if row["dataset"] == dataset]))
    return float(np.mean(means))


def summarize(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    source = yaml.safe_load(Path(spec["task1"]["source_config"]).read_text(encoding="utf-8"))
    conditions = spec["corruptions"]
    own = _own_rows(spec)
    expected = {
        "Task1": len(source["datasets"]) * len(spec["task1"]["arms"])
        * len(spec["task1"]["kmeans_seeds"]) * len(conditions),
        "Task2": len(source["datasets"]) * len(spec["task2"]["arms"])
        * len(spec["task2"]["training_seeds"]) * len(conditions),
        "Task3": len(source["datasets"]) * len(spec["task3"]["arms"])
        * len(spec["task3"]["training_seeds"]) * len(conditions),
    }
    observed = {name: len(rows) for name, rows in own.items()}
    if observed != expected:
        raise RuntimeError(f"incomplete noise 1.2 outputs: {observed} != {expected}")
    reference = _reference_rows(spec)
    table = []
    for task_name in ("Task1", "Task2", "Task3"):
        arms = [("reference", arm) for arm in sorted({row["arm"] for row in reference[task_name]})]
        arms += [("own", arm) for arm in _arm_ids(spec, task_name)]
        if len({arm for _, arm in arms}) != len(arms):
            raise RuntimeError(f"{task_name}: arm identifiers overlap between 1.1 and 1.2")
        for origin, arm in arms:
            rows = reference[task_name] if origin == "reference" else own[task_name]
            for condition in conditions:
                selected = [
                    row for row in rows
                    if row["condition"] == condition["id"] and row["arm"] == arm
                ]
                if not selected:
                    raise RuntimeError(f"{task_name}/{arm}/{condition['id']} has no rows")
                row = {
                    "task": task_name, "condition": condition["id"],
                    "kind": condition["kind"], "level": float(condition["level"]),
                    "arm": arm, "source_experiment": (
                        Path(spec["reference_experiment"]).name if origin == "reference"
                        else spec["experiment"]
                    ),
                    "dataset_macro_f1": _dataset_macro(selected, "f1"),
                    "dataset_count": len({r["dataset"] for r in selected}),
                    "run_count": len(selected),
                }
                if task_name == "Task3":
                    row["dataset_macro_average_precision"] = _dataset_macro(
                        selected, "average_precision"
                    )
                table.append(row)
    clean = {(row["task"], row["arm"]): row for row in table if row["condition"] == "clean"}
    for row in table:
        base = clean[(row["task"], row["arm"])]
        row["f1_change_from_clean"] = row["dataset_macro_f1"] - base["dataset_macro_f1"]
        row["f1_retention"] = (
            row["dataset_macro_f1"] / base["dataset_macro_f1"]
            if base["dataset_macro_f1"] > 0 else float("nan")
        )
        if "dataset_macro_average_precision" in row:
            row["average_precision_change_from_clean"] = (
                row["dataset_macro_average_precision"] - base["dataset_macro_average_precision"]
            )
    gains = []
    lookup = {(row["task"], row["arm"], row["condition"]): row for row in table}
    for task_name in ("Task1", "Task2", "Task3"):
        baselines = sorted({row["arm"] for row in table if row["task"] == task_name} - {"fmt"})
        for baseline in baselines:
            for condition in conditions:
                fmt = lookup[(task_name, "fmt", condition["id"])]
                other = lookup[(task_name, baseline, condition["id"])]
                item = {
                    "task": task_name, "baseline": baseline, "condition": condition["id"],
                    "kind": condition["kind"], "level": float(condition["level"]),
                    "baseline_f1": other["dataset_macro_f1"], "fmt_f1": fmt["dataset_macro_f1"],
                    "fmt_minus_baseline_f1": fmt["dataset_macro_f1"] - other["dataset_macro_f1"],
                }
                if task_name == "Task3":
                    item.update({
                        "baseline_average_precision": other["dataset_macro_average_precision"],
                        "fmt_average_precision": fmt["dataset_macro_average_precision"],
                        "fmt_minus_baseline_average_precision": (
                            fmt["dataset_macro_average_precision"]
                            - other["dataset_macro_average_precision"]
                        ),
                    })
                gains.append(item)
    output = Path(spec["output_root"])
    _write_csv(output / "robustness_table.csv", table)
    _write_csv(output / "paired_gain_table.csv", gains)
    summary = {
        "schema": 1, "experiment": spec["experiment"],
        "reference_experiment": spec["reference_experiment"],
        "record_counts": observed,
        "reference_record_counts": {name: len(rows) for name, rows in reference.items()},
        "table": table, "paired_gains": gains,
        "interpretation": (
            "All trainable and clustering models were fitted on clean data; only "
            "confirmation pathlines were corrupted. FMT and original Raw arms are the "
            "audited 1.1 rows; corruption realizations are shared through the same "
            "corruption seed, conditions and repeat seeds."
        ),
    }
    target = output / "summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([g for g in gains if g["condition"] in {"clean", "gaussian_010", "short_track_050"}], indent=2))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_Task123_NoiseRobustness_1.2.yaml")
    parser.add_argument(
        "--mode", required=True,
        choices=("task1", "task1-merge", "task2", "task3", "summarize"),
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode in {"task1", "task2", "task3"} and args.job_index is None:
        parser.error(f"{args.mode} requires --job-index")
    if args.mode == "task1":
        run_task1(args.config, args.job_index)
    elif args.mode == "task1-merge":
        merge_task1(args.config)
    elif args.mode == "task2":
        run_task2(args.config, args.job_index)
    elif args.mode == "task3":
        run_task3(args.config, args.job_index)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
