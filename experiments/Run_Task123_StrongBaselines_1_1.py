"""Run stronger non-FMT baselines for the frozen 3D Task1--Task3 protocols."""

from __future__ import annotations

import argparse
import csv
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.PathlineClassifier_3D import PathlineBinaryClassifier3D
from FMT_Utils.RobustnessFeatures_3D import non_fmt_feature_matrix
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
    fit_kmeans_transform,
)
from experiments.Run_Task2_3D_Main import _prepare_inputs
from experiments.Verify_HighReVAE import _train
from experiments.Verify_Task3_FMTClassifier import (
    _classification_metrics,
    _loader,
    _predict,
)
from experiments.Verify_Task3_FMTResidual import _load_raw_model
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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_spec(path: str | Path) -> tuple[Path, dict]:
    config = Path(path)
    spec = yaml.safe_load(config.read_text(encoding="utf-8"))
    if set(("experiment", "output_root", "task1", "task2", "task3")) - set(spec):
        raise ValueError("strong-baseline config misses required task sections")
    return config, spec


def _task1_source(task: dict) -> dict:
    return yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))


DEFAULT_TASK1_FAMILIES = ("time_domain", "plain_fourier")


def _task1_families(task: dict) -> list[str]:
    """Return the baseline families that each freeze one confirmation winner."""
    families = [str(value) for value in task.get("baseline_families", DEFAULT_TASK1_FAMILIES)]
    if len(set(families)) != len(families) or not families:
        raise ValueError("task1.baseline_families must be a non-empty unique list")
    declared = {str(row["family"]) for row in task["representations"]}
    if declared != set(families):
        raise ValueError(
            f"task1 representations declare families {sorted(declared)}, "
            f"config freezes {families}"
        )
    return families


def _task1_source_for_dataset(source: dict, dataset: str) -> dict:
    matches = [
        value for value in source["sources"].values()
        if dataset in value["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"Task1 dataset {dataset} matched {len(matches)} sources")
    return matches[0]


def _task1_records(source: dict, dataset: str, split: str, ordinals=None):
    item = _task1_source_for_dataset(source, dataset)
    root = item[f"{split}_cache"]
    expected = int(source[f"{split}_count"])
    return load_cache_records(
        Path(root) / dataset, expected_count=expected, ordinals=ordinals
    )


def _record_by_ordinal(records: list[dict]) -> dict[int, dict]:
    result = {int(record["ordinal"]): record for record in records}
    if len(result) != len(records):
        raise RuntimeError("cache records contain duplicate ordinals")
    return result


def _task1_matrix(records: list[dict], name: str, num_freq: int) -> np.ndarray:
    return np.concatenate([
        non_fmt_feature_matrix(record["raw"], name, num_freq=num_freq)
        for record in records
    ])


def _task1_score(train_records, evaluate_records, name, pca_dim, seed, n_init, num_freq):
    train = _task1_matrix(train_records, name, num_freq)
    evaluate = _task1_matrix(evaluate_records, name, num_freq)
    fitted = fit_kmeans_transform(train, pca_dim, seed, n_init)
    labels = fitted.predict(evaluate)
    reference = stack_reference(evaluate_records)
    vortex_cluster = calibrate_vortex_cluster(reference, labels)
    return binary_cluster_metrics(reference, labels, vortex_cluster)


def task1_select(config_path: str | Path, job_index: int) -> Path:
    config, spec = _load_spec(config_path)
    task = spec["task1"]
    source = _task1_source(task)
    output = Path(spec["output_root"]) / "task1"
    output.mkdir(parents=True, exist_ok=True)
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets):
        raise IndexError("Task1 selection array index is out of range")
    dataset = datasets[index]
    rows = []
    train_ids = [int(value) for value in task["selection_train"]]
    validation_ids = [int(value) for value in task["selection_validation"]]
    opened = sorted(set(train_ids) | set(validation_ids))
    records = _record_by_ordinal(
        _task1_records(source, dataset, "development", opened)
    )
    train_records = [records[value] for value in train_ids]
    validation_records = [records[value] for value in validation_ids]
    # Cache each representation once; PCA/KMeans vary, not the input feature.
    matrices = {
        representation["name"]: (
            _task1_matrix(train_records, representation["name"], int(task["num_freq"])),
            _task1_matrix(validation_records, representation["name"], int(task["num_freq"])),
        )
        for representation in task["representations"]
    }
    reference = stack_reference(validation_records)
    for representation in task["representations"]:
        train_x, validation_x = matrices[representation["name"]]
        for pca_dim in task["pca_dims"]:
            fitted = fit_kmeans_transform(
                train_x, pca_dim, int(task["selection_seed"]),
                int(task["selection_kmeans_n_init"]),
            )
            labels = fitted.predict(validation_x)
            vortex_cluster = calibrate_vortex_cluster(reference, labels)
            score = binary_cluster_metrics(reference, labels, vortex_cluster)
            rows.append({
                "dataset": dataset,
                "family": source["families"][dataset],
                "baseline_id": representation["id"],
                "representation": representation["name"],
                "baseline_family": representation["family"],
                "pca_dim": "none" if pca_dim is None else int(pca_dim),
                **score,
            })
    target = output / "development_selection_shards" / f"{dataset}.csv"
    _write_csv(target, rows)
    return target


def task1_freeze(config_path: str | Path) -> Path:
    config, spec = _load_spec(config_path)
    task = spec["task1"]
    source = _task1_source(task)
    output = Path(spec["output_root"]) / "task1"
    rows = []
    for dataset in source["datasets"]:
        rows.extend(_read_csv(
            output / "development_selection_shards" / f"{dataset}.csv"
        ))
    expected = (
        len(source["datasets"]) * len(task["representations"])
        * len(task["pca_dims"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"incomplete Task1 selection: {len(rows)} != {expected}")
    _write_csv(output / "development_selection.csv", rows)
    winners = {}
    leaderboard = []
    datasets = list(source["datasets"])
    for family in _task1_families(task):
        candidates = sorted({
            (row["baseline_id"], row["representation"], row["pca_dim"])
            for row in rows if row["baseline_family"] == family
        })
        ranked = []
        for baseline_id, representation, pca_dim in candidates:
            subset = [
                row for row in rows
                if row["baseline_id"] == baseline_id
                and row["pca_dim"] == pca_dim
            ]
            if {row["dataset"] for row in subset} != set(datasets):
                continue
            f1 = np.asarray([float(row["f1"]) for row in subset])
            ari = np.asarray([float(row["ari"]) for row in subset])
            ranked.append({
                "baseline_family": family,
                "baseline_id": baseline_id,
                "representation": representation,
                "pca_dim": pca_dim,
                "dataset_macro_f1": float(f1.mean()),
                "worst_dataset_f1": float(f1.min()),
                "dataset_macro_ari": float(ari.mean()),
            })
        ranked.sort(
            key=lambda row: (
                row["dataset_macro_f1"], row["worst_dataset_f1"],
                row["dataset_macro_ari"], row["baseline_id"], str(row["pca_dim"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        if not ranked:
            raise RuntimeError(f"Task1 {family} leaderboard is empty")
        leaderboard.extend(ranked)
        winners[family] = ranked[0]
    _write_csv(output / "development_leaderboard.csv", leaderboard)
    manifest = {
        "schema": 1,
        "experiment": spec["experiment"],
        "task": "Task1",
        "status": "frozen_before_confirmation",
        "opened_development_ordinals": sorted(
            set(int(value) for value in task["selection_train"])
            | set(int(value) for value in task["selection_validation"])
        ),
        "confirmation_opened": False,
        "selection_rule": task["selection_rule"],
        "config_sha256": _sha256(config),
        "source_config_sha256": _sha256(task["source_config"]),
        "winners": winners,
    }
    target = output / "frozen_baselines.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def task1_run(config_path: str | Path, job_index: int) -> Path:
    config, spec = _load_spec(config_path)
    task = spec["task1"]
    source = _task1_source(task)
    output = Path(spec["output_root"]) / "task1"
    manifest_path = output / "frozen_baselines.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != _sha256(config):
        raise RuntimeError("Task1 strong-baseline config changed after freeze")
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets):
        raise IndexError("Task1 confirmation array index is out of range")
    dataset = datasets[index]
    rows = []
    development = _record_by_ordinal(_task1_records(
        source, dataset, "development", task["development_ordinals"]
    ))
    confirmation = _task1_records(source, dataset, "confirmation")
    train = [development[int(value)] for value in task["final_train"]]
    calibration = [development[int(value)] for value in task["cluster_calibration"]]
    calibration_reference = stack_reference(calibration)
    confirmation_reference = stack_reference(confirmation)
    for baseline_family, winner in manifest["winners"].items():
        pca_dim = None if winner["pca_dim"] == "none" else int(winner["pca_dim"])
        train_x = _task1_matrix(train, winner["representation"], int(task["num_freq"]))
        calibration_x = _task1_matrix(
            calibration, winner["representation"], int(task["num_freq"])
        )
        confirmation_x = _task1_matrix(
            confirmation, winner["representation"], int(task["num_freq"])
        )
        for seed in task["final_kmeans_seeds"]:
            fitted = fit_kmeans_transform(
                train_x, pca_dim, int(seed), int(task["final_kmeans_n_init"])
            )
            calibration_labels = fitted.predict(calibration_x)
            vortex_cluster = calibrate_vortex_cluster(
                calibration_reference, calibration_labels
            )
            score = binary_cluster_metrics(
                confirmation_reference, fitted.predict(confirmation_x), vortex_cluster
            )
            rows.append({
                "experiment": spec["experiment"], "task": "Task1",
                "dataset": dataset, "family": source["families"][dataset],
                "baseline_family": baseline_family,
                "baseline_id": winner["baseline_id"],
                "representation": winner["representation"],
                "pca_dim": winner["pca_dim"], "kmeans_seed": int(seed),
                "cluster_as_vortex": int(vortex_cluster), **score,
            })
    target = output / "confirmation_shards" / f"{dataset}.csv"
    _write_csv(target, rows)
    print(target)
    return target


def task1_merge(config_path: str | Path) -> Path:
    _, spec = _load_spec(config_path)
    task = spec["task1"]
    source = _task1_source(task)
    output = Path(spec["output_root"]) / "task1"
    rows = []
    for dataset in source["datasets"]:
        rows.extend(_read_csv(output / "confirmation_shards" / f"{dataset}.csv"))
    expected = (
        len(source["datasets"]) * len(_task1_families(task))
        * len(task["final_kmeans_seeds"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"incomplete Task1 confirmation: {len(rows)} != {expected}")
    target = output / "confirmation_runs.csv"
    _write_csv(target, rows)
    return target


def _task2_architecture(source: dict, identifier: str) -> dict:
    matches = [row for row in source["architectures"] if row["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"Task2 architecture {identifier} is ambiguous")
    return dict(matches[0])


def task2_freeze(config_path: str | Path) -> Path:
    config, spec = _load_spec(config_path)
    task = spec["task2"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    selection = json.loads(Path(task["source_selection"]).read_text(encoding="utf-8"))
    audit = json.loads(Path(task["source_selection_audit"]).read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or not audit.get(
        "audit_is_independent_of_formal_selector", False
    ):
        raise RuntimeError("Task2 source selection lacks an independent PASS audit")
    if selection.get("outer_ordinals_opened", True):
        raise RuntimeError("Task2 source selection opened outer ordinals")
    leaderboard = _read_csv(Path(source["output_root"]) / "global_leaderboard.csv")
    raw_by_architecture = {}
    for row in leaderboard:
        raw_by_architecture[str(row["architecture"])] = float(row["raw_f1_macro"])
    strongest = max(raw_by_architecture, key=raw_by_architecture.get)
    raw_tuned = next(row for row in task["arms"] if row["id"] == "raw_tuned")
    if strongest != raw_tuned["architecture"]:
        raise RuntimeError(
            f"configured Raw-tuned architecture {raw_tuned['architecture']} is not "
            f"development winner {strongest}"
        )
    for arm in task["arms"]:
        _task2_architecture(source, arm["architecture"])
    manifest = {
        "schema": 1,
        "experiment": spec["experiment"],
        "task": "Task2",
        "status": "frozen_before_confirmation",
        "confirmation_opened": False,
        "config_sha256": _sha256(config),
        "source_config_sha256": _sha256(task["source_config"]),
        "source_selection_sha256": _sha256(task["source_selection"]),
        "source_selection_audit_sha256": _sha256(task["source_selection_audit"]),
        "raw_architecture_development_macro_f1": raw_by_architecture,
        "strongest_raw_architecture": strongest,
        "arms": task["arms"],
        "main_comparison_unchanged": True,
        "role": "appendix pressure-test baselines",
    }
    output = Path(spec["output_root"]) / "task2"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "frozen_baselines.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def _task2_group(source: dict, dataset: str) -> dict:
    matches = [group for group in source["groups"].values() if dataset in group["datasets"]]
    if len(matches) != 1:
        raise ValueError(f"Task2 dataset {dataset} matched {len(matches)} groups")
    return matches[0]


def _standardize(train: np.ndarray, evaluate: np.ndarray):
    scaler = StandardScaler().fit(train)
    return (
        scaler.transform(train).astype(np.float32),
        scaler.transform(evaluate).astype(np.float32),
    )


def _task2_inputs(train_records, evaluate_records, representation, num_freq, device,
                  pca_random_state=7068):
    if representation == "center_relative":
        return _prepare_inputs(
            train_records, evaluate_records, "raw", "fmt_all+kin4", device
        )
    if representation == "center_relative_pca189":
        train, evaluate = _prepare_inputs(
            train_records, evaluate_records, "raw", "fmt_all+kin4", device
        )
        pca = PCA(
            n_components=189, svd_solver="randomized",
            random_state=int(pca_random_state),
        )
        train = pca.fit_transform(train).astype(np.float32)
        evaluate = pca.transform(evaluate).astype(np.float32)
        return _standardize(train, evaluate)
    train = np.concatenate([
        non_fmt_feature_matrix(record["raw"], representation, num_freq=num_freq)
        for record in train_records
    ])
    evaluate = np.concatenate([
        non_fmt_feature_matrix(record["raw"], representation, num_freq=num_freq)
        for record in evaluate_records
    ])
    return _standardize(train, evaluate)


def task2_run(config_path: str | Path, job_index: int) -> Path:
    config, spec = _load_spec(config_path)
    task = spec["task2"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    manifest_path = Path(spec["output_root"]) / "task2" / "frozen_baselines.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != _sha256(config):
        raise RuntimeError("Task2 strong-baseline config changed after freeze")
    arms = list(task["arms"])
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets) * len(arms):
        raise IndexError("Task2 strong-baseline array index is out of range")
    dataset_index, arm_index = divmod(index, len(arms))
    dataset, arm = datasets[dataset_index], arms[arm_index]
    group = _task2_group(source, dataset)
    all_ordinals = sorted({
        int(value) for key in ("train", "cluster_calibration", "confirmation")
        for value in task[key]
    })
    records = load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(source["expected_slices"]),
        ordinals=all_ordinals,
    )
    by_ordinal = _record_by_ordinal(records)
    train_records = [by_ordinal[int(value)] for value in task["train"]]
    calibration_records = [
        by_ordinal[int(value)] for value in task["cluster_calibration"]
    ]
    confirmation_records = [
        by_ordinal[int(value)] for value in task["confirmation"]
    ]
    evaluation_records = [*calibration_records, *confirmation_records]
    calibration_count = sum(len(row["reference"]) for row in calibration_records)
    calibration_reference = stack_reference(calibration_records)
    confirmation_reference = stack_reference(confirmation_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, evaluate_x = _task2_inputs(
        train_records, evaluation_records, arm["representation"],
        int(task["num_freq"]), device,
        pca_random_state=int(task.get("pca_random_state", 7068)),
    )
    architecture = _task2_architecture(source, arm["architecture"])
    source_config = EasyConfig(group["source_config"])
    target = (
        Path(spec["output_root"]) / "task2" / "shards"
        / f"{dataset}_{arm['id']}.csv"
    )
    rows = _read_csv(target)
    completed = {int(row["training_seed"]) for row in rows}
    for seed_value in task["training_seeds"]:
        seed = int(seed_value)
        if seed in completed:
            continue
        train_mu, evaluation_mu, losses = _train(
            train_x, evaluate_x, architecture, source_config, seed, device
        )
        model = KMeans(
            n_clusters=2, random_state=int(task["kmeans_seed"]),
            n_init=int(task["kmeans_n_init"]),
        ).fit(train_mu)
        calibration_labels = model.predict(evaluation_mu[:calibration_count])
        vortex_cluster = calibrate_vortex_cluster(
            calibration_reference, calibration_labels
        )
        score = binary_cluster_metrics(
            confirmation_reference,
            model.predict(evaluation_mu[calibration_count:]),
            vortex_cluster,
        )
        rows.append({
            "experiment": spec["experiment"],
            "task": "Task2",
            "dataset": dataset,
            "arm": arm["id"],
            "category": arm["category"],
            "representation": arm["representation"],
            "architecture": arm["architecture"],
            "input_dim": int(train_x.shape[1]),
            "training_seed": seed,
            "cluster_as_vortex": int(vortex_cluster),
            **score,
            **losses,
        })
        _write_csv(target, rows)
        completed.add(seed)
        print(f"{dataset}/{arm['id']}/seed{seed}: F1={score['f1']:.5f}", flush=True)
    return target


def _task3_wide_model(checkpoint_path: Path, fmt_dim: int, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "raw_wide":
        raise ValueError(f"expected Raw-wide checkpoint at {checkpoint_path}")
    model_spec = checkpoint["config"]["model"]
    model = PathlineBinaryClassifier3D(
        variant="raw_wide", fmt_dim=int(fmt_dim),
        temporal_width=int(model_spec["temporal_width"]),
        embedding_dim=int(model_spec["embedding_dim"]),
        auxiliary_dim=int(model_spec["auxiliary_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def task3_run(config_path: str | Path, job_index: int) -> Path:
    _, spec = _load_spec(config_path)
    task = spec["task3"]
    source = yaml.safe_load(Path(task["source_config"]).read_text(encoding="utf-8"))
    datasets = list(source["datasets"])
    index = int(job_index)
    if not 0 <= index < len(datasets):
        raise IndexError("Task3 strong-baseline array index is out of range")
    dataset = datasets[index]
    candidate = next(
        dict(row) for row in source["candidates"]
        if row["id"] == "g08_aivd1w3_dft"
    )
    _, group = task3_search._group_for_dataset(source, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = task3_search._load_records(
        source, dataset, candidate, device, ordinals=task["confirmation"]
    )
    confirmation = task3_search._stack_split(records, task["confirmation"])
    target = Path(spec["output_root"]) / "task3" / "shards" / f"{dataset}.csv"
    rows = _read_csv(target)
    completed = {(row["baseline"], int(row["seed"])) for row in rows}
    for seed_value in task["seeds"]:
        seed = int(seed_value)
        raw_path = Path(group["raw_checkpoint_dir"]) / f"{dataset}_raw_seed{seed}.pt"
        raw_model, raw_checkpoint = _load_raw_model(raw_path, 1, device)
        raw_mean = np.asarray(raw_checkpoint["normalization"]["raw_mean"], dtype=np.float32)
        raw_std = np.asarray(raw_checkpoint["normalization"]["raw_std"], dtype=np.float32)
        normalized_raw = ((confirmation[0] - raw_mean) / raw_std).astype(np.float32)
        dummy = np.zeros((len(normalized_raw), 1), dtype=np.float32)
        split = (normalized_raw, dummy, confirmation[2].astype(np.float32))
        loader = _loader(split, 1024, False, seed, device.type == "cuda")
        models = {"raw": (raw_model, raw_checkpoint)}
        wide_path = (
            Path(group["raw_checkpoint_dir"]) / f"{dataset}_raw_wide_seed{seed}.pt"
        )
        models["raw_wide"] = _task3_wide_model(wide_path, 1, device)
        for baseline, (model, checkpoint) in models.items():
            if (baseline, seed) in completed:
                continue
            for key in ("raw_mean", "raw_std"):
                if not np.array_equal(
                    np.asarray(checkpoint["normalization"][key]),
                    np.asarray(raw_checkpoint["normalization"][key]),
                ):
                    raise RuntimeError(f"{dataset}/{baseline} normalization changed")
            targets, probabilities = _predict(model, loader, device)
            score = _classification_metrics(
                targets, probabilities, float(checkpoint["threshold"])
            )
            rows.append({
                "experiment": spec["experiment"],
                "task": "Task3",
                "dataset": dataset,
                "baseline": baseline,
                "seed": seed,
                "threshold": float(checkpoint["threshold"]),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                **score,
            })
            _write_csv(target, rows)
            completed.add((baseline, seed))
            print(f"{dataset}/{baseline}/seed{seed}: F1={score['f1']:.5f}")
    return target


def _aggregate(rows: list[dict], arm_key: str, metric: str) -> list[dict]:
    result = []
    for (dataset, arm) in sorted({(row["dataset"], row[arm_key]) for row in rows}):
        values = np.asarray([
            float(row[metric]) for row in rows
            if row["dataset"] == dataset and row[arm_key] == arm
        ])
        result.append({
            "dataset": dataset,
            "arm": arm,
            f"{metric}_mean": float(values.mean()),
            f"{metric}_std": float(values.std(ddof=0)),
            "repeat_count": int(len(values)),
        })
    return result


def summarize(config_path: str | Path) -> Path:
    _, spec = _load_spec(config_path)
    output = Path(spec["output_root"])
    task1_rows = _read_csv(output / "task1" / "confirmation_runs.csv")
    task2_rows = []
    source2 = yaml.safe_load(
        Path(spec["task2"]["source_config"]).read_text(encoding="utf-8")
    )
    for dataset in source2["datasets"]:
        for arm in spec["task2"]["arms"]:
            task2_rows.extend(_read_csv(
                output / "task2" / "shards" / f"{dataset}_{arm['id']}.csv"
            ))
    task3_rows = []
    source3 = yaml.safe_load(
        Path(spec["task3"]["source_config"]).read_text(encoding="utf-8")
    )
    for dataset in source3["datasets"]:
        task3_rows.extend(_read_csv(
            output / "task3" / "shards" / f"{dataset}.csv"
        ))
    task1_families = _task1_families(spec["task1"])
    expected = {
        "task1": len(source2["datasets"]) * len(task1_families)
        * len(spec["task1"]["final_kmeans_seeds"]),
        "task2": len(source2["datasets"]) * len(spec["task2"]["arms"])
        * len(spec["task2"]["training_seeds"]),
        "task3": len(source3["datasets"]) * len(spec["task3"]["baselines"])
        * len(spec["task3"]["seeds"]),
    }
    observed = {"task1": len(task1_rows), "task2": len(task2_rows), "task3": len(task3_rows)}
    if observed != expected:
        raise RuntimeError(f"incomplete strong-baseline outputs: {observed} != {expected}")
    tables = {
        "task1": _aggregate(task1_rows, "baseline_family", "f1"),
        "task2": _aggregate(task2_rows, "arm", "f1"),
        "task3_f1": _aggregate(task3_rows, "baseline", "f1"),
        "task3_ap": _aggregate(task3_rows, "baseline", "average_precision"),
    }
    summary = {
        "schema": 1,
        "experiment": spec["experiment"],
        "record_counts": observed,
        "tables": tables,
        "dataset_macro": {
            name: float(np.mean([row[f"{metric}_mean"] for row in table]))
            for name, table, metric in (
                *[(f"task1_{family}_f1", [r for r in tables["task1"] if r["arm"] == family], "f1")
                  for family in task1_families],
                *[(f"task2_{arm['id']}_f1", [r for r in tables["task2"] if r["arm"] == arm["id"]], "f1") for arm in spec["task2"]["arms"]],
                ("task3_raw_f1", [r for r in tables["task3_f1"] if r["arm"] == "raw"], "f1"),
                ("task3_raw_wide_f1", [r for r in tables["task3_f1"] if r["arm"] == "raw_wide"], "f1"),
                ("task3_raw_ap", [r for r in tables["task3_ap"] if r["arm"] == "raw"], "average_precision"),
                ("task3_raw_wide_ap", [r for r in tables["task3_ap"] if r["arm"] == "raw_wide"], "average_precision"),
            )
        },
        "main_summaries": {
            "task1": json.loads(Path(spec["task1"]["main_summary"]).read_text(encoding="utf-8")),
            "task2": json.loads(Path(spec["task2"]["main_summary"]).read_text(encoding="utf-8")),
            "task3": json.loads(Path(spec["task3"]["main_summary"]).read_text(encoding="utf-8")),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _write_csv(output / f"{name}.csv", table)
    target = output / "summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["dataset_macro"], indent=2))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Verify_Task123_StrongBaselines_1.1.yaml"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "task1-select", "task1-freeze", "task1-run", "task1-merge",
            "task2-freeze", "task2-run",
            "task3-run", "summarize",
        ),
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "task1-select":
        if args.job_index is None:
            parser.error("task1-select requires --job-index")
        task1_select(args.config, args.job_index)
    elif args.mode == "task1-freeze":
        task1_freeze(args.config)
    elif args.mode == "task1-run":
        if args.job_index is None:
            parser.error("task1-run requires --job-index")
        task1_run(args.config, args.job_index)
    elif args.mode == "task1-merge":
        task1_merge(args.config)
    elif args.mode == "task2-freeze":
        task2_freeze(args.config)
    elif args.mode == "task2-run":
        if args.job_index is None:
            parser.error("task2-run requires --job-index")
        task2_run(args.config, args.job_index)
    elif args.mode == "task3-run":
        if args.job_index is None:
            parser.error("task3-run requires --job-index")
        task3_run(args.config, args.job_index)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
