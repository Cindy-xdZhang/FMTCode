"""Run one frozen Task-wide FMT recipe on held-out 3D ordinals.

The Task2 and Task3 development selectors remain separate. This runner only
opens confirmation ordinals after the formal global selection and an
independent selection audit both exist. It never changes the selected recipe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from experiments.Run_Task2_3D_Main import _prepare_inputs
from experiments.Verify_HighReVAE import _train
from experiments import Search_Task2_FMTVAE_3D as task2_search
from experiments import Search_Task3_FMTResidual_3D as task3_search


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_confirmation_spec(path: str | Path) -> tuple[Path, dict]:
    config_path = Path(path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "experiment", "task", "selection_config", "selection_path",
        "selection_audit_path", "output_root", "splits", "training_seeds",
        "expected_slices", "checkpoint_policy",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"confirmation config misses keys: {missing}")
    task = str(spec["task"])
    if task not in {"Task2", "Task3"}:
        raise ValueError("uniform confirmation supports Task2 or Task3")
    seeds = [int(value) for value in spec["training_seeds"]]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("confirmation requires at least three unique seeds")
    split_sets = {
        name: {int(value) for value in values}
        for name, values in spec["splits"].items()
    }
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if split_sets[left] & split_sets[right]:
                raise ValueError(f"confirmation splits overlap: {left}/{right}")
    opened = set().union(*split_sets.values())
    if opened != set(range(int(spec["expected_slices"]))):
        raise ValueError(
            "confirmation splits must partition every registered ordinal"
        )
    expected = (
        {"train", "cluster_calibration", "confirmation"}
        if task == "Task2" else {"train", "validation", "confirmation"}
    )
    if set(split_sets) != expected:
        raise ValueError(f"{task} split names must be {sorted(expected)}")
    return config_path, spec


def _source_state(spec: dict) -> tuple[Path, dict, Path, dict, Path, dict]:
    source_config = Path(spec["selection_config"])
    selection_path = Path(spec["selection_path"])
    audit_path = Path(spec["selection_audit_path"])
    for path in (source_config, selection_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(path)
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(source.get("task")) != str(spec["task"]):
        raise RuntimeError("selection and confirmation tasks differ")
    if str(selection.get("experiment")) != str(source["experiment"]):
        raise RuntimeError("selection JSON and source config differ")
    if bool(selection.get("outer_ordinals_opened", True)):
        raise RuntimeError("global selector opened outer ordinals")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("global selector opened confirmation data")
    if audit.get("status") != "PASS":
        raise RuntimeError("independent selection audit did not pass")
    if not bool(audit.get("audit_is_independent_of_formal_selector", False)):
        raise RuntimeError("selection audit is not declared independent")
    if bool(audit.get("outer_or_confirmation_data_opened", True)):
        raise RuntimeError("selection audit reports held-out data opened")
    if str(audit.get("config_sha256", "")).lower() != _sha256(source_config):
        raise RuntimeError("selection audit/source config hash mismatch")
    audited = dict(audit.get("evidence_sha256", {}))
    if audited.get(selection_path.as_posix(), "").lower() != _sha256(selection_path):
        raise RuntimeError("selection JSON is absent from audited evidence")
    return source_config, source, selection_path, selection, audit_path, audit


def _manifest_path(spec: dict) -> Path:
    return Path(spec["output_root"]) / "frozen_recipe_manifest.json"


def freeze_recipe(config_path: str | Path) -> Path:
    config_path, spec = _load_confirmation_spec(config_path)
    (
        source_config, source, selection_path, selection, audit_path, audit,
    ) = _source_state(spec)
    payload = {
        "schema": 1,
        "experiment": str(spec["experiment"]),
        "task": str(spec["task"]),
        "selection_scope": "one unchanged recipe across all datasets",
        "confirmation_config_sha256": _sha256(config_path),
        "selection_config_sha256": _sha256(source_config),
        "selection_sha256": _sha256(selection_path),
        "selection_audit_sha256": _sha256(audit_path),
        "selection_audit_status": str(audit["status"]),
        "source_experiment": str(source["experiment"]),
        "winner": dict(selection["winner"]),
        "datasets": list(source["datasets"]),
        "splits": spec["splits"],
        "training_seeds": [int(value) for value in spec["training_seeds"]],
        "outer_opened_only_after_recipe_freeze": True,
        "checkpoint_policy": str(spec["checkpoint_policy"]),
    }
    target = _manifest_path(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != serialised:
        raise RuntimeError("frozen confirmation recipe changed")
    target.write_text(serialised, encoding="utf-8")
    print(target)
    return target


def _frozen_state(config_path: str | Path) -> tuple[Path, dict, dict, dict, dict]:
    config_path, spec = _load_confirmation_spec(config_path)
    source_config, source, selection_path, selection, audit_path, _ = (
        _source_state(spec)
    )
    manifest_path = _manifest_path(spec)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"freeze the global recipe before confirmation: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "confirmation_config_sha256": _sha256(config_path),
        "selection_config_sha256": _sha256(source_config),
        "selection_sha256": _sha256(selection_path),
        "selection_audit_sha256": _sha256(audit_path),
    }
    for key, value in expected.items():
        if str(manifest.get(key, "")).lower() != value:
            raise RuntimeError(f"frozen recipe identity changed: {key}")
    if manifest.get("winner") != selection.get("winner"):
        raise RuntimeError("frozen winner differs from global selection")
    if manifest.get("splits") != spec.get("splits"):
        raise RuntimeError("frozen confirmation split changed")
    if manifest.get("training_seeds") != [
        int(value) for value in spec["training_seeds"]
    ]:
        raise RuntimeError("frozen confirmation seeds changed")
    return manifest_path, manifest, spec, source, selection


def _task2_architecture(source: dict, winner: dict) -> dict:
    rows = [
        dict(row) for row in source["architectures"]
        if str(row["id"]) == str(winner["architecture"])
    ]
    if len(rows) != 1:
        raise RuntimeError("Task2 frozen architecture is ambiguous")
    return rows[0]


def _task2_score(train_mu, calibration_mu, confirmation_mu,
                 calibration_reference, confirmation_reference, spec):
    model = KMeans(
        n_clusters=2,
        random_state=int(spec["kmeans_seed"]),
        n_init=int(spec["kmeans_n_init"]),
    ).fit(train_mu)
    calibration_labels = model.predict(calibration_mu)
    vortex_cluster = calibrate_vortex_cluster(
        calibration_reference, calibration_labels
    )
    calibration = binary_cluster_metrics(
        calibration_reference, calibration_labels, vortex_cluster
    )
    confirmation = binary_cluster_metrics(
        confirmation_reference, model.predict(confirmation_mu), vortex_cluster
    )
    return int(vortex_cluster), calibration, confirmation


def _run_task2(config_path: str | Path, job_index: int) -> Path:
    manifest_path, _, spec, source, selection = _frozen_state(config_path)
    datasets = list(source["datasets"])
    arms = ("raw", "fmt")
    index = int(job_index)
    if not 0 <= index < len(datasets) * len(arms):
        raise IndexError("Task2 confirmation job index outside registered array")
    dataset_index, arm_index = divmod(index, len(arms))
    dataset, arm = datasets[dataset_index], arms[arm_index]
    winner = dict(selection["winner"])
    architecture = _task2_architecture(source, winner)
    _, group = task2_search._group_for_dataset(source, dataset)
    all_ordinals = sorted({
        int(value) for values in spec["splits"].values() for value in values
    })
    records = load_cache_records(
        Path(group["development_cache"]) / dataset,
        expected_count=int(spec["expected_slices"]),
        ordinals=all_ordinals,
    )
    by_ordinal = {int(record["ordinal"]): record for record in records}

    def take(name: str) -> list[dict]:
        return [by_ordinal[int(value)] for value in spec["splits"][name]]

    train_records = take("train")
    calibration_records = take("cluster_calibration")
    confirmation_records = take("confirmation")
    evaluation_records = [*calibration_records, *confirmation_records]
    calibration_count = sum(
        len(record["reference"]) for record in calibration_records
    )
    calibration_reference = stack_reference(calibration_records)
    confirmation_reference = stack_reference(confirmation_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, evaluation_x = _prepare_inputs(
        train_records,
        evaluation_records,
        arm,
        str(winner["fmt_feature"]),
        device,
    )
    source_config = EasyConfig(group["source_config"])
    target = (
        Path(spec["output_root"]) / "shards" / "task2"
        / f"{dataset}_{arm}.csv"
    )
    rows = _read_csv(target)
    manifest_hash = _sha256(manifest_path)
    if rows and {row["recipe_manifest_sha256"] for row in rows} != {
        manifest_hash
    }:
        raise RuntimeError(f"stale Task2 confirmation shard: {target}")
    completed = {int(row["training_seed"]) for row in rows}
    for seed_value in spec["training_seeds"]:
        seed = int(seed_value)
        if seed in completed:
            continue
        train_mu, evaluation_mu, losses = _train(
            train_x,
            evaluation_x,
            architecture,
            source_config,
            seed,
            device,
        )
        vortex_cluster, calibration, confirmation = _task2_score(
            train_mu,
            evaluation_mu[:calibration_count],
            evaluation_mu[calibration_count:],
            calibration_reference,
            confirmation_reference,
            spec,
        )
        row = {
            "experiment": spec["experiment"],
            "task": "Task2",
            "dataset": dataset,
            "arm": arm,
            "training_seed": seed,
            "recipe_manifest_sha256": manifest_hash,
            "selection_sha256": _sha256(spec["selection_path"]),
            "fmt_feature": str(winner["fmt_feature"]),
            "feature_id": str(winner["feature_id"]),
            "architecture": str(winner["architecture"]),
            "input_dim": int(train_x.shape[1]),
            "cluster_as_vortex": vortex_cluster,
            **{f"calibration_{key}": value for key, value in calibration.items()},
            **{
                f"confirmation_{key}": value
                for key, value in confirmation.items()
            },
            **losses,
        }
        rows.append(row)
        _write_csv(target, rows)
        completed.add(seed)
        print(
            f"Task2 uniform {dataset}/{arm}/seed={seed}: "
            f"F1={confirmation['f1']:.5f}",
            flush=True,
        )
    return target


def _task3_candidate(source: dict, winner: dict) -> dict:
    rows = [
        dict(row) for row in source["candidates"]
        if str(row["id"]) == str(winner["candidate_id"])
    ]
    if len(rows) != 1:
        raise RuntimeError("Task3 frozen candidate is ambiguous")
    return rows[0]


def _run_task3(config_path: str | Path, job_index: int) -> Path:
    manifest_path, _, spec, source, selection = _frozen_state(config_path)
    datasets = list(source["datasets"])
    sources = ("fmt", "raw_pca")
    index = int(job_index)
    if not 0 <= index < len(datasets) * len(sources):
        raise IndexError("Task3 confirmation job index outside registered array")
    dataset_index, source_index = divmod(index, len(sources))
    dataset, auxiliary_source = datasets[dataset_index], sources[source_index]
    winner = dict(selection["winner"])
    candidate = _task3_candidate(source, winner)
    _, group = task3_search._group_for_dataset(source, dataset)
    device_name = str(source["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    all_ordinals = sorted({
        int(value) for values in spec["splits"].values() for value in values
    })
    records = task3_search._load_records(
        source, dataset, candidate, device, ordinals=all_ordinals
    )
    train = task3_search._stack_split(records, spec["splits"]["train"])
    validation = task3_search._stack_split(
        records, spec["splits"]["validation"]
    )
    confirmation = task3_search._stack_split(
        records, spec["splits"]["confirmation"]
    )
    raw_stats = task3_search._frozen_raw_normalization(
        group, dataset, int(spec["training_seeds"][0])
    )
    train, validation, confirmation, stats = (
        task3_search._normalize_train_only(
            train, validation, confirmation, raw_stats=raw_stats
        )
    )
    fmt_dim = int(train[1].shape[1])
    target = (
        Path(spec["output_root"]) / "shards" / "task3"
        / f"{dataset}_{auxiliary_source}.csv"
    )
    rows = _read_csv(target)
    manifest_hash = _sha256(manifest_path)
    if rows and {row["recipe_manifest_sha256"] for row in rows} != {
        manifest_hash
    }:
        raise RuntimeError(f"stale Task3 confirmation shard: {target}")
    completed = {int(row["seed"]) for row in rows}
    for seed_value in spec["training_seeds"]:
        seed = int(seed_value)
        if seed in completed:
            continue
        output_dir = (
            Path(spec["output_root"]) / "temporary_training"
            / dataset / f"seed{seed}" / auxiliary_source
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        run_spec = task3_search._candidate_spec(
            source,
            group,
            candidate,
            dataset,
            seed,
            auxiliary_source,
            output_dir,
            fmt_dim,
        )
        run_spec["experiment"] = str(spec["experiment"])
        run_spec["split"] = {
            "train_ordinals": list(spec["splits"]["train"]),
            "validation_ordinals": list(spec["splits"]["validation"]),
            "test_ordinals": list(spec["splits"]["confirmation"]),
        }
        run_spec["evaluation"] = {"test_enabled": True}
        row = task3_search._train_one(
            run_spec,
            dataset,
            seed,
            (train, validation, confirmation),
            stats,
            device,
            output_dir,
        )
        row.update({
            "experiment": spec["experiment"],
            "task": "Task3",
            "candidate_id": str(winner["candidate_id"]),
            "fmt_feature": str(winner["fmt_feature"]),
            "fmt_dim": fmt_dim,
            "source": auxiliary_source,
            "recipe_manifest_sha256": manifest_hash,
            "selection_sha256": _sha256(spec["selection_path"]),
        })
        rows.append(row)
        _write_csv(target, rows)
        completed.add(seed)
        print(
            f"Task3 uniform {dataset}/{auxiliary_source}/seed={seed}: "
            f"F1={float(row['test_f1']):.5f} "
            f"AP={float(row['test_average_precision']):.5f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return target


def run_job(config_path: str | Path, job_index: int) -> Path:
    _, spec = _load_confirmation_spec(config_path)
    if spec["task"] == "Task2":
        return _run_task2(config_path, job_index)
    return _run_task3(config_path, job_index)


def _family_for_dataset(source: dict, dataset: str) -> str:
    matches = [
        name for name, group in source["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"dataset {dataset} has {len(matches)} families")
    return matches[0]


def _collect_rows(spec: dict, source: dict) -> list[dict]:
    task = str(spec["task"])
    arms = ("raw", "fmt") if task == "Task2" else ("raw_pca", "fmt")
    rows = []
    for dataset in source["datasets"]:
        for arm in arms:
            path = (
                Path(spec["output_root"]) / "shards" / task.lower()
                / f"{dataset}_{arm}.csv"
            )
            values = _read_csv(path)
            if len(values) != len(spec["training_seeds"]):
                raise RuntimeError(
                    f"incomplete {task} confirmation shard {path}: "
                    f"{len(values)}"
                )
            rows.extend(values)
    expected = len(source["datasets"]) * 2 * len(spec["training_seeds"])
    if len(rows) != expected:
        raise RuntimeError(f"{task} expected {expected} rows, found {len(rows)}")
    return rows


def _metric(row: dict, task: str, metric: str) -> float:
    key = f"confirmation_{metric}" if task == "Task2" else f"test_{metric}"
    return float(row[key])


def summarize(config_path: str | Path) -> Path:
    manifest_path, manifest, spec, source, _ = _frozen_state(config_path)
    task = str(spec["task"])
    rows = _collect_rows(spec, source)
    manifest_hash = _sha256(manifest_path)
    if {row["recipe_manifest_sha256"] for row in rows} != {manifest_hash}:
        raise RuntimeError("confirmation rows mix frozen recipe identities")
    arm_names = ("raw", "fmt") if task == "Task2" else ("raw_pca", "fmt")
    arm_field = "arm" if task == "Task2" else "source"
    seeds = [int(value) for value in spec["training_seeds"]]
    table = []
    seed_gains = {seed: [] for seed in seeds}
    dataset_details = {}
    for dataset in source["datasets"]:
        subset = [row for row in rows if row["dataset"] == dataset]
        by_arm = {}
        for arm in arm_names:
            arm_rows = [row for row in subset if row[arm_field] == arm]
            seed_key = "training_seed" if task == "Task2" else "seed"
            if {int(row[seed_key]) for row in arm_rows} != set(seeds):
                raise RuntimeError(f"{dataset}/{arm} seed set changed")
            by_arm[arm] = arm_rows
        if task == "Task3":
            for seed in seeds:
                counts = {
                    int(next(
                        row["trainable_residual_parameter_count"]
                        for row in by_arm[arm] if int(row["seed"]) == seed
                    )) for arm in arm_names
                }
                if len(counts) != 1:
                    raise RuntimeError(
                        f"Task3 capacity mismatch for {dataset}/seed{seed}"
                    )
        metrics = ("f1",) if task == "Task2" else (
            "f1", "average_precision"
        )
        details = {"family": _family_for_dataset(source, dataset)}
        for metric in metrics:
            values = {}
            for arm in arm_names:
                values[arm] = np.asarray([
                    _metric(row, task, metric) for row in by_arm[arm]
                ], dtype=np.float64)
                details[f"{arm}_{metric}"] = float(values[arm].mean())
                details[f"{arm}_{metric}_std"] = float(values[arm].std(ddof=0))
            details[f"fmt_minus_{arm_names[0]}_{metric}"] = float(
                values["fmt"].mean() - values[arm_names[0]].mean()
            )
        seed_key = "training_seed" if task == "Task2" else "seed"
        for seed in seeds:
            raw = next(
                row for row in by_arm[arm_names[0]]
                if int(row[seed_key]) == seed
            )
            fmt = next(
                row for row in by_arm["fmt"] if int(row[seed_key]) == seed
            )
            seed_gains[seed].append(
                _metric(fmt, task, "f1") - _metric(raw, task, "f1")
            )
        dataset_details[dataset] = details
        table.append({"dataset": dataset, **details})

    raw_f1 = float(np.mean([
        row[f"{arm_names[0]}_f1"] for row in dataset_details.values()
    ]))
    fmt_f1 = float(np.mean([
        row["fmt_f1"] for row in dataset_details.values()
    ]))
    family_names = sorted({row["family"] for row in dataset_details.values()})
    family_f1_gains = {
        family: float(np.mean([
            row[f"fmt_minus_{arm_names[0]}_f1"]
            for row in dataset_details.values() if row["family"] == family
        ])) for family in family_names
    }
    seed_macro = {
        str(seed): float(np.mean(values)) for seed, values in seed_gains.items()
    }
    summary = {
        "schema": 1,
        "experiment": spec["experiment"],
        "task": task,
        "selection_scope": "one unchanged recipe across all datasets",
        "winner": manifest["winner"],
        "recipe_manifest_sha256": manifest_hash,
        "selection_sha256": manifest["selection_sha256"],
        "selection_audit_sha256": manifest["selection_audit_sha256"],
        "record_count": len(rows),
        "dataset_count": len(dataset_details),
        "training_seed_count": len(seeds),
        "raw_f1_dataset_macro": raw_f1,
        "fmt_f1_dataset_macro": fmt_f1,
        "fmt_minus_raw_f1_dataset_macro": fmt_f1 - raw_f1,
        "positive_f1_dataset_count": sum(
            row[f"fmt_minus_{arm_names[0]}_f1"] > 0.0
            for row in dataset_details.values()
        ),
        "positive_f1_family_count": sum(
            value > 0.0 for value in family_f1_gains.values()
        ),
        "positive_f1_seed_count": sum(
            value > 0.0 for value in seed_macro.values()
        ),
        "family_f1_gains": family_f1_gains,
        "seed_f1_gains": seed_macro,
        "datasets": dataset_details,
    }
    if task == "Task3":
        raw_ap = float(np.mean([
            row[f"{arm_names[0]}_average_precision"]
            for row in dataset_details.values()
        ]))
        fmt_ap = float(np.mean([
            row["fmt_average_precision"] for row in dataset_details.values()
        ]))
        summary.update({
            "raw_average_precision_dataset_macro": raw_ap,
            "fmt_average_precision_dataset_macro": fmt_ap,
            "fmt_minus_raw_average_precision_dataset_macro": fmt_ap - raw_ap,
            "positive_average_precision_dataset_count": sum(
                row[f"fmt_minus_{arm_names[0]}_average_precision"] > 0.0
                for row in dataset_details.values()
            ),
        })
    output = Path(spec["output_root"]) / "confirmation"
    per_run_path = output / "per_run.csv"
    table_path = output / "paper_table.csv"
    _write_csv(per_run_path, rows)
    _write_csv(table_path, table)
    summary["per_run_sha256"] = _sha256(per_run_path)
    summary["paper_table_sha256"] = _sha256(table_path)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"))
    return summary_path


def cleanup_temporary_training(config_path: str | Path) -> Path:
    """Delete only audited temporary training artifacts under output_root."""
    _, _, spec, _, _ = _frozen_state(config_path)
    output_root = Path(spec["output_root"]).resolve()
    temporary_root = (output_root / "temporary_training").resolve()
    try:
        temporary_root.relative_to(output_root)
    except ValueError as error:
        raise RuntimeError("temporary training root escapes output_root") from error
    if temporary_root.name != "temporary_training":
        raise RuntimeError("refusing to clean an unexpected directory")
    audit_path = (
        output_root / "confirmation" / "independent_confirmation_audit.json"
    )
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("temporary artifacts require a PASS audit")
    checkpoint_count = (
        len(list(temporary_root.rglob("*.pt")))
        if temporary_root.exists() else 0
    )
    expected = int(audit.get("temporary_checkpoint_count_before_cleanup", 0))
    if checkpoint_count != expected:
        raise RuntimeError(
            "temporary checkpoint count changed after audit: "
            f"expected={expected}, observed={checkpoint_count}"
        )
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": "PASS",
        "independent_confirmation_audit_sha256": _sha256(audit_path),
        "deleted_temporary_checkpoint_count": checkpoint_count,
        "temporary_training_exists_after_cleanup": temporary_root.exists(),
    }
    target = output_root / "confirmation" / "cleanup.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("freeze", "run", "summarize", "cleanup"),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze_recipe(args.config)
    elif args.mode == "run":
        if args.job_index is None:
            parser.error("run mode requires --job-index")
        run_job(args.config, args.job_index)
    elif args.mode == "summarize":
        summarize(args.config)
    else:
        cleanup_temporary_training(args.config)


if __name__ == "__main__":
    main()
