"""Sealed spatial confirmation for the frozen Task3 11.1 winner.

The static preflight never opens the 11.1 selector or the confirmation
population.  The freeze stage runs only after the complete 11.1 selector,
records every relevant SHA-256, and still does not read confirmation samples.
Only subsequent cache/label/evaluation jobs may expose the pre-registered 5.2
spatial phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

import Build_Task3_SpatialRobust_Confirmation_5_2 as spatial
from Evaluate_Task3_FrozenConfirmation import (
    _evaluate_residual,
    _load_residual,
)
from Run_Task3_FMTResidual_Frozen_4_1 import _load_confirmation
from Search_Task3_FMTResidual_3D import (
    _group_for_dataset,
    _read_csv,
    _write_csv,
)
from Search_Task3_LossOptimization_7_1 import (
    _load_optimization_spec,
    _optimization_candidate,
    _result_path,
)
from Verify_Task3_FMTClassifier import _stack_split


ROOT = Path(__file__).resolve().parent


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "optimization_repo_root",
        "optimization_config", "optimization_selection",
        "optimization_preflight_manifest", "expected_optimization_experiment",
        "recipe_manifest", "confirmation_count", "confirmation_roots",
        "paired_seeds", "datasets", "expected_ivd_percentile",
        "confirmation_seed_grid_phase", "require_confirmation_reference_match",
        "target_dataset_macro_f1_gain", "aspirational_dataset_macro_f1_gain",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing Task3 12.1 config keys: {missing}")
    datasets = [str(value) for value in spec["datasets"]]
    seeds = [int(value) for value in spec["paired_seeds"]]
    if len(datasets) != len(set(datasets)) or len(datasets) != 10:
        raise ValueError("Task3 12.1 requires ten unique datasets")
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("paired_seeds must be non-empty and unique")
    if int(spec["confirmation_count"]) != 4:
        raise ValueError("Task3 12.1 requires four confirmation slices")
    if not np.isclose(float(spec["expected_ivd_percentile"]), 95.0):
        raise ValueError("Task3 12.1 requires whole-field IVD-p95 labels")
    if not bool(spec["require_confirmation_reference_match"]):
        raise ValueError("Task3 12.1 must require source/reference label identity")
    phase = [float(value) for value in spec["confirmation_seed_grid_phase"]]
    if phase != [float(value) for value in spatial.SEED_GRID_PHASE]:
        raise ValueError("Task3 12.1 confirmation seed-grid phase changed")
    roots = dict(spec["confirmation_roots"])
    if set(roots) != set(spatial.SETTINGS):
        raise ValueError("Task3 12.1 confirmation groups changed")
    grouped = []
    for group_name, group in roots.items():
        group_datasets = [str(value) for value in group.get("datasets", [])]
        expected_group = list(spatial.SETTINGS[group_name]["indices"])
        if group_datasets != expected_group:
            raise ValueError(
                f"Task3 12.1 {group_name} dataset order changed"
            )
        if Path(str(group.get("source_root"))) != Path(str(
            spatial.SETTINGS[group_name]["cache_dir"]
        )):
            raise ValueError(
                f"Task3 12.1 {group_name} source root changed"
            )
        label_spec = yaml.safe_load(Path(
            spatial.SETTINGS[group_name]["label_config"]
        ).read_text(encoding="utf-8"))
        expected_label_root = str(Path(label_spec["output_dir"]) / "labels")
        if Path(str(group.get("label_root"))) != Path(expected_label_root):
            raise ValueError(
                f"Task3 12.1 {group_name} label root changed"
            )
        grouped.extend(group_datasets)
    if len(grouped) != len(set(grouped)) or set(grouped) != set(datasets):
        raise ValueError("Task3 12.1 confirmation roots do not partition datasets")
    if bool(spec.get("confirmation_opened_by_static_preflight", True)):
        raise RuntimeError("static preflight must not open confirmation")
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    spec["datasets"] = datasets
    spec["paired_seeds"] = seeds
    spec["confirmation_seed_grid_phase"] = phase
    return spec


def _optimization_root(spec: dict) -> Path:
    override = os.environ.get("TASK3_OPTIMIZATION_REPO_ROOT")
    return Path(override if override else spec["optimization_repo_root"])


def _under(root: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else root / value


def _optimization_state(spec: dict, require_selection: bool) -> tuple:
    optimization_root = _optimization_root(spec)
    config_path = _under(optimization_root, spec["optimization_config"])
    optimization = _load_optimization_spec(config_path)
    optimization["output_root"] = str(
        _under(optimization_root, optimization["output_root"])
    )
    preflight_path = _under(
        optimization_root, spec["optimization_preflight_manifest"]
    )
    if not require_selection:
        return optimization_root, optimization, preflight_path, None, None
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    selection_path = _under(
        optimization_root, spec["optimization_selection"]
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if str(selection.get("experiment")) != str(
        spec["expected_optimization_experiment"]
    ):
        raise RuntimeError("Task3 11.1 selection experiment changed")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("Task3 11.1 selector opened confirmation")
    if selection.get("preflight_manifest_sha256") != _sha256(preflight_path):
        raise RuntimeError("Task3 11.1 selector/preflight hash mismatch")
    if selection.get("optimization_config_sha256") != optimization[
        "optimization_config_sha256"
    ]:
        raise RuntimeError("Task3 11.1 selector/config hash mismatch")
    if set(selection.get("primary_by_group", {})) != set(
        optimization["groups"]
    ):
        raise RuntimeError("Task3 11.1 selected physical families changed")
    if [int(value) for value in selection.get("paired_seeds", [])] != spec[
        "paired_seeds"
    ]:
        raise RuntimeError("Task3 11.1 paired seeds changed")
    return optimization_root, optimization, preflight_path, selection_path, selection


def static_preflight(config_path: str | Path) -> dict:
    spec = _load_spec(config_path)
    _, optimization, preflight_path, _, _ = _optimization_state(
        spec, require_selection=False
    )
    payload = {
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "expected_optimization_experiment": spec[
            "expected_optimization_experiment"
        ],
        "optimization_candidate_count": len(
            optimization["optimization_candidates"]
        ),
        "dataset_count": len(spec["datasets"]),
        "paired_seed_count": len(spec["paired_seeds"]),
        "paired_arm_count": 2,
        "expected_confirmation_runs": (
            len(spec["datasets"]) * len(spec["paired_seeds"]) * 2
        ),
        "optimization_preflight_path": str(preflight_path),
        "optimization_selection_opened": False,
        "confirmation_opened": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _selected_candidate(optimization: dict, preflight: dict,
                        selection: dict, dataset: str) -> tuple[str, dict]:
    group_name, _ = _group_for_dataset(optimization, dataset)
    row = selection["primary_by_group"][group_name]
    selected_id = str(row["optimization_id"])
    indexes = [
        index for index, candidate in enumerate(
            optimization["optimization_candidates"]
        ) if str(candidate["id"]) == selected_id
    ]
    if len(indexes) != 1:
        raise RuntimeError(f"selected Task3 recipe not found: {selected_id}")
    candidate = _optimization_candidate(
        optimization, preflight, dataset, indexes[0]
    )
    if json.loads(str(row["optimization_recipe_json"])) != candidate[
        "optimization_recipe"
    ]:
        raise RuntimeError(f"resolved Task3 recipe changed for {group_name}")
    return group_name, candidate


def _confirmation_artifact_counts(spec: dict) -> dict:
    """Count unopened confirmation artifacts without loading any sample."""
    result = {}
    for group_name, group in spec["confirmation_roots"].items():
        source_root = Path(group["source_root"])
        label_root = Path(group["label_root"])
        result[group_name] = {
            "source_npz": (
                sum(1 for _ in source_root.rglob("*.npz"))
                if source_root.exists() else 0
            ),
            "label_npz": (
                sum(1 for _ in label_root.rglob("*.npz"))
                if label_root.exists() else 0
            ),
        }
    return result


def freeze(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    (optimization_root, optimization, preflight_path, selection_path,
     selection) = _optimization_state(spec, require_selection=True)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    source_manifest = spatial._assert_recipe_frozen()
    if list(source_manifest["seed_grid_phase"]) != spec[
        "confirmation_seed_grid_phase"
    ]:
        raise RuntimeError("Task3 5.2 confirmation phase changed before freeze")
    artifact_counts = _confirmation_artifact_counts(spec)
    if any(
        count
        for group in artifact_counts.values()
        for count in group.values()
    ):
        raise RuntimeError(
            "Task3 12.1 confirmation population was opened before recipe freeze"
        )
    selected = {}
    for group_name, group in optimization["groups"].items():
        dataset = str(group["datasets"][0])
        observed_group, candidate = _selected_candidate(
            optimization, preflight, selection, dataset
        )
        if observed_group != group_name:
            raise RuntimeError("physical-family lookup changed")
        selected[group_name] = {
            "optimization_id": candidate["optimization_id"],
            "optimization_recipe": candidate["optimization_recipe"],
            "fmt_feature": candidate["fmt_feature"],
            "upstream_candidate_id": candidate["upstream_candidate_id"],
        }
    payload = {
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "optimization_repo_root": str(optimization_root),
        "optimization_config_sha256": optimization[
            "optimization_config_sha256"
        ],
        "optimization_preflight_manifest": str(preflight_path),
        "optimization_preflight_manifest_sha256": _sha256(preflight_path),
        "optimization_selection": str(selection_path),
        "optimization_selection_sha256": _sha256(selection_path),
        "selected_by_group": selected,
        "paired_seeds": spec["paired_seeds"],
        "confirmation_source_experiment": source_manifest["experiment"],
        "confirmation_seed_grid_phase": source_manifest["seed_grid_phase"],
        "confirmation_artifact_counts_at_freeze": artifact_counts,
        "confirmation_data_opened": False,
    }
    source_staging = spatial.source_staging_identity()
    if source_staging is not None:
        payload["source_staging"] = source_staging
    target = Path(spec["recipe_manifest"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError("Task3 12.1 frozen recipe manifest changed")
    else:
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(target)
    return target


def _frozen_state(spec: dict) -> tuple:
    manifest_path = Path(spec["recipe_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != spec["config_sha256"]:
        raise RuntimeError("Task3 12.1 config changed after freeze")
    state = _optimization_state(spec, require_selection=True)
    preflight_path, selection_path = state[2], state[3]
    if _sha256(preflight_path) != manifest[
        "optimization_preflight_manifest_sha256"
    ]:
        raise RuntimeError("Task3 11.1 preflight changed after freeze")
    if _sha256(selection_path) != manifest["optimization_selection_sha256"]:
        raise RuntimeError("Task3 11.1 selection changed after freeze")
    staged = spatial.source_staging_identity()
    frozen_staged = manifest.get("source_staging")
    if staged is not None and staged != frozen_staged:
        raise RuntimeError("Task3 confirmation source staging changed after freeze")
    if frozen_staged is not None and staged is None:
        raise RuntimeError("frozen Task3 source staging manifest is unavailable")
    return (*state, manifest_path, manifest)


def source_preflight(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    _frozen_state(spec)
    report = spatial.source_preflight()
    target = Path(spec["output_root"]) / "source_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous != report:
            raise RuntimeError("Task3 source preflight changed")
    else:
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(target)
    return target


def build_cache(config_path: str | Path, job_index: int,
                overwrite: bool = False) -> Path:
    spec = _load_spec(config_path)
    _frozen_state(spec)
    return spatial.build_job(int(job_index), overwrite=bool(overwrite))


def build_labels(config_path: str | Path, group_index: int,
                 overwrite: bool = False) -> Path:
    spec = _load_spec(config_path)
    _frozen_state(spec)
    groups = ("old8", "new2")
    index = int(group_index)
    if not 0 <= index < len(groups):
        raise IndexError("label group index outside [0,2)")
    group = groups[index]
    spatial.build_labels(spatial.SETTINGS[group]["label_config"], overwrite)
    return Path(spatial.SETTINGS[group]["label_config"])


def _checkpoint_from_result(optimization_root: Path, result_path: Path,
                            expected: dict) -> Path:
    rows = _read_csv(result_path)
    if len(rows) != 1:
        raise RuntimeError(f"selected result missing or duplicated: {result_path}")
    row = rows[0]
    for key, value in expected.items():
        if str(row.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"selected result hash mismatch: {key}")
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = optimization_root / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    expected_checkpoint_dir = (result_path.parent / "checkpoints").resolve()
    if checkpoint.resolve().parent != expected_checkpoint_dir:
        raise RuntimeError(
            f"selected checkpoint escaped its result directory: {checkpoint}"
        )
    return checkpoint


def _shard_path(spec: dict, dataset: str) -> Path:
    return Path(spec["output_root"]) / "shards" / f"{dataset}.csv"


def _validate_shard_rows(rows: list[dict], spec: dict, dataset: str,
                         family: str, selection_hash: str,
                         manifest_hash: str, require_complete: bool) -> None:
    expected_keys = {
        (source, int(seed))
        for seed in spec["paired_seeds"]
        for source in ("fmt", "raw_pca")
    }
    observed_keys = []
    for row in rows:
        key = (str(row.get("source")), int(row.get("seed", -1)))
        observed_keys.append(key)
        expected = {
            "experiment": spec["experiment"],
            "recipe_manifest_sha256": manifest_hash,
            "optimization_selection_sha256": selection_hash,
            "dataset": dataset,
            "physical_family": family,
            "method": (
                "fmt_residual" if key[0] == "fmt"
                else "raw_pca_residual"
            ),
        }
        for name, value in expected.items():
            if str(row.get(name, "")).lower() != str(value).lower():
                raise RuntimeError(
                    f"stale Task3 12.1 shard {dataset}: {name}"
                )
        if key not in expected_keys:
            raise RuntimeError(f"unexpected Task3 12.1 shard key {dataset}/{key}")
        for metric in ("f1", "average_precision"):
            if not np.isfinite(float(row.get(metric, "nan"))):
                raise RuntimeError(
                    f"non-finite Task3 12.1 {dataset}/{key} {metric}"
                )
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.exists() or _sha256(checkpoint) != row[
            "checkpoint_sha256"
        ]:
            raise RuntimeError(
                f"Task3 12.1 checkpoint changed for {dataset}/{key}"
            )
    if len(observed_keys) != len(set(observed_keys)):
        raise RuntimeError(f"duplicate Task3 12.1 shard key for {dataset}")
    observed = set(observed_keys)
    if not observed.issubset(expected_keys):
        raise RuntimeError(f"unexpected Task3 12.1 shard rows for {dataset}")
    if require_complete and observed != expected_keys:
        missing = sorted(expected_keys - observed)
        raise RuntimeError(
            f"incomplete Task3 12.1 shard {dataset}; missing {missing}"
        )


def run_dataset(config_path: str | Path, dataset: str) -> Path:
    spec = _load_spec(config_path)
    (optimization_root, optimization, preflight_path, selection_path,
     selection, manifest_path, manifest) = _frozen_state(spec)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown Task3 12.1 dataset {dataset!r}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    group_name, candidate = _selected_candidate(
        optimization, preflight, selection, dataset
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confirmation_records = _load_confirmation(
        spec, optimization, dataset, candidate, device
    )
    confirmation = _stack_split(
        confirmation_records, list(range(int(spec["confirmation_count"])))
    )
    target = _shard_path(spec, dataset)
    rows = _read_csv(target)
    selection_hash = _sha256(selection_path)
    manifest_hash = _sha256(manifest_path)
    _validate_shard_rows(
        rows, spec, dataset, group_name, selection_hash, manifest_hash,
        require_complete=False,
    )
    completed = {(row["source"], int(row["seed"])) for row in rows}
    for seed in spec["paired_seeds"]:
        for source in ("fmt", "raw_pca"):
            if (source, int(seed)) in completed:
                continue
            result_path = _result_path(
                optimization, candidate, dataset, int(seed), source
            )
            checkpoint_path = _checkpoint_from_result(
                optimization_root,
                result_path,
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "variant": (
                        "raw_fmt_residual" if source == "fmt"
                        else "raw_pca_residual"
                    ),
                    "auxiliary_source": source,
                    "optimization_id": candidate["optimization_id"],
                    "optimization_recipe_json": json.dumps(
                        candidate["optimization_recipe"], sort_keys=True
                    ),
                    "fmt_feature": candidate["fmt_feature"],
                    "optimization_config_sha256": optimization[
                        "optimization_config_sha256"
                    ],
                    "preflight_manifest_sha256": _sha256(preflight_path),
                    "upstream_selection_sha256": preflight[
                        "upstream_selection_sha256"
                    ],
                },
            )
            model, checkpoint = _load_residual(
                checkpoint_path, confirmation[1].shape[1], device,
                checkpoint_root=optimization_root,
            )
            targets, _, metrics = _evaluate_residual(
                model, checkpoint, confirmation,
                int(optimization["training"]["batch_size"]), int(seed), device,
            )
            rows.append({
                "experiment": spec["experiment"],
                "recipe_manifest_sha256": manifest_hash,
                "optimization_selection_sha256": selection_hash,
                "dataset": dataset,
                "physical_family": group_name,
                "seed": int(seed),
                "source": source,
                "method": (
                    "fmt_residual" if source == "fmt"
                    else "raw_pca_residual"
                ),
                "optimization_id": candidate["optimization_id"],
                "sample_count": len(targets),
                "positive_fraction": float(targets.mean()),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                **metrics,
            })
            _write_csv(target, rows)
            completed.add((source, int(seed)))
        print(f"Task3 12.1 {dataset} seed={seed} complete", flush=True)
    return target


def _aggregate(rows: list[dict], datasets: list[str]) -> dict:
    dataset_summary = {}
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        if not selected:
            raise RuntimeError(f"missing Task3 12.1 dataset {dataset}")
        methods = {}
        for source in ("raw_pca", "fmt"):
            values = [row for row in selected if row["source"] == source]
            if not values:
                raise RuntimeError(
                    f"missing Task3 12.1 {dataset}/{source} arm"
                )
            methods[source] = {
                metric: float(np.mean([float(row[metric]) for row in values]))
                for metric in ("f1", "average_precision")
            }
        dataset_summary[dataset] = {
            "physical_family": selected[0]["physical_family"],
            "raw_pca_residual": methods["raw_pca"],
            "fmt_residual": methods["fmt"],
            "f1_gain": methods["fmt"]["f1"] - methods["raw_pca"]["f1"],
            "average_precision_gain": (
                methods["fmt"]["average_precision"]
                - methods["raw_pca"]["average_precision"]
            ),
        }
    families = sorted({
        row["physical_family"] for row in dataset_summary.values()
    })
    family_summary = {}
    for family in families:
        values = [
            row for row in dataset_summary.values()
            if row["physical_family"] == family
        ]
        family_summary[family] = {
            "datasets": sorted(
                dataset for dataset, row in dataset_summary.items()
                if row["physical_family"] == family
            ),
            "f1_gain": float(np.mean([row["f1_gain"] for row in values])),
            "average_precision_gain": float(np.mean([
                row["average_precision_gain"] for row in values
            ])),
        }
    return {
        "datasets": dataset_summary,
        "families": family_summary,
        "dataset_macro_f1_gain_vs_raw_pca": float(np.mean([
            row["f1_gain"] for row in dataset_summary.values()
        ])),
        "dataset_macro_ap_gain_vs_raw_pca": float(np.mean([
            row["average_precision_gain"]
            for row in dataset_summary.values()
        ])),
        "family_macro_f1_gain_vs_raw_pca": float(np.mean([
            row["f1_gain"] for row in family_summary.values()
        ])),
        "family_macro_ap_gain_vs_raw_pca": float(np.mean([
            row["average_precision_gain"]
            for row in family_summary.values()
        ])),
        "positive_dataset_f1_gain_count": int(np.count_nonzero([
            row["f1_gain"] > 0 for row in dataset_summary.values()
        ])),
        "positive_family_f1_gain_count": int(np.count_nonzero([
            row["f1_gain"] > 0 for row in family_summary.values()
        ])),
        "minimum_dataset_f1_gain": float(min(
            row["f1_gain"] for row in dataset_summary.values()
        )),
    }


def summarize(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    (_, optimization, _, selection_path, _, manifest_path,
     manifest) = _frozen_state(spec)
    rows = []
    selection_hash = _sha256(selection_path)
    manifest_hash = _sha256(manifest_path)
    for dataset in spec["datasets"]:
        values = _read_csv(_shard_path(spec, dataset))
        family, _ = _group_for_dataset(optimization, dataset)
        _validate_shard_rows(
            values, spec, dataset, family, selection_hash, manifest_hash,
            require_complete=True,
        )
        rows.extend(values)
    output = Path(spec["output_root"])
    _write_csv(output / "per_run.csv", rows)
    aggregate = _aggregate(rows, spec["datasets"])
    result = {
        "experiment": spec["experiment"],
        "comparison": (
            "FMT residual minus same-width same-structure train-only "
            "Raw-PCA residual"
        ),
        "optimization_selection_sha256": selection_hash,
        "recipe_manifest_sha256": manifest_hash,
        "confirmation_seed_grid_phase": manifest[
            "confirmation_seed_grid_phase"
        ],
        "confirmation_data_was_not_used_for_selection": True,
        "paired_seeds": spec["paired_seeds"],
        **aggregate,
    }
    primary_target = float(spec["target_dataset_macro_f1_gain"])
    aspirational_target = float(spec["aspirational_dataset_macro_f1_gain"])
    result.update({
        "target_dataset_macro_f1_gain": primary_target,
        "target_reached": (
            result["dataset_macro_f1_gain_vs_raw_pca"] >= primary_target
        ),
        "aspirational_dataset_macro_f1_gain": aspirational_target,
        "aspirational_target_reached": (
            result["dataset_macro_f1_gain_vs_raw_pca"] >= aspirational_target
        ),
    })
    target = output / "summary.json"
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=("static-preflight", "freeze", "source-preflight", "cache", "labels", "dataset", "summary"),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "static-preflight":
        static_preflight(args.config)
    elif args.mode == "freeze":
        freeze(args.config)
    elif args.mode == "source-preflight":
        source_preflight(args.config)
    elif args.mode == "cache":
        if args.job_index is None:
            parser.error("cache mode requires --job-index")
        build_cache(args.config, args.job_index, args.overwrite)
    elif args.mode == "labels":
        if args.job_index is None:
            parser.error("labels mode requires --job-index")
        build_labels(args.config, args.job_index, args.overwrite)
    elif args.mode == "dataset":
        if args.dataset is None:
            parser.error("dataset mode requires --dataset")
        run_dataset(args.config, args.dataset)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
