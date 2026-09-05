"""Sealed final Task3 confirmation on a fourth spatial population.

The frozen 22.1 family-specific FMT and Raw-PCA residual models are recorded
before any 6.1 primitive or IVD label is generated.  No training, threshold
selection, residual-scale selection, or feature selection occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import Build_Task3_AnchoredFeature_Confirmation_6_1 as spatial
import Replay_Task3_AnchoredFeatureSpatial_46_1 as replay
from Confirm_Task3_CombinedOptimization_12_1 import _aggregate
from Evaluate_Task3_FrozenConfirmation import _evaluate_residual, _load_residual
from Run_Task3_FMTResidual_Frozen_4_1 import _load_confirmation
from Search_Task3_FMTResidual_3D import _read_csv, _write_csv
from Verify_Task3_FMTClassifier import _stack_split


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "task", "status", "output_root", "recipe_manifest",
        "source_model", "source_staging", "datasets", "paired_seeds",
        "confirmation_count", "expected_ivd_percentile",
        "require_confirmation_reference_match", "batch_size",
        "phase_key", "phase_key_sha256", "halton_index",
        "confirmation_seed_grid_phase", "confirmation_roots",
        "target_dataset_macro_f1_gain",
        "aspirational_dataset_macro_f1_gain",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing Task3 6.1 config keys: {missing}")
    if spec["experiment"] != spatial.EXPERIMENT or spec["task"] != "Task3":
        raise ValueError("Task3 6.1 experiment identity changed")
    if spec["status"] != "fresh_spatial_confirmation":
        raise ValueError("Task3 6.1 must remain a fresh confirmation")
    datasets = [str(value) for value in spec["datasets"]]
    seeds = [int(value) for value in spec["paired_seeds"]]
    if len(datasets) != 10 or len(set(datasets)) != 10:
        raise ValueError("Task3 6.1 requires ten unique datasets")
    if seeds != [40, 41]:
        raise ValueError("Task3 6.1 freezes the two 22.1 seeds")
    if int(spec["confirmation_count"]) != 4:
        raise ValueError("Task3 6.1 requires four spatial slices")
    if not np.isclose(float(spec["expected_ivd_percentile"]), 95.0):
        raise ValueError("Task3 6.1 requires whole-field IVD-p95")
    if not bool(spec["require_confirmation_reference_match"]):
        raise ValueError("Task3 6.1 requires source/reference identity")
    if str(spec["phase_key"]) != spatial.PHASE_KEY:
        raise ValueError("Task3 6.1 phase key changed")
    if str(spec["phase_key_sha256"]) != spatial.PHASE_KEY_SHA256:
        raise ValueError("Task3 6.1 phase-key SHA-256 changed")
    if int(spec["halton_index"]) != spatial.HALTON_INDEX:
        raise ValueError("Task3 6.1 Halton index changed")
    if [float(value) for value in spec["confirmation_seed_grid_phase"]] != list(
        spatial.SEED_GRID_PHASE
    ):
        raise ValueError("Task3 6.1 spatial phase changed")
    if float(spec["target_dataset_macro_f1_gain"]) != 0.15:
        raise ValueError("Task3 6.1 primary target changed")
    if not bool(spec.get("new_spatial_primitive_population", False)):
        raise ValueError("Task3 6.1 must declare a new primitive population")
    if bool(spec.get("confirmation_opened_before_freeze", True)):
        raise ValueError("Task3 6.1 cannot be open before freeze")

    roots = dict(spec["confirmation_roots"])
    if set(roots) != set(spatial.SETTINGS):
        raise ValueError("Task3 6.1 root groups changed")
    grouped = []
    for name, group in roots.items():
        expected = list(spatial.SETTINGS[name]["indices"])
        observed = [str(value) for value in group.get("datasets", [])]
        if observed != expected:
            raise ValueError(f"Task3 6.1 {name} dataset order changed")
        if Path(group["source_root"]) != Path(spatial.SETTINGS[name]["cache_dir"]):
            raise ValueError(f"Task3 6.1 {name} source root changed")
        label_root = Path(spatial.SETTINGS[name]["label_config"])
        label_spec = yaml.safe_load(label_root.read_text(encoding="utf-8"))
        if Path(group["label_root"]) != Path(label_spec["output_dir"]) / "labels":
            raise ValueError(f"Task3 6.1 {name} label root changed")
        grouped.extend(observed)
    if set(grouped) != set(datasets) or len(grouped) != len(set(grouped)):
        raise ValueError("Task3 6.1 roots do not partition datasets")
    for key, value in spec["source_model"]["sha256"].items():
        if len(str(value)) != 64:
            raise ValueError(f"source_model.sha256.{key} is incomplete")
    if len(str(spec["source_staging"]["parent_manifest_sha256"])) != 64:
        raise ValueError("source-staging parent SHA-256 is incomplete")
    spec["datasets"] = datasets
    spec["paired_seeds"] = seeds
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    return spec


def _source_state(spec: dict) -> tuple:
    return replay._source_state(spec)


def _collect_models(spec: dict, source_root: Path, source: dict,
                    selection: dict) -> list[dict]:
    models = []
    for dataset in spec["datasets"]:
        family, candidate = replay._selected_candidate(source, selection, dataset)
        for seed in spec["paired_seeds"]:
            paired = []
            for arm, variant in (
                ("fmt", "raw_fmt_residual"),
                ("raw_pca", "raw_pca_residual"),
            ):
                result_path = replay._source_result_path(
                    source_root, source, candidate, dataset, seed, arm
                )
                checkpoint, row = replay._checkpoint_from_result(
                    source_root,
                    result_path,
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "auxiliary_source": arm,
                        "candidate_id": candidate["id"],
                        "fmt_feature": candidate["fmt_feature"],
                    },
                )
                item = {
                    "dataset": dataset,
                    "physical_family": family,
                    "seed": int(seed),
                    "source": arm,
                    "variant": variant,
                    "candidate_id": str(candidate["id"]),
                    "fmt_feature": str(candidate["fmt_feature"]),
                    "fmt_dim": int(row["fmt_dim"]),
                    "parameter_count": int(row["parameter_count"]),
                    "trainable_residual_parameter_count": int(
                        row["trainable_residual_parameter_count"]
                    ),
                    "result": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
                models.append(item)
                paired.append(item)
            for key in (
                "fmt_dim", "parameter_count",
                "trainable_residual_parameter_count",
            ):
                if paired[0][key] != paired[1][key]:
                    raise RuntimeError(
                        f"Task3 6.1 paired {key} mismatch: {dataset}/seed{seed}"
                    )
    if len(models) != 40:
        raise RuntimeError("Task3 6.1 must freeze exactly 40 models")
    return models


def _artifact_counts(spec: dict) -> dict:
    counts = {}
    for name, group in spec["confirmation_roots"].items():
        source_root = Path(group["source_root"])
        label_root = Path(group["label_root"])
        counts[name] = {
            "source_npz": (
                sum(1 for _ in source_root.rglob("*.npz"))
                if source_root.exists() else 0
            ),
            "label_npz": (
                sum(1 for _ in label_root.rglob("*.npz"))
                if label_root.exists() else 0
            ),
        }
    return counts


def static_preflight(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    source_root, source_paths, source, _, selection = _source_state(spec)
    models = _collect_models(spec, source_root, source, selection)
    staging = spatial.source_staging_identity()
    counts = _artifact_counts(spec)
    if any(value for group in counts.values() for value in group.values()):
        raise RuntimeError("Task3 6.1 confirmation existed before static preflight")
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "source_model_artifact_sha256": {
            key: _sha256(path) for key, path in source_paths.items()
        },
        "source_model_selection_sha256": spec["source_model"]["sha256"][
            "selection"
        ],
        "source_staging": staging,
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "confirmation_artifact_counts": counts,
        "frozen_model_count": len(models),
        "expected_evaluations": len(models),
        "confirmation_opened": False,
    }
    target = Path(spec["output_root"]) / "static_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Task3 6.1 static preflight changed")
    else:
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(target)
    return target


def freeze(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    static_path = Path(spec["output_root"]) / "static_preflight.json"
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if static.get("config_sha256") != spec["config_sha256"]:
        raise RuntimeError("Task3 6.1 static preflight/config mismatch")
    if bool(static.get("confirmation_opened", True)):
        raise RuntimeError("Task3 6.1 static preflight opened confirmation")
    counts = _artifact_counts(spec)
    if any(value for group in counts.values() for value in group.values()):
        raise RuntimeError("Task3 6.1 confirmation appeared before freeze")
    source_root, source_paths, source, _, selection = _source_state(spec)
    models = _collect_models(spec, source_root, source, selection)
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": spec["config_sha256"],
        "static_preflight_sha256": _sha256(static_path),
        "source_model_repo_root": str(source_root),
        "source_model_artifact_sha256": {
            key: _sha256(path) for key, path in source_paths.items()
        },
        "source_model_selection_sha256": spec["source_model"]["sha256"][
            "selection"
        ],
        "source_staging": spatial.source_staging_identity(),
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "phase_key": spatial.PHASE_KEY,
        "phase_key_sha256": spatial.PHASE_KEY_SHA256,
        "halton_index": spatial.HALTON_INDEX,
        "same_physical_times_as_prior_spatial_checks": True,
        "new_spatial_primitive_population": True,
        "confirmation_artifact_counts_at_freeze": counts,
        "confirmation_data_opened": False,
        "models": models,
    }
    target = Path(spec["recipe_manifest"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Task3 6.1 frozen recipe changed")
    else:
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(target)
    return target


def _frozen_state(spec: dict) -> tuple[Path, dict, Path, dict, dict]:
    manifest_path = Path(spec["recipe_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": spec["config_sha256"],
        "source_model_selection_sha256": spec["source_model"]["sha256"][
            "selection"
        ],
        "confirmation_data_opened": False,
    }
    for key, value in expected.items():
        if str(manifest.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"Task3 6.1 frozen recipe changed: {key}")
    if list(manifest.get("confirmation_seed_grid_phase", [])) != list(
        spatial.SEED_GRID_PHASE
    ):
        raise RuntimeError("Task3 6.1 frozen phase changed")
    source_root, _, source, _, selection = _source_state(spec)
    if manifest.get("source_staging") != spatial.source_staging_identity():
        raise RuntimeError("Task3 6.1 source staging changed after freeze")
    current_models = _collect_models(spec, source_root, source, selection)
    if current_models != manifest.get("models"):
        raise RuntimeError("Task3 6.1 model set changed after freeze")
    return manifest_path, manifest, source_root, source, selection


def source_preflight(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    _frozen_state(spec)
    report = spatial.source_preflight()
    target = Path(spec["output_root"]) / "source_preflight.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != report:
            raise RuntimeError("Task3 6.1 source preflight changed")
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
    return spatial.build_job(job_index, overwrite)


def build_labels(config_path: str | Path, group_index: int,
                 overwrite: bool = False) -> Path:
    spec = _load_spec(config_path)
    _frozen_state(spec)
    groups = ("old8", "new2")
    if not 0 <= int(group_index) < len(groups):
        raise IndexError("Task3 6.1 label group outside [0,2)")
    return spatial.build_labels(groups[int(group_index)], overwrite)


def evaluation_preflight(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    manifest_path, manifest, _, _, _ = _frozen_state(spec)
    counts = {}
    for dataset in spec["datasets"]:
        matches = [
            group for group in spec["confirmation_roots"].values()
            if dataset in group["datasets"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Task3 6.1 root lookup failed: {dataset}")
        group = matches[0]
        source_names = sorted(
            path.name for path in (Path(group["source_root"]) / dataset).glob("*.npz")
        )
        label_names = sorted(
            path.name for path in (Path(group["label_root"]) / dataset).glob("*.npz")
        )
        if len(source_names) != int(spec["confirmation_count"]):
            raise RuntimeError(f"Task3 6.1 source count changed: {dataset}")
        if label_names != source_names:
            raise RuntimeError(f"Task3 6.1 label/source names differ: {dataset}")
        counts[dataset] = {
            "source_npz": len(source_names),
            "label_npz": len(label_names),
            "filenames": source_names,
        }
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": spec["config_sha256"],
        "recipe_manifest_sha256": _sha256(manifest_path),
        "source_model_selection_sha256": manifest[
            "source_model_selection_sha256"
        ],
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "confirmation_was_generated_after_recipe_freeze": True,
        "confirmation_artifact_counts": counts,
        "models": manifest["models"],
        "expected_evaluations": len(manifest["models"]),
    }
    target = Path(spec["output_root"]) / "evaluation_preflight.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Task3 6.1 evaluation preflight changed")
    else:
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(target)
    return target


def _evaluation_state(spec: dict) -> tuple[Path, dict]:
    path = Path(spec["output_root"]) / "evaluation_preflight.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path, manifest, _, _, _ = _frozen_state(spec)
    expected = {
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": spec["config_sha256"],
        "recipe_manifest_sha256": _sha256(manifest_path),
        "source_model_selection_sha256": manifest[
            "source_model_selection_sha256"
        ],
        "confirmation_was_generated_after_recipe_freeze": True,
        "expected_evaluations": 40,
    }
    for key, value in expected.items():
        if str(payload.get(key, "")).lower() != str(value).lower():
            raise RuntimeError(f"Task3 6.1 evaluation preflight changed: {key}")
    return path, payload


def _model_entry(manifest: dict, dataset: str, seed: int,
                 source: str) -> dict:
    matches = [
        row for row in manifest["models"]
        if row["dataset"] == dataset
        and int(row["seed"]) == int(seed)
        and row["source"] == source
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Task3 6.1 model missing: {dataset}/{seed}/{source}")
    return matches[0]


def _shard_path(spec: dict, dataset: str) -> Path:
    return Path(spec["output_root"]) / "shards" / f"{dataset}.csv"


def _validate_rows(rows: list[dict], spec: dict, manifest: dict,
                   manifest_hash: str, eval_hash: str, dataset: str,
                   require_complete: bool) -> None:
    expected_keys = {
        (source, seed)
        for seed in spec["paired_seeds"]
        for source in ("fmt", "raw_pca")
    }
    observed = []
    for row in rows:
        key = (str(row.get("source")), int(row.get("seed", -1)))
        observed.append(key)
        if key not in expected_keys:
            raise RuntimeError(f"unexpected Task3 6.1 row: {dataset}/{key}")
        model = _model_entry(manifest, dataset, key[1], key[0])
        expected = {
            "experiment": spec["experiment"],
            "status": spec["status"],
            "config_sha256": spec["config_sha256"],
            "recipe_manifest_sha256": manifest_hash,
            "evaluation_preflight_sha256": eval_hash,
            "dataset": dataset,
            "physical_family": model["physical_family"],
            "candidate_id": model["candidate_id"],
            "fmt_feature": model["fmt_feature"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "method": "fmt_residual" if key[0] == "fmt" else "raw_pca_residual",
        }
        for name, value in expected.items():
            if str(row.get(name, "")).lower() != str(value).lower():
                raise RuntimeError(f"stale Task3 6.1 row: {dataset}/{name}")
        for metric in ("f1", "average_precision"):
            if not np.isfinite(float(row.get(metric, "nan"))):
                raise RuntimeError(f"non-finite Task3 6.1 metric: {dataset}")
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"duplicate Task3 6.1 rows: {dataset}")
    if require_complete and set(observed) != expected_keys:
        raise RuntimeError(f"incomplete Task3 6.1 rows: {dataset}")
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed, paired in by_seed.items():
        if len(paired) == 2:
            counts = {int(row["sample_count"]) for row in paired}
            fractions = {round(float(row["positive_fraction"]), 12) for row in paired}
            if len(counts) != 1 or len(fractions) != 1:
                raise RuntimeError(f"Task3 6.1 paired targets differ: {dataset}/{seed}")


def run_dataset(config_path: str | Path, dataset: str) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown Task3 6.1 dataset: {dataset}")
    manifest_path, manifest, source_root, source, selection = _frozen_state(spec)
    eval_path, _ = _evaluation_state(spec)
    manifest_hash = _sha256(manifest_path)
    eval_hash = _sha256(eval_path)
    family, candidate = replay._selected_candidate(source, selection, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = _load_confirmation(spec, source, dataset, candidate, device)
    confirmation = _stack_split(
        records, list(range(int(spec["confirmation_count"])))
    )
    target = _shard_path(spec, dataset)
    rows = _read_csv(target)
    _validate_rows(
        rows, spec, manifest, manifest_hash, eval_hash, dataset, False
    )
    completed = {(row["source"], int(row["seed"])) for row in rows}
    for seed in spec["paired_seeds"]:
        for arm in ("fmt", "raw_pca"):
            if (arm, seed) in completed:
                continue
            model_entry = _model_entry(manifest, dataset, seed, arm)
            checkpoint_path = Path(model_entry["checkpoint"])
            if _sha256(checkpoint_path) != model_entry["checkpoint_sha256"]:
                raise RuntimeError(f"Task3 6.1 checkpoint changed: {dataset}/{seed}/{arm}")
            model, checkpoint = _load_residual(
                checkpoint_path,
                confirmation[1].shape[1],
                device,
                checkpoint_root=source_root,
            )
            targets, _, metrics = _evaluate_residual(
                model, checkpoint, confirmation, int(spec["batch_size"]), seed, device
            )
            rows.append({
                "experiment": spec["experiment"],
                "status": spec["status"],
                "config_sha256": spec["config_sha256"],
                "recipe_manifest_sha256": manifest_hash,
                "evaluation_preflight_sha256": eval_hash,
                "dataset": dataset,
                "physical_family": family,
                "candidate_id": candidate["id"],
                "fmt_feature": candidate["fmt_feature"],
                "seed": int(seed),
                "source": arm,
                "method": "fmt_residual" if arm == "fmt" else "raw_pca_residual",
                "sample_count": int(len(targets)),
                "positive_fraction": float(targets.mean()),
                "frozen_threshold": float(checkpoint["threshold"]),
                "frozen_alpha": float(checkpoint["alpha"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": model_entry["checkpoint_sha256"],
                **metrics,
            })
            _write_csv(target, rows)
            completed.add((arm, seed))
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"Task3 6.1 {dataset} seed={seed} complete", flush=True)
    return target


def summarize(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    manifest_path, manifest, _, _, selection = _frozen_state(spec)
    eval_path, _ = _evaluation_state(spec)
    manifest_hash = _sha256(manifest_path)
    eval_hash = _sha256(eval_path)
    rows = []
    for dataset in spec["datasets"]:
        shard = _read_csv(_shard_path(spec, dataset))
        _validate_rows(
            shard, spec, manifest, manifest_hash, eval_hash, dataset, True
        )
        rows.extend(shard)
    output = Path(spec["output_root"])
    _write_csv(output / "per_run.csv", rows)
    aggregate = _aggregate(rows, spec["datasets"])
    datasets = aggregate["datasets"]
    raw_f1 = float(np.mean([
        value["raw_pca_residual"]["f1"] for value in datasets.values()
    ]))
    fmt_f1 = float(np.mean([
        value["fmt_residual"]["f1"] for value in datasets.values()
    ]))
    raw_ap = float(np.mean([
        value["raw_pca_residual"]["average_precision"]
        for value in datasets.values()
    ]))
    fmt_ap = float(np.mean([
        value["fmt_residual"]["average_precision"]
        for value in datasets.values()
    ]))
    target = float(spec["target_dataset_macro_f1_gain"])
    aspirational = float(spec["aspirational_dataset_macro_f1_gain"])
    result = {
        "experiment": spec["experiment"],
        "status": spec["status"],
        "comparison": (
            "frozen 22.1 FMT residual minus its same-width, same-structure "
            "train-only Raw-PCA residual"
        ),
        "fresh_confirmation": True,
        "confirmation_data_was_not_used_for_selection": True,
        "recipe_manifest_sha256": manifest_hash,
        "evaluation_preflight_sha256": eval_hash,
        "source_model_selection_sha256": manifest[
            "source_model_selection_sha256"
        ],
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "phase_key_sha256": spatial.PHASE_KEY_SHA256,
        "halton_index": spatial.HALTON_INDEX,
        "paired_seeds": spec["paired_seeds"],
        "dataset_macro_raw_pca_f1": raw_f1,
        "dataset_macro_fmt_f1": fmt_f1,
        "dataset_macro_raw_pca_ap": raw_ap,
        "dataset_macro_fmt_ap": fmt_ap,
        "source_development_f1_gain": float(selection[
            "development_dataset_macro_f1_gain_vs_raw_pca"
        ]),
        **aggregate,
        "target_dataset_macro_f1_gain": target,
        "target_reached": aggregate[
            "dataset_macro_f1_gain_vs_raw_pca"
        ] >= target,
        "aspirational_dataset_macro_f1_gain": aspirational,
        "aspirational_target_reached": aggregate[
            "dataset_macro_f1_gain_vs_raw_pca"
        ] >= aspirational,
    }
    target_path = output / "summary.json"
    target_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target_path.read_text(encoding="utf-8"))
    return target_path


def _decode_dataset(spec: dict, index: int) -> str:
    if not 0 <= int(index) < len(spec["datasets"]):
        raise IndexError("Task3 6.1 dataset job outside [0,10)")
    return spec["datasets"][int(index)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "static-preflight", "freeze", "source-preflight", "cache",
            "labels", "evaluation-preflight", "dataset", "summary",
        ),
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
    elif args.mode == "evaluation-preflight":
        evaluation_preflight(args.config)
    elif args.mode == "dataset":
        spec = _load_spec(args.config)
        dataset = args.dataset
        if dataset is None:
            if args.job_index is None:
                parser.error("dataset mode requires --dataset or --job-index")
            dataset = _decode_dataset(spec, args.job_index)
        run_dataset(args.config, dataset)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
