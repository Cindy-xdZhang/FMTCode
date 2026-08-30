"""Frozen Task2 confirmation for the latent-dimension search 5.1.

The primary recipe is fixed by ``Verify_Task2_LatentBottleneck_5.1`` before
the fifth spatial primitive population is generated.  Raw and FMT use the
same VAE hyperparameters and paired training seed within every recipe.  The
original Task2 4.1 latent dimension is evaluated as a predeclared control on
the same population; it cannot replace the selected primary recipe.
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
from sklearn.cluster import KMeans

import Build_Task2_LatentConfirmation_5_2 as spatial
from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics,
    calibrate_vortex_cluster,
)
from Run_Task2_3D_Main import _prepare_inputs
from Search_Task2_LatentBottleneck_5_1 import _write_csv
from Verify_HighReVAE import _train


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalized_text_sha256(path: str | Path) -> str:
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "task", "status", "output_root", "recipe_manifest",
        "evaluation_preflight", "source_search", "source_staging", "datasets",
        "final_training_seeds", "splits", "confirmation_count", "recipes",
        "kmeans_seed", "kmeans_n_init", "phase_key", "phase_key_sha256",
        "halton_index", "confirmation_seed_grid_phase", "confirmation_roots",
        "target_dataset_macro_f1_gain", "aspirational_dataset_macro_f1_gain",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing Task2 5.2 config keys: {missing}")
    if spec["experiment"] != spatial.EXPERIMENT or spec["task"] != "Task2":
        raise ValueError("Task2 5.2 experiment identity changed")
    if spec["status"] != "fresh_spatial_confirmation":
        raise ValueError("Task2 5.2 must remain a fresh confirmation")
    datasets = [str(value) for value in spec["datasets"]]
    if len(datasets) != 10 or len(set(datasets)) != 10:
        raise ValueError("Task2 5.2 requires ten unique datasets")
    seeds = [int(value) for value in spec["final_training_seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Task2 5.2 requires five unique final training seeds")
    train = [int(value) for value in spec["splits"]["final_train"]]
    calibration = [
        int(value) for value in spec["splits"]["cluster_calibration"]
    ]
    if train != list(range(8)) or calibration != [8, 9]:
        raise ValueError("Task2 5.2 development split changed")
    if set(train) & set(calibration):
        raise ValueError("Task2 5.2 training and calibration overlap")
    if list(spec["recipes"]) != ["selected", "control"]:
        raise ValueError("Task2 5.2 must report selected then control")
    if int(spec["confirmation_count"]) != 4:
        raise ValueError("Task2 5.2 requires four confirmation slices")
    if str(spec["phase_key"]) != spatial.PHASE_KEY:
        raise ValueError("Task2 5.2 phase key changed")
    if str(spec["phase_key_sha256"]) != spatial.PHASE_KEY_SHA256:
        raise ValueError("Task2 5.2 phase-key SHA-256 changed")
    if int(spec["halton_index"]) != spatial.HALTON_INDEX:
        raise ValueError("Task2 5.2 Halton index changed")
    if [float(value) for value in spec["confirmation_seed_grid_phase"]] != list(
        spatial.SEED_GRID_PHASE
    ):
        raise ValueError("Task2 5.2 spatial phase changed")
    if bool(spec.get("confirmation_opened_before_freeze", True)):
        raise ValueError("Task2 5.2 confirmation cannot be open before freeze")
    if not bool(spec.get("new_spatial_primitive_population", False)):
        raise ValueError("Task2 5.2 must declare a new primitive population")
    roots = dict(spec["confirmation_roots"])
    if set(roots) != set(spatial.SETTINGS):
        raise ValueError("Task2 5.2 confirmation root groups changed")
    grouped = []
    for group_name, group in roots.items():
        observed = [str(value) for value in group["datasets"]]
        expected = list(spatial.SETTINGS[group_name]["indices"])
        if observed != expected:
            raise ValueError(f"Task2 5.2 {group_name} dataset order changed")
        if Path(group["root"]) != Path(spatial.SETTINGS[group_name]["cache_dir"]):
            raise ValueError(f"Task2 5.2 {group_name} cache root changed")
        grouped.extend(observed)
    if set(grouped) != set(datasets) or len(grouped) != len(set(grouped)):
        raise ValueError("Task2 5.2 roots do not partition datasets")
    for key in ("search_config_sha256", "selection_sha256"):
        if len(str(spec["source_search"][key])) != 64:
            raise ValueError(f"source_search.{key} is incomplete")
    for key in (
        "artifact_repo_root", "development_repo_root",
        "search_config", "selection", "expected_search_experiment",
    ):
        if key not in spec["source_search"]:
            raise ValueError(f"source_search.{key} is missing")
    spec["datasets"] = datasets
    spec["final_training_seeds"] = seeds
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    return spec


def _root_from_source(spec: dict, prefix: str) -> Path:
    source = spec["source_search"]
    env_name = str(source.get(f"{prefix}_repo_root_env", ""))
    value = os.environ.get(env_name) if env_name else None
    return Path(value or source[f"{prefix}_repo_root"]).resolve()


def _artifact_path(spec: dict, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _root_from_source(spec, "artifact") / path


def _development_path(spec: dict, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _root_from_source(
        spec, "development"
    ) / path


def _source_state(spec: dict) -> tuple[dict, dict, Path, Path]:
    source = spec["source_search"]
    search_path = _artifact_path(spec, source["search_config"])
    selection_path = _artifact_path(spec, source["selection"])
    if _normalized_text_sha256(search_path) != str(
        source["search_config_sha256"]
    ):
        raise RuntimeError("Task2 5.1 search config changed")
    if _sha256(selection_path) != str(source["selection_sha256"]):
        raise RuntimeError("Task2 5.1 selection changed")
    search = yaml.safe_load(search_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if search.get("experiment") != source["expected_search_experiment"]:
        raise RuntimeError("Task2 5.1 search experiment changed")
    if selection.get("experiment") != source["expected_search_experiment"]:
        raise RuntimeError("Task2 5.1 selection experiment changed")
    if bool(selection.get("confirmation_opened", True)):
        raise RuntimeError("Task2 5.1 selection opened confirmation data")
    if bool(selection.get("forbidden_ordinals_opened", True)):
        raise RuntimeError("Task2 5.1 selection opened ordinals 8--9")
    groups = set(search["groups"])
    if set(selection["winner_by_group"]) != groups:
        raise RuntimeError("Task2 5.1 winner groups changed")
    if set(selection["control_by_group"]) != groups:
        raise RuntimeError("Task2 5.1 control groups changed")
    return search, selection, search_path, selection_path


def _group_for_dataset(search: dict, dataset: str) -> tuple[str, dict]:
    matches = [
        (name, group) for name, group in search["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} matched {len(matches)} groups")
    return matches[0]


def _recipe_settings(search: dict, selection: dict, group_name: str,
                     recipe: str) -> dict:
    group = search["groups"][group_name]
    selected = selection[
        "winner_by_group" if recipe == "selected" else "control_by_group"
    ][group_name]
    latent_dim = int(selected["latent_dim"])
    settings = dict(group["base_vae"])
    control_latent = int(settings.pop("control_latent_dim"))
    settings["hidden_dims"] = [int(value) for value in settings["hidden_dims"]]
    settings["latent_dim"] = latent_dim
    settings["id"] = f"{recipe}_latent_{latent_dim:03d}"
    if recipe == "control" and latent_dim != control_latent:
        raise RuntimeError(f"{group_name}: frozen control latent changed")
    if str(selected["fmt_feature"]) != str(group["fmt_feature"]):
        raise RuntimeError(f"{group_name}: frozen FMT feature changed")
    return settings


def _confirmation_root(spec: dict, dataset: str) -> Path:
    for group in spec["confirmation_roots"].values():
        if dataset in group["datasets"]:
            return Path(group["root"]) / dataset
    raise ValueError(f"no Task2 5.2 confirmation root for {dataset}")


def _artifact_counts(spec: dict) -> dict:
    return {
        group_name: (
            sum(1 for _ in Path(group["root"]).rglob("*.npz"))
            if Path(group["root"]).exists() else 0
        )
        for group_name, group in spec["confirmation_roots"].items()
    }


def static_preflight(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    search, selection, search_path, selection_path = _source_state(spec)
    if _artifact_counts(spec) != {"old8": 0, "new2": 0}:
        raise RuntimeError("Task2 5.2 confirmation existed before static preflight")
    winners = {}
    controls = {}
    datasets = []
    for group_name, group in search["groups"].items():
        winners[group_name] = _recipe_settings(
            search, selection, group_name, "selected"
        )
        controls[group_name] = _recipe_settings(
            search, selection, group_name, "control"
        )
        for dataset in group["datasets"]:
            cache = _development_path(spec, group["development_cache"]) / dataset
            paths = sorted(cache.glob("slice_*.npz"))
            if len(paths) != 10:
                raise RuntimeError(f"{dataset}: expected 10 development slices")
            datasets.append(dataset)
    if set(datasets) != set(spec["datasets"]):
        raise RuntimeError("Task2 5.2 source dataset set changed")
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "search_config": str(search_path),
        "search_config_sha256": _normalized_text_sha256(search_path),
        "search_config_raw_sha256": _sha256(search_path),
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "selection_result_sha256": selection["result_sha256"],
        "selected_recipes": winners,
        "control_recipes": controls,
        "final_training_seeds": spec["final_training_seeds"],
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "confirmation_artifact_counts": _artifact_counts(spec),
        "opened_development_ordinals": [],
        "confirmation_opened": False,
        "checkpoint_policy": "no checkpoints are written",
    }
    target = Path(spec["output_root"]) / "static_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Task2 5.2 static preflight changed")
    else:
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target)
    return target


def freeze(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    static_path = Path(spec["output_root"]) / "static_preflight.json"
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if static.get("config_sha256") != spec["config_sha256"]:
        raise RuntimeError("Task2 5.2 static preflight/config mismatch")
    if bool(static.get("confirmation_opened", True)):
        raise RuntimeError("Task2 5.2 static preflight opened confirmation")
    if _artifact_counts(spec) != {"old8": 0, "new2": 0}:
        raise RuntimeError("Task2 5.2 confirmation appeared before freeze")
    _, selection, _, selection_path = _source_state(spec)
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "status": spec["status"],
        "config_sha256": spec["config_sha256"],
        "static_preflight_sha256": _sha256(static_path),
        "source_selection_sha256": _sha256(selection_path),
        "source_selection_result_sha256": selection["result_sha256"],
        "source_staging": spatial.source_staging_identity(),
        "selected_recipes": static["selected_recipes"],
        "control_recipes": static["control_recipes"],
        "recipes": list(spec["recipes"]),
        "final_training_seeds": spec["final_training_seeds"],
        "final_train_ordinals": list(spec["splits"]["final_train"]),
        "cluster_calibration_ordinals": list(
            spec["splits"]["cluster_calibration"]
        ),
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "phase_key": spatial.PHASE_KEY,
        "phase_key_sha256": spatial.PHASE_KEY_SHA256,
        "halton_index": spatial.HALTON_INDEX,
        "target_dataset_macro_f1_gain": float(
            spec["target_dataset_macro_f1_gain"]
        ),
        "primary_recipe": "selected",
        "control_is_diagnostic_only": True,
        "confirmation_data_opened": False,
        "checkpoint_policy": "no checkpoints are written",
    }
    target = Path(spec["recipe_manifest"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Task2 5.2 frozen recipe changed")
    else:
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target)
    return target


def _require_recipe(spec: dict) -> tuple[Path, dict]:
    path = Path(spec["recipe_manifest"])
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != spec["experiment"]:
        raise RuntimeError("Task2 5.2 recipe experiment changed")
    if payload.get("config_sha256") != spec["config_sha256"]:
        raise RuntimeError("Task2 5.2 recipe/config mismatch")
    if payload.get("source_selection_sha256") != str(
        spec["source_search"]["selection_sha256"]
    ):
        raise RuntimeError("Task2 5.2 selection hash changed after freeze")
    if list(payload.get("confirmation_seed_grid_phase", [])) != list(
        spatial.SEED_GRID_PHASE
    ):
        raise RuntimeError("Task2 5.2 recipe phase changed")
    return path, payload


def evaluation_preflight(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    recipe_path, recipe = _require_recipe(spec)
    search, selection, _, selection_path = _source_state(spec)
    if recipe["source_selection_sha256"] != _sha256(selection_path):
        raise RuntimeError("Task2 5.2 source selection changed before evaluation")
    checked = {}
    for dataset in spec["datasets"]:
        records = load_cache_records(
            _confirmation_root(spec, dataset),
            expected_count=int(spec["confirmation_count"]),
        )
        phases = {
            tuple(record["metadata"].get("seed_grid_phase", []))
            for record in records
        }
        if phases != {tuple(spatial.SEED_GRID_PHASE)}:
            raise RuntimeError(f"{dataset}: confirmation phase mismatch")
        if any(not np.isfinite(record["raw"]).all() for record in records):
            raise ValueError(f"{dataset}: non-finite confirmation raw features")
        if any(not np.isfinite(record["fmt"]).all() for record in records):
            raise ValueError(f"{dataset}: non-finite confirmation FMT features")
        positives = [float(record["reference"].mean()) for record in records]
        if any(not 0.0 < value < 1.0 for value in positives):
            raise RuntimeError(f"{dataset}: degenerate confirmation IVD-p95 labels")
        group_name, _ = _group_for_dataset(search, dataset)
        checked[dataset] = {
            "group": group_name,
            "slice_count": len(records),
            "positive_fraction": positives,
            "selected_latent_dim": int(
                selection["winner_by_group"][group_name]["latent_dim"]
            ),
            "control_latent_dim": int(
                selection["control_by_group"][group_name]["latent_dim"]
            ),
        }
    payload = {
        "schema": 1,
        "experiment": spec["experiment"],
        "config_sha256": spec["config_sha256"],
        "recipe_manifest_sha256": _sha256(recipe_path),
        "source_selection_sha256": _sha256(selection_path),
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "datasets": checked,
        "expected_trainings": (
            len(spec["datasets"]) * len(spec["recipes"])
            * 2 * len(spec["final_training_seeds"])
        ),
        "confirmation_opened_after_freeze": True,
    }
    target = Path(spec["evaluation_preflight"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target)
    return target


def _score(train_mu, calibration_mu, confirmation_mu,
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
        confirmation_reference,
        model.predict(confirmation_mu),
        vortex_cluster,
    )
    return vortex_cluster, calibration, confirmation


def run_dataset(config_path: str | Path, dataset: str) -> Path:
    spec = _load_spec(config_path)
    if dataset not in spec["datasets"]:
        raise ValueError(f"unknown Task2 5.2 dataset {dataset!r}")
    recipe_path, recipe_manifest = _require_recipe(spec)
    preflight_path = Path(spec["evaluation_preflight"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("recipe_manifest_sha256") != _sha256(recipe_path):
        raise RuntimeError("Task2 5.2 evaluation preflight/recipe mismatch")
    search, selection, _, selection_path = _source_state(spec)
    group_name, group = _group_for_dataset(search, dataset)

    ordinals = [
        *spec["splits"]["final_train"],
        *spec["splits"]["cluster_calibration"],
    ]
    development = load_cache_records(
        _development_path(spec, group["development_cache"]) / dataset,
        expected_count=10,
        ordinals=ordinals,
    )
    by_ordinal = {int(record["ordinal"]): record for record in development}
    train_records = [
        by_ordinal[int(value)] for value in spec["splits"]["final_train"]
    ]
    calibration_records = [
        by_ordinal[int(value)]
        for value in spec["splits"]["cluster_calibration"]
    ]
    confirmation_records = load_cache_records(
        _confirmation_root(spec, dataset),
        expected_count=int(spec["confirmation_count"]),
    )
    evaluation_records = [*calibration_records, *confirmation_records]
    calibration_reference = stack_reference(calibration_records)
    confirmation_reference = stack_reference(confirmation_records)
    calibration_count = len(calibration_reference)
    source = EasyConfig(str(_development_path(spec, group["source_config"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = Path(spec["output_root"]) / "shards" / f"{dataset}.csv"
    rows = _read_csv(target)
    selection_hash = _sha256(selection_path)
    recipe_hash = _sha256(recipe_path)
    if rows and {row["selection_sha256"] for row in rows} != {selection_hash}:
        raise RuntimeError(f"{dataset}: stale Task2 5.2 selection")
    if rows and {row["recipe_manifest_sha256"] for row in rows} != {recipe_hash}:
        raise RuntimeError(f"{dataset}: stale Task2 5.2 recipe")
    completed = {
        (row["recipe"], row["arm"], int(row["training_seed"]))
        for row in rows
    }

    prepared = {}
    for arm in ("raw", "fmt"):
        prepared[arm] = _prepare_inputs(
            train_records,
            evaluation_records,
            arm,
            str(group["fmt_feature"]),
            device,
        )
    for recipe in spec["recipes"]:
        settings = _recipe_settings(search, selection, group_name, recipe)
        for arm in ("raw", "fmt"):
            train_x, evaluation_x = prepared[arm]
            for seed in spec["final_training_seeds"]:
                key = (recipe, arm, int(seed))
                if key in completed:
                    continue
                train_mu, evaluation_mu, losses = _train(
                    train_x, evaluation_x, settings, source, int(seed), device
                )
                vortex_cluster, calibration_metrics, confirmation_metrics = _score(
                    train_mu,
                    evaluation_mu[:calibration_count],
                    evaluation_mu[calibration_count:],
                    calibration_reference,
                    confirmation_reference,
                    spec,
                )
                row = {
                    "experiment": spec["experiment"],
                    "selection_sha256": selection_hash,
                    "recipe_manifest_sha256": recipe_hash,
                    "evaluation_preflight_sha256": _sha256(preflight_path),
                    "dataset": dataset,
                    "group": group_name,
                    "recipe": recipe,
                    "arm": arm,
                    "fmt_feature": str(group["fmt_feature"]),
                    "latent_dim": int(settings["latent_dim"]),
                    "hidden_dims": "x".join(
                        str(value) for value in settings["hidden_dims"]
                    ),
                    "training_seed": int(seed),
                    "input_dim": int(train_x.shape[1]),
                    "cluster_as_vortex": int(vortex_cluster),
                    "confirmation_seed_grid_phase": json.dumps(
                        spatial.SEED_GRID_PHASE
                    ),
                    **{
                        f"calibration_{key}": value
                        for key, value in calibration_metrics.items()
                    },
                    **{
                        f"confirmation_{key}": value
                        for key, value in confirmation_metrics.items()
                    },
                    **losses,
                }
                rows.append(row)
                _write_csv(target, rows)
                completed.add(key)
                print(
                    f"Task2 5.2 {dataset}/{recipe}/{arm}/seed={seed}: "
                    f"F1={confirmation_metrics['f1']:.5f}",
                    flush=True,
                )
    return target


def _aggregate_recipe(rows: list[dict], spec: dict, recipe: str) -> dict:
    selected = [row for row in rows if row["recipe"] == recipe]
    datasets = {}
    paired = []
    for dataset in spec["datasets"]:
        subset = [row for row in selected if row["dataset"] == dataset]
        per_arm = {}
        for arm in ("raw", "fmt"):
            values = np.asarray([
                float(row["confirmation_f1"])
                for row in subset if row["arm"] == arm
            ])
            per_arm[arm] = {
                "f1": float(values.mean()),
                "f1_std": float(values.std(ddof=0)),
            }
        seed_gains = []
        for seed in spec["final_training_seeds"]:
            raw = next(
                float(row["confirmation_f1"]) for row in subset
                if row["arm"] == "raw" and int(row["training_seed"]) == seed
            )
            fmt = next(
                float(row["confirmation_f1"]) for row in subset
                if row["arm"] == "fmt" and int(row["training_seed"]) == seed
            )
            seed_gains.append(fmt - raw)
            paired.append({"dataset": dataset, "seed": seed, "gain": fmt - raw})
        datasets[dataset] = {
            "group": subset[0]["group"],
            "latent_dim": int(subset[0]["latent_dim"]),
            "raw_f1": per_arm["raw"]["f1"],
            "raw_f1_std": per_arm["raw"]["f1_std"],
            "fmt_f1": per_arm["fmt"]["f1"],
            "fmt_f1_std": per_arm["fmt"]["f1_std"],
            "fmt_minus_raw_f1": float(np.mean(seed_gains)),
            "fmt_minus_raw_f1_std": float(np.std(seed_gains, ddof=0)),
            "seed_gains": seed_gains,
        }
    raw_macro = float(np.mean([value["raw_f1"] for value in datasets.values()]))
    fmt_macro = float(np.mean([value["fmt_f1"] for value in datasets.values()]))
    family_names = sorted({value["group"] for value in datasets.values()})
    family_gains = {
        family: float(np.mean([
            value["fmt_minus_raw_f1"] for value in datasets.values()
            if value["group"] == family
        ]))
        for family in family_names
    }
    seed_macro_gains = {
        str(seed): float(np.mean([
            value["gain"] for value in paired if value["seed"] == seed
        ]))
        for seed in spec["final_training_seeds"]
    }
    return {
        "datasets": datasets,
        "raw_f1": raw_macro,
        "fmt_f1": fmt_macro,
        "fmt_minus_raw_f1": fmt_macro - raw_macro,
        "positive_datasets": sum(
            value["fmt_minus_raw_f1"] > 0.0 for value in datasets.values()
        ),
        "positive_families": sum(value > 0.0 for value in family_gains.values()),
        "positive_seed_macros": sum(
            value > 0.0 for value in seed_macro_gains.values()
        ),
        "dataset_count": len(datasets),
        "family_count": len(family_gains),
        "seed_count": len(seed_macro_gains),
        "worst_dataset_f1_gain": min(
            value["fmt_minus_raw_f1"] for value in datasets.values()
        ),
        "family_macro_f1_gain": float(np.mean(list(family_gains.values()))),
        "family_gains": family_gains,
        "seed_macro_gains": seed_macro_gains,
    }


def summarize(config_path: str | Path) -> Path:
    spec = _load_spec(config_path)
    recipe_path, _ = _require_recipe(spec)
    rows = []
    expected_per_dataset = (
        len(spec["recipes"]) * 2 * len(spec["final_training_seeds"])
    )
    for dataset in spec["datasets"]:
        values = _read_csv(
            Path(spec["output_root"]) / "shards" / f"{dataset}.csv"
        )
        keys = {
            (row["recipe"], row["arm"], int(row["training_seed"]))
            for row in values
        }
        if len(values) != expected_per_dataset or len(keys) != expected_per_dataset:
            raise RuntimeError(
                f"Task2 5.2 {dataset} has {len(values)} rows, "
                f"expected {expected_per_dataset} unique rows"
            )
        rows.extend(values)
    output = Path(spec["output_root"])
    _write_csv(output / "per_run.csv", rows)
    selected = _aggregate_recipe(rows, spec, "selected")
    control = _aggregate_recipe(rows, spec, "control")
    summary = {
        "experiment": spec["experiment"],
        "comparison": "same VAE and paired seed: FMT+VAE minus Raw+VAE",
        "primary_recipe": "selected",
        "control_is_diagnostic_only": True,
        "selection_sha256": str(spec["source_search"]["selection_sha256"]),
        "recipe_manifest_sha256": _sha256(recipe_path),
        "confirmation_seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "new_spatial_primitive_population": True,
        "selected": selected,
        "control": control,
        "selected_gain_change_from_control": (
            selected["fmt_minus_raw_f1"] - control["fmt_minus_raw_f1"]
        ),
        "target_dataset_macro_f1_gain": float(
            spec["target_dataset_macro_f1_gain"]
        ),
        "target_reached": selected["fmt_minus_raw_f1"] >= float(
            spec["target_dataset_macro_f1_gain"]
        ),
        "aspirational_dataset_macro_f1_gain": float(
            spec["aspirational_dataset_macro_f1_gain"]
        ),
        "aspirational_target_reached": selected["fmt_minus_raw_f1"] >= float(
            spec["aspirational_dataset_macro_f1_gain"]
        ),
        "checkpoint_policy": "no checkpoints were written",
    }
    table = []
    for dataset in spec["datasets"]:
        primary = selected["datasets"][dataset]
        baseline = control["datasets"][dataset]
        table.append({
            "dataset": dataset,
            "family": primary["group"],
            "selected_latent_dim": primary["latent_dim"],
            "selected_raw_f1": primary["raw_f1"],
            "selected_fmt_f1": primary["fmt_f1"],
            "selected_f1_gain": primary["fmt_minus_raw_f1"],
            "control_latent_dim": baseline["latent_dim"],
            "control_raw_f1": baseline["raw_f1"],
            "control_fmt_f1": baseline["fmt_f1"],
            "control_f1_gain": baseline["fmt_minus_raw_f1"],
        })
    _write_csv(output / "paper_table.csv", table)
    target = output / "summary.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "static-preflight", "freeze", "evaluation-preflight",
            "dataset", "summary",
        ),
        required=True,
    )
    parser.add_argument("--dataset")
    args = parser.parse_args()
    if args.mode == "static-preflight":
        static_preflight(args.config)
    elif args.mode == "freeze":
        freeze(args.config)
    elif args.mode == "evaluation-preflight":
        evaluation_preflight(args.config)
    elif args.mode == "summary":
        summarize(args.config)
    elif args.dataset is None:
        parser.error("dataset mode requires --dataset")
    else:
        run_dataset(args.config, args.dataset)


if __name__ == "__main__":
    main()
