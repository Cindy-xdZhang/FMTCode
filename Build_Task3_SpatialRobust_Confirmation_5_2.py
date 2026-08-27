"""Build the sealed Task3 5.2 spatial confirmation population.

The physical times match 4.1 and 5.1.  Only the seed-grid phase changes, so
the final check measures generalization to a third primitive population.  No
cache may be generated until Stage-2 family recipes have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from Build_Channel_Killing_Cache import build as build_channel
from Build_Task2_Universality_Cache import build_dataset
from Build_Task3_GlobalIVD_Labels import build as build_labels
from DeepUtils.utils import EasyConfig


SETTINGS = {
    "old8": {
        "base": "config/mainExp_Task3_3D_3.1_confirmation_old8.yaml",
        "cache_dir": "outputs/mainExp_Task3_3D_5.2/confirmation_cache_old8",
        "label_config": "config/mainExp_Task3_3D_5.2_labels_old8.yaml",
        "indices": {
            "cylinder3d": [34, 58, 82, 137],
            "halfcylinderRe640": [19, 28, 39, 60],
            "halfcylinderRe6400": [34, 58, 82, 137],
            "tangaroa": [45, 77, 110, 187],
            "deltaWing_resampled": [39, 69, 108, 157],
            "deltaWing_LBM": [53, 94, 146, 220],
            "f22raptor": [36, 62, 89, 145],
            "channel": [36, 62, 89, 145],
        },
    },
    "new2": {
        "base": "config/mainExp_Task3_3D_3.1_confirmation_new2.yaml",
        "cache_dir": "outputs/mainExp_Task3_3D_5.2/confirmation_cache_new2",
        "label_config": "config/mainExp_Task3_3D_5.2_labels_new2.yaml",
        "indices": {
            "boeing747": [45, 77, 110, 185],
            "smokeBuoyancy": [36, 62, 89, 146],
        },
    },
}

# Deterministically pre-registered before any 5.2 Stage-1/Stage-2 result.
# SHA-256("mainExp_Task3_3D_5.2|final-phase-v1") selects centered Halton
# index 395 in bases (2,3,5).  This avoids manually choosing a favorable
# phase; all components are inside (-0.5, 0.5).
SEED_GRID_PHASE = [0.318359375, 0.4561042524005485, -0.3352]
SELECTION_FILE = "outputs/Verify_Task3_SpatialRobust_5.2/stage2_selection.json"
MANIFEST = "outputs/mainExp_Task3_3D_5.2/frozen_recipe_manifest.json"


def _portable_manifest_path(path: Path) -> str:
    return path.as_posix()


def _require_exposed_population(
    selection: dict,
    key: str,
    source_experiment: str,
    seed_grid_phase: list[float],
) -> None:
    population = selection.get(key)
    if not isinstance(population, dict):
        raise RuntimeError(f"frozen selection is missing {key}")
    if population.get("status") != "exposed_development":
        raise RuntimeError(f"{key} was not declared exposed development data")
    if population.get("source_experiment") != source_experiment:
        raise RuntimeError(f"{key} source experiment changed")
    if list(population.get("seed_grid_phase", [])) != list(seed_grid_phase):
        raise RuntimeError(f"{key} seed-grid phase changed")


def _assert_recipe_frozen() -> dict:
    selection_path = Path(SELECTION_FILE)
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Stage-2 selection must exist before confirmation: {selection_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if bool(selection.get("confirmation_opened", False)):
        raise RuntimeError("selection unexpectedly claims current confirmation access")
    _require_exposed_population(
        selection,
        "exposed_spatial_training",
        "mainExp_Task3_3D_4.1",
        [0.31, -0.23, 0.17],
    )
    _require_exposed_population(
        selection,
        "exposed_spatial_validation",
        "mainExp_Task3_3D_5.1",
        [-0.37, 0.29, -0.11],
    )
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    portable_selection_path = _portable_manifest_path(selection_path)
    frozen = {
        "experiment": "mainExp_Task3_3D_5.2",
        "selection": {
            "path": portable_selection_path,
            "sha256": selection_hash,
            "experiment": selection["experiment"],
        },
        "selections": {
            "task3": {
                "path": portable_selection_path,
                "sha256": selection_hash,
                "experiment": selection["experiment"],
            }
        },
        "seed_grid_phase": list(SEED_GRID_PHASE),
        "same_physical_times_as_4_1_and_5_1": True,
        "new_spatial_primitive_population": True,
        "training_population": "mainExp_Task3_3D_4.1",
        "validation_population": "mainExp_Task3_3D_5.1",
    }
    path = Path(MANIFEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != frozen:
            raise RuntimeError("Task3 5.2 frozen recipe manifest changed")
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    return frozen


def _cache_config(group: str) -> EasyConfig:
    settings = SETTINGS[group]
    config = EasyConfig(settings["base"])
    config.experiment = f"mainExp_Task3_3D_5.2_confirmation_{group}"
    config.sampling.timeslices = 4
    config.sampling.fixed_time_indices_by_dataset = settings["indices"]
    config.sampling.seed_grid_phase = list(SEED_GRID_PHASE)
    config.output.cache_dir = settings["cache_dir"]
    config.output.result_dir = (
        f"outputs/mainExp_Task3_3D_5.2/unused_reference_{group}"
    )
    return config


def _jobs() -> list[tuple[str, str]]:
    return [
        (group, dataset)
        for group, settings in SETTINGS.items()
        for dataset in settings["indices"]
    ]


def build_cache(group: str, dataset: str, overwrite=False) -> Path:
    _assert_recipe_frozen()
    config = _cache_config(group)
    if dataset not in SETTINGS[group]["indices"]:
        raise ValueError(f"unknown {group} dataset: {dataset}")
    if dataset == "channel":
        return Path(build_channel(config, overwrite=overwrite))
    return Path(build_dataset(config, dataset, overwrite=overwrite))


def build_job(index: int, overwrite=False) -> Path:
    jobs = _jobs()
    index = int(index)
    if not 0 <= index < len(jobs):
        raise IndexError(f"confirmation job index {index} outside [0,{len(jobs)})")
    return build_cache(*jobs[index], overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cache", "labels"), required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--group", choices=sorted(SETTINGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "cache":
        if args.job_index is None:
            parser.error("cache mode requires --job-index")
        build_job(args.job_index, args.overwrite)
    else:
        if args.group is None:
            parser.error("labels mode requires --group")
        _assert_recipe_frozen()
        build_labels(SETTINGS[args.group]["label_config"], args.overwrite)


if __name__ == "__main__":
    main()
