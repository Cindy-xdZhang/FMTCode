"""Build the frozen fifth spatial population for Task2 mainExp 5.2.

The temporal velocity windows and pathline integration settings are inherited
unchanged from the audited Task3 source packs.  Only the spatial seed-grid
phase changes.  Cache generation is forbidden until the Task2 recipe,
including the Task2 5.1 selection SHA-256, has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from Build_Channel_Killing_Cache import build as build_channel
from Build_Task2_Universality_Cache import build_dataset
from DeepUtils.utils import EasyConfig
from FMT_Utils.NetCDF_window_3D import inspect_netcdf_3d


EXPERIMENT = "mainExp_Task2_3D_5.2"
PHASE_KEY = "mainExp_Task2_3D_5.2|fifth-spatial-population-v1"
PHASE_KEY_SHA256 = (
    "0abfd21f52056f9beed62d3b2568fb0620f2369cbd08ccc009b6447554627d54"
)
HALTON_INDEX = 544
SEED_GRID_PHASE = [-0.4833984375, -0.028120713305898493, 0.4344]

SETTINGS = {
    "old8": {
        "base": "config/mainExp_Task3_3D_3.1_confirmation_old8.yaml",
        "cache_dir": "outputs/mainExp_Task2_3D_5.2/confirmation_cache_old8",
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
        "cache_dir": "outputs/mainExp_Task2_3D_5.2/confirmation_cache_new2",
        "indices": {
            "boeing747": [45, 77, 110, 185],
            "smokeBuoyancy": [36, 62, 89, 146],
        },
    },
}

SOURCE_STAGING_ENV = "TASK2_LATENT52_SOURCE_MANIFEST"
RECIPE_MANIFEST_ENV = "TASK2_LATENT52_RECIPE_MANIFEST"
DEFAULT_SOURCE_STAGING = (
    "/ibex/scratch/zhanx0o/FMT_Task2_LatentBottleneck_5_2/"
    "source_staging_manifest.json"
)
DEFAULT_RECIPE_MANIFEST = (
    "outputs/mainExp_Task2_3D_5.2/frozen_recipe_manifest.json"
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected_datasets() -> set[str]:
    return {
        dataset
        for settings in SETTINGS.values()
        for dataset in settings["indices"]
    }


def source_staging_manifest() -> tuple[Path, dict]:
    path = Path(os.environ.get(SOURCE_STAGING_ENV, DEFAULT_SOURCE_STAGING))
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != f"{EXPERIMENT}_source_staging":
        raise RuntimeError("Task2 5.2 source-staging experiment changed")
    if set(payload.get("datasets", {})) != _expected_datasets():
        raise RuntimeError("Task2 5.2 source-staging dataset set changed")
    if list(payload.get("seed_grid_phase", [])) != SEED_GRID_PHASE:
        raise RuntimeError("Task2 5.2 source-staging phase changed")
    if not bool(payload.get("scientific_protocol_unchanged", False)):
        raise RuntimeError("Task2 5.2 source staging lacks protocol identity")
    if not bool(payload.get("temporal_sources_are_phase_independent", False)):
        raise RuntimeError("Task2 5.2 temporal source independence is missing")
    return path, payload


def source_staging_identity() -> dict:
    path, payload = source_staging_manifest()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "experiment": payload["experiment"],
        "parent_manifest_sha256": payload["parent_manifest_sha256"],
    }


def _require_recipe_frozen() -> tuple[Path, dict]:
    path = Path(os.environ.get(RECIPE_MANIFEST_ENV, DEFAULT_RECIPE_MANIFEST))
    if not path.is_file():
        raise FileNotFoundError(
            f"Task2 5.2 recipe must be frozen before opening data: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT:
        raise RuntimeError("Task2 5.2 frozen experiment changed")
    if list(payload.get("confirmation_seed_grid_phase", [])) != SEED_GRID_PHASE:
        raise RuntimeError("Task2 5.2 frozen phase changed")
    if bool(payload.get("confirmation_data_opened", True)):
        raise RuntimeError("Task2 5.2 recipe says confirmation was opened")
    if payload.get("source_staging") != source_staging_identity():
        raise RuntimeError("Task2 5.2 source staging changed after freeze")
    return path, payload


def _cache_config(group: str) -> EasyConfig:
    settings = SETTINGS[group]
    config = EasyConfig(settings["base"])
    config.experiment = f"{EXPERIMENT}_confirmation_{group}"
    config.sampling.timeslices = 4
    config.sampling.fixed_time_indices_by_dataset = settings["indices"]
    config.sampling.seed_grid_phase = list(SEED_GRID_PHASE)
    config.output.cache_dir = settings["cache_dir"]
    config.output.result_dir = f"outputs/{EXPERIMENT}/unused_reference_{group}"

    staging_path, staging = source_staging_manifest()
    entries = []
    for raw_item in config.datasets:
        item = dict(raw_item)
        dataset = str(item["id"])
        source = dict(staging["datasets"][dataset])
        original = [int(value) for value in source["original_fixed_indices"]]
        expected = [int(value) for value in settings["indices"][dataset]]
        if original != expected:
            raise RuntimeError(f"{dataset}: original confirmation times changed")
        effective = [int(value) for value in source["effective_fixed_indices"]]
        if len(effective) != len(expected):
            raise RuntimeError(f"{dataset}: staged time count changed")
        staged_path = Path(source["path"])
        if not staged_path.exists():
            raise FileNotFoundError(staged_path)
        expected_hash = source.get("sha256")
        if expected_hash and _sha256(staged_path) != str(expected_hash):
            raise RuntimeError(f"{dataset}: staged source SHA-256 changed")
        item["path"] = str(staged_path)
        entries.append(item)
    config.datasets = entries
    config.sampling.fixed_time_indices_by_dataset = {
        dataset: [
            int(value)
            for value in staging["datasets"][dataset]["effective_fixed_indices"]
        ]
        for dataset in settings["indices"]
    }
    config.sampling.original_fixed_time_indices_by_dataset = {
        dataset: [int(value) for value in indices]
        for dataset, indices in settings["indices"].items()
    }
    config.sampling.source_staging_manifest = str(staging_path.resolve())
    config.sampling.source_staging_manifest_sha256 = _sha256(staging_path)
    return config


def source_preflight() -> dict:
    _require_recipe_frozen()
    staging_path, staging = source_staging_manifest()
    checked = {}
    for group, settings in SETTINGS.items():
        config = _cache_config(group)
        future_intervals = int(np.ceil(
            float(config.pathlines.dt_scale)
            * int(config.pathlines.integration_steps)
        ))
        frame_count = future_intervals + 2
        entries = {str(item["id"]): item for item in config.datasets}
        for dataset in settings["indices"]:
            source = dict(staging["datasets"][dataset])
            path = Path(entries[dataset]["path"])
            effective = [int(value) for value in source["effective_fixed_indices"]]
            if path.suffix.lower() == ".nc":
                info = inspect_netcdf_3d(path)
                if effective[-1] + frame_count > int(info["shape"]["t"]):
                    raise RuntimeError(f"{dataset}: source lacks final window")
            elif dataset != "channel" or path.suffix.lower() != ".vtk":
                raise RuntimeError(f"{dataset}: unsupported source {path}")
            checked[dataset] = {
                "group": group,
                "path": str(path.resolve()),
                "original_fixed_indices": [
                    int(value) for value in settings["indices"][dataset]
                ],
                "effective_fixed_indices": effective,
                "frame_count": frame_count,
                "sha256_verified": bool(source.get("sha256")),
            }
    return {
        "experiment": EXPERIMENT,
        "source_staging_manifest": str(staging_path.resolve()),
        "source_staging_manifest_sha256": _sha256(staging_path),
        "scientific_protocol_unchanged": True,
        "seed_grid_phase": list(SEED_GRID_PHASE),
        "datasets": checked,
    }


def write_source_preflight() -> Path:
    payload = source_preflight()
    target = Path(f"outputs/{EXPERIMENT}/source_preflight.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target)
    return target


def jobs() -> list[tuple[str, str]]:
    return [
        (group, dataset)
        for group, settings in SETTINGS.items()
        for dataset in settings["indices"]
    ]


def build_cache(group: str, dataset: str, overwrite: bool = False) -> Path:
    _require_recipe_frozen()
    if dataset not in SETTINGS[group]["indices"]:
        raise ValueError(f"unknown {group} dataset: {dataset}")
    config = _cache_config(group)
    if dataset == "channel":
        return Path(build_channel(config, overwrite=overwrite))
    return Path(build_dataset(config, dataset, overwrite=overwrite))


def build_job(index: int, overwrite: bool = False) -> Path:
    all_jobs = jobs()
    index = int(index)
    if not 0 <= index < len(all_jobs):
        raise IndexError(f"cache job index {index} outside [0,{len(all_jobs)})")
    return build_cache(*all_jobs[index], overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source-preflight", "cache"), required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "source-preflight":
        write_source_preflight()
    else:
        if args.job_index is None:
            parser.error("cache mode requires --job-index")
        build_job(args.job_index, args.overwrite)


if __name__ == "__main__":
    main()
