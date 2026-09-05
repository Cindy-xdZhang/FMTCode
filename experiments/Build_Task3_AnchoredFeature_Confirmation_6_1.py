"""Build the sealed fourth spatial population for Task3 mainExp 6.1.

The physical time slices and pathline integration settings are unchanged from
the earlier spatial checks.  Only the seed-grid phase changes.  Cache or label
generation is forbidden until the frozen 6.1 recipe manifest exists.
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
from Build_Task3_GlobalIVD_Labels import build as build_global_ivd_labels
from DeepUtils.utils import EasyConfig
from FMT_Utils.NetCDF_window_3D import inspect_netcdf_3d


EXPERIMENT = "mainExp_Task3_3D_6.1"
PHASE_KEY = "mainExp_Task3_3D_6.1|fourth-spatial-population-v1"
PHASE_KEY_SHA256 = (
    "dce639a01fa0139281f3fedb76c3d8d40ed3530de35dec0507c965fac4bb9b3a"
)
HALTON_INDEX = 417
SEED_GRID_PHASE = [0.021484375, -0.34224965706447186, 0.0328]

SETTINGS = {
    "old8": {
        "base": "config/mainExp_Task3_3D_3.1_confirmation_old8.yaml",
        "cache_dir": "outputs/mainExp_Task3_3D_6.1/confirmation_cache_old8",
        "label_config": "config/mainExp_Task3_3D_6.1_labels_old8.yaml",
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
        "cache_dir": "outputs/mainExp_Task3_3D_6.1/confirmation_cache_new2",
        "label_config": "config/mainExp_Task3_3D_6.1_labels_new2.yaml",
        "indices": {
            "boeing747": [45, 77, 110, 185],
            "smokeBuoyancy": [36, 62, 89, 146],
        },
    },
}

SOURCE_STAGING_ENV = "TASK3_ANCHORED6_SOURCE_MANIFEST"
RECIPE_MANIFEST_ENV = "TASK3_ANCHORED6_RECIPE_MANIFEST"
DEFAULT_SOURCE_STAGING = (
    "/ibex/scratch/zhanx0o/FMT_Task3_AnchoredFeature_6_1/"
    "source_staging_manifest.json"
)
DEFAULT_RECIPE_MANIFEST = (
    "outputs/mainExp_Task3_3D_6.1/frozen_recipe_manifest.json"
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
        raise RuntimeError("Task3 6.1 source-staging experiment changed")
    if set(payload.get("datasets", {})) != _expected_datasets():
        raise RuntimeError("Task3 6.1 source-staging dataset set changed")
    if list(payload.get("seed_grid_phase", [])) != SEED_GRID_PHASE:
        raise RuntimeError("Task3 6.1 source-staging phase changed")
    if not bool(payload.get("scientific_protocol_unchanged", False)):
        raise RuntimeError("Task3 6.1 source-staging equivalence is missing")
    if not bool(payload.get("temporal_sources_are_phase_independent", False)):
        raise RuntimeError("Task3 6.1 temporal-pack independence is missing")
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
            f"Task3 6.1 recipe must be frozen before opening data: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT:
        raise RuntimeError("Task3 6.1 frozen experiment changed")
    if list(payload.get("confirmation_seed_grid_phase", [])) != SEED_GRID_PHASE:
        raise RuntimeError("Task3 6.1 frozen phase changed")
    if bool(payload.get("confirmation_data_opened", True)):
        raise RuntimeError("Task3 6.1 recipe says confirmation was opened")
    staged = source_staging_identity()
    if payload.get("source_staging") != staged:
        raise RuntimeError("Task3 6.1 source staging changed after freeze")
    return path, payload


def _cache_config(group: str) -> EasyConfig:
    settings = SETTINGS[group]
    config = EasyConfig(settings["base"])
    config.experiment = f"{EXPERIMENT}_confirmation_{group}"
    config.sampling.timeslices = 4
    config.sampling.fixed_time_indices_by_dataset = settings["indices"]
    config.sampling.seed_grid_phase = list(SEED_GRID_PHASE)
    config.output.cache_dir = settings["cache_dir"]
    config.output.result_dir = (
        f"outputs/{EXPERIMENT}/unused_reference_{group}"
    )

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
                "kind": str(source["kind"]),
                "path": str(path.resolve()),
                "original_fixed_indices": [
                    int(value) for value in settings["indices"][dataset]
                ],
                "effective_fixed_indices": effective,
                "frame_count": frame_count,
                "exists": True,
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
    index = int(index)
    all_jobs = jobs()
    if not 0 <= index < len(all_jobs):
        raise IndexError(f"cache job index {index} outside [0,{len(all_jobs)})")
    return build_cache(*all_jobs[index], overwrite=overwrite)


def build_labels(group: str, overwrite: bool = False) -> Path:
    _require_recipe_frozen()
    if group not in SETTINGS:
        raise ValueError(f"unknown label group: {group}")
    config_path = Path(SETTINGS[group]["label_config"])
    build_global_ivd_labels(config_path, overwrite)
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source-preflight", "cache", "labels"), required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--group", choices=sorted(SETTINGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "source-preflight":
        print(json.dumps(source_preflight(), indent=2, sort_keys=True))
    elif args.mode == "cache":
        if args.job_index is None:
            parser.error("cache mode requires --job-index")
        build_job(args.job_index, args.overwrite)
    else:
        if args.group is None:
            parser.error("labels mode requires --group")
        build_labels(args.group, args.overwrite)


if __name__ == "__main__":
    main()
