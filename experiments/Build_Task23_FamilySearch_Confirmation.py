"""Build phased-grid confirmation caches after Task2/Task3 recipes freeze.

The script deliberately inherits dataset paths from the already deployed
Task3 confirmation configs, so local Windows and Ibex asset roots remain
portable.  It changes only exact start indices, spatial seed phase, output,
and the number of slices.
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
        "cache_dir": "outputs/mainExp_Task23_3D_4.1/confirmation_cache_old8",
        "label_config": "config/mainExp_Task23_3D_4.1_labels_old8.yaml",
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
        "cache_dir": "outputs/mainExp_Task23_3D_4.1/confirmation_cache_new2",
        "label_config": "config/mainExp_Task23_3D_4.1_labels_new2.yaml",
        "indices": {
            "boeing747": [45, 77, 110, 185],
            "smokeBuoyancy": [36, 62, 89, 146],
        },
    },
}
SEED_GRID_PHASE = [0.31, -0.23, 0.17]
SELECTION_FILES = {
    "task2": "outputs/Verify_Task2_FMTVAEFamilySearch_4.1/stage2_selection.json",
    "task3": "outputs/Verify_Task3_FMTResidualFamilySearch_4.1/stage2_selection.json",
}


def _assert_recipes_frozen() -> dict:
    selections = {}
    for task, value in SELECTION_FILES.items():
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(
                f"{task} stage2 recipe must be frozen before confirmation: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bool(payload.get("confirmation_opened", False)):
            raise RuntimeError(f"{task} selection already claims confirmation access")
        selections[task] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "experiment": payload["experiment"],
        }
    root = Path("outputs/mainExp_Task23_3D_4.1")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "frozen_recipe_manifest.json"
    frozen = {
        "experiment": "mainExp_Task23_3D_4.1",
        "selections": selections,
        "seed_grid_phase": list(SEED_GRID_PHASE),
        "historical_temporal_independence": False,
        "new_spatial_primitive_population": True,
    }
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous != frozen:
            raise RuntimeError("confirmation recipes changed after cache access")
    else:
        temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, marker)
    return frozen


def _cache_config(group: str) -> EasyConfig:
    settings = SETTINGS[group]
    config = EasyConfig(settings["base"])
    config.experiment = f"mainExp_Task23_3D_4.1_confirmation_{group}"
    config.sampling.timeslices = 4
    config.sampling.fixed_time_indices_by_dataset = settings["indices"]
    config.sampling.seed_grid_phase = list(SEED_GRID_PHASE)
    config.output.cache_dir = settings["cache_dir"]
    config.output.result_dir = (
        f"outputs/mainExp_Task23_3D_4.1/unused_reference_{group}"
    )
    return config


def build_cache(group: str, dataset: str, overwrite=False) -> Path:
    _assert_recipes_frozen()
    config = _cache_config(group)
    available = [
        str(item["id"] if isinstance(item, dict) else item.id)
        for item in config.datasets
    ]
    requested = available if dataset == "all" else [dataset]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown {group} datasets: {unknown}")
    target = None
    for name in requested:
        if name == "channel":
            target = build_channel(config, overwrite=overwrite)
        else:
            target = build_dataset(config, name, overwrite=overwrite)
    return Path(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(SETTINGS), required=True)
    parser.add_argument("--stage", choices=("cache", "labels"), required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "cache":
        build_cache(args.group, args.dataset, args.overwrite)
    else:
        _assert_recipes_frozen()
        if args.dataset != "all":
            raise ValueError("label stage processes all datasets in a group")
        build_labels(SETTINGS[args.group]["label_config"], args.overwrite)


if __name__ == "__main__":
    main()
