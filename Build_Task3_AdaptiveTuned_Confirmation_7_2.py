"""Build the sealed sixth spatial population for Task3 mainExp 7.2.

This module reuses the validated 7.1/6.1 cache builder and replaces only the
experiment identity, output roots, and the pre-registered seed-grid phase.
Physical times, temporal source windows, pathline integration settings, and
whole-field IVD-p95 labels remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import Build_Task3_FinalTuned_Confirmation_7_1 as _base


EXPERIMENT = "mainExp_Task3_3D_7.2"
PHASE_KEY = "mainExp_Task3_3D_7.2|sixth-spatial-population-v1"
PHASE_KEY_SHA256 = (
    "ff3312a55c80504295453f30f21340683392157b0e1feabc05a798442ab0581a"
)
HALTON_INDEX = 678
SEED_GRID_PHASE = [
    -0.1044921875,
    -0.3655692729766804,
    0.11632000000000009,
]

SETTINGS = {
    "old8": {
        "base": "config/mainExp_Task3_3D_3.1_confirmation_old8.yaml",
        "cache_dir": "outputs/mainExp_Task3_3D_7.2/confirmation_cache_old8",
        "label_config": "config/mainExp_Task3_3D_7.2_labels_old8.yaml",
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
        "cache_dir": "outputs/mainExp_Task3_3D_7.2/confirmation_cache_new2",
        "label_config": "config/mainExp_Task3_3D_7.2_labels_new2.yaml",
        "indices": {
            "boeing747": [45, 77, 110, 185],
            "smokeBuoyancy": [36, 62, 89, 146],
        },
    },
}

SOURCE_STAGING_ENV = "TASK3_TUNED72_SOURCE_MANIFEST"
RECIPE_MANIFEST_ENV = "TASK3_TUNED72_RECIPE_MANIFEST"
DEFAULT_SOURCE_STAGING = (
    "/ibex/scratch/zhanx0o/FMT_Task3_AdaptiveTuned_7_2/"
    "source_staging_manifest.json"
)
DEFAULT_RECIPE_MANIFEST = (
    "outputs/mainExp_Task3_3D_7.2/frozen_recipe_manifest.json"
)


@contextmanager
def _configured_base():
    """Apply 7.2 constants only for one validated 7.1 builder call."""
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "PHASE_KEY": PHASE_KEY,
        "PHASE_KEY_SHA256": PHASE_KEY_SHA256,
        "HALTON_INDEX": HALTON_INDEX,
        "SEED_GRID_PHASE": SEED_GRID_PHASE,
        "SETTINGS": SETTINGS,
        "SOURCE_STAGING_ENV": SOURCE_STAGING_ENV,
        "RECIPE_MANIFEST_ENV": RECIPE_MANIFEST_ENV,
        "DEFAULT_SOURCE_STAGING": DEFAULT_SOURCE_STAGING,
        "DEFAULT_RECIPE_MANIFEST": DEFAULT_RECIPE_MANIFEST,
    }
    previous = {name: getattr(_base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def _expected_datasets() -> set[str]:
    return {
        dataset
        for settings in SETTINGS.values()
        for dataset in settings["indices"]
    }


def source_staging_manifest() -> tuple[Path, dict]:
    with _configured_base():
        return _base.source_staging_manifest()


def source_staging_identity() -> dict:
    with _configured_base():
        return _base.source_staging_identity()


def _require_recipe_frozen() -> tuple[Path, dict]:
    with _configured_base():
        return _base._require_recipe_frozen()


def _cache_config(group: str):
    with _configured_base():
        return _base._cache_config(group)


def source_preflight() -> dict:
    with _configured_base():
        return _base.source_preflight()


def jobs() -> list[tuple[str, str]]:
    return [
        (group, dataset)
        for group, settings in SETTINGS.items()
        for dataset in settings["indices"]
    ]


def build_cache(group: str, dataset: str, overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.build_cache(group, dataset, overwrite)


def build_job(index: int, overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.build_job(index, overwrite)


def build_labels(group: str, overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.build_labels(group, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("source-preflight", "cache", "labels"),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--group", choices=sorted(SETTINGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "source-preflight":
        import json

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
