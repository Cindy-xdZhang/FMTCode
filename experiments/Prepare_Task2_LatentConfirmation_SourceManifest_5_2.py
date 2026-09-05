"""Derive the Task2 5.2 temporal-source manifest without copying flow data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

import Build_Task2_LatentConfirmation_5_2 as spatial


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(config_path: str | Path, overwrite: bool = False) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    section = dict(spec["source_staging"])
    parent = Path(section["parent_manifest"])
    if not parent.is_file():
        raise FileNotFoundError(parent)
    parent_hash = _sha256(parent)
    if parent_hash != str(section["parent_manifest_sha256"]):
        raise RuntimeError("Task2 5.2 parent source manifest changed")
    source = json.loads(parent.read_text(encoding="utf-8"))
    if not bool(source.get("scientific_protocol_unchanged", False)):
        raise RuntimeError("parent source manifest lacks equivalence declaration")
    expected = spatial._expected_datasets()
    if set(source.get("datasets", {})) != expected:
        raise RuntimeError("parent source manifest dataset set changed")

    datasets = {}
    for group in spatial.SETTINGS.values():
        for dataset, indices in group["indices"].items():
            item = dict(source["datasets"][dataset])
            observed = [int(value) for value in item["original_fixed_indices"]]
            if observed != [int(value) for value in indices]:
                raise RuntimeError(f"{dataset}: parent physical times changed")
            path = Path(item["path"])
            if not path.exists():
                raise FileNotFoundError(path)
            expected_hash = item.get("sha256")
            if expected_hash and _sha256(path) != str(expected_hash):
                raise RuntimeError(f"{dataset}: temporal source changed")
            datasets[dataset] = {
                key: item[key]
                for key in (
                    "kind", "path", "original_fixed_indices",
                    "effective_fixed_indices", "sha256",
                    "all_windows_verified_exact",
                )
                if key in item
            }

    payload = {
        "schema": 1,
        "experiment": f"{spatial.EXPERIMENT}_source_staging",
        "parent_manifest": str(parent.resolve()),
        "parent_manifest_sha256": parent_hash,
        "parent_equivalence": source.get("equivalence", ""),
        "scientific_protocol_unchanged": True,
        "temporal_sources_are_phase_independent": True,
        "seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "phase_key": spatial.PHASE_KEY,
        "phase_key_sha256": spatial.PHASE_KEY_SHA256,
        "halton_index": spatial.HALTON_INDEX,
        "datasets": datasets,
    }
    target = Path(section["derived_manifest"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError("Task2 5.2 derived source manifest changed")
        print(target)
        return target
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, target)
    print(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    derive(args.config, args.overwrite)


if __name__ == "__main__":
    main()
