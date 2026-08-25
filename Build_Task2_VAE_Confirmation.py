"""Freeze a development-selected Task2 VAE configuration for confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_confirmation_config(
    sweep_config_path: str | Path,
    selection_path: str | Path,
    output_path: str | Path,
) -> Path:
    sweep = yaml.safe_load(Path(sweep_config_path).read_text(encoding="utf-8"))
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if not selection.get("hierarchy_satisfied", False):
        raise RuntimeError(
            "development hierarchy was not satisfied; confirmation is forbidden"
        )

    variants = {variant["id"]: variant for variant in sweep["variants"]}
    selected_ids = {
        group: values["variant"] for group, values in selection["selected"].items()
    }
    missing = sorted(set(sweep["groups"]).difference(selected_ids))
    if missing:
        raise RuntimeError(f"development selection is missing groups: {missing}")
    selected_variants = []
    seen = set()
    for group in sweep["groups"]:
        identifier = selected_ids[group]
        if identifier not in variants:
            raise RuntimeError(f"selected variant {identifier!r} is absent from sweep")
        if identifier not in seen:
            selected_variants.append(variants[identifier])
            seen.add(identifier)

    groups = {
        group: {
            "datasets": values["datasets"],
            "fmt_feature": values["fmt_feature"],
            "fixed_architecture": selected_ids[group],
        }
        for group, values in sweep["groups"].items()
    }
    config = {
        "experiment": "mainExp_Task2_3D_3.2",
        "source_config": "config/Verify_Task2Universality_1.1.yaml",
        "output_dir": "outputs/mainExp_Task2_3D_3.2",
        "cache_roots": {
            "development": "outputs/Verify_Task2Universality_1.1/cache",
            "confirmation": "outputs/mainExp_Task3Universality_2.2/confirmation_cache",
        },
        "cache_overrides": {
            dataset: {
                "development": "outputs/mainExp_Task123NewFlows_1.1/development_cache",
                "confirmation": "outputs/mainExp_Task123NewFlows_1.1/confirmation_cache",
            }
            for dataset in ("boeing747", "smokeBuoyancy")
        },
        "groups": groups,
        "splits": {
            "final_train": list(range(8)),
            "cluster_calibration": [8, 9],
        },
        "architectures": selected_variants,
        # New seeds are disjoint from development-selection seeds 7068/7069.
        "final_training_seeds": [8068, 8069, 8070, 8071, 8072],
        "kmeans_seed": 7068,
        "kmeans_n_init": 20,
        "selection_provenance": {
            "sweep_experiment": sweep["experiment"],
            "selection_file": str(selection_path),
            "development_task1_f1_mean": selection["task1_f1_mean"],
            "development_raw_f1_mean": selection["raw_f1_mean"],
            "development_fmt_f1_mean": selection["fmt_f1_mean"],
            "development_minimum_hierarchy_margin": selection[
                "minimum_hierarchy_margin"
            ],
            "confirmation_labels_used_for_selection": False,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote frozen confirmation config: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-config", default="config/Verify_Task2_VAEGrid_3D_3.1.yaml"
    )
    parser.add_argument(
        "--selection",
        default="outputs/Verify_Task2_VAEGrid_3D_3.1/development_selection.json",
    )
    parser.add_argument(
        "--output", default="config/mainExp_Task2_3D_3.2.yaml"
    )
    args = parser.parse_args()
    build_confirmation_config(args.sweep_config, args.selection, args.output)
