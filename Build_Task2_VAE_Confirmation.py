"""Freeze a development-selected Task2 VAE configuration for confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from Sweep_Task2_VAE_3D import _load_spec


def build_confirmation_config(
    sweep_config_path: str | Path,
    selection_path: str | Path,
    output_path: str | Path,
    *,
    experiment: str = "mainExp_Task2_3D_3.2",
    output_dir: str = "outputs/mainExp_Task2_3D_3.2",
    confirmation_cache: str = (
        "outputs/mainExp_Task3Universality_2.2/confirmation_cache"
    ),
    newflow_confirmation_cache: str = (
        "outputs/mainExp_Task123NewFlows_1.1/confirmation_cache"
    ),
    confirmation_count: int = 4,
    final_training_seeds: list[int] | None = None,
) -> Path:
    sweep = _load_spec(sweep_config_path)
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
        "experiment": experiment,
        "source_config": "config/Verify_Task2Universality_1.1.yaml",
        "output_dir": output_dir,
        "cache_roots": {
            "development": "outputs/Verify_Task2Universality_1.1/cache",
            "confirmation": confirmation_cache,
        },
        "cache_overrides": {
            dataset: {
                "development": "outputs/mainExp_Task123NewFlows_1.1/development_cache",
                "confirmation": newflow_confirmation_cache,
            }
            for dataset in ("boeing747", "smokeBuoyancy")
        },
        "groups": groups,
        "splits": {
            "final_train": list(range(8)),
            "cluster_calibration": [8, 9],
            "confirmation_count": int(confirmation_count),
        },
        "architectures": selected_variants,
        # New seeds are disjoint from development-selection seeds 7068/7069.
        "final_training_seeds": final_training_seeds or [8068, 8069, 8070, 8071, 8072],
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
    parser.add_argument("--experiment", default="mainExp_Task2_3D_3.2")
    parser.add_argument("--output-dir", default="outputs/mainExp_Task2_3D_3.2")
    parser.add_argument(
        "--confirmation-cache",
        default="outputs/mainExp_Task3Universality_2.2/confirmation_cache",
    )
    parser.add_argument(
        "--newflow-confirmation-cache",
        default="outputs/mainExp_Task123NewFlows_1.1/confirmation_cache",
    )
    parser.add_argument("--confirmation-count", type=int, default=4)
    parser.add_argument("--final-training-seeds", type=int, nargs="+")
    args = parser.parse_args()
    build_confirmation_config(
        args.sweep_config, args.selection, args.output,
        experiment=args.experiment,
        output_dir=args.output_dir,
        confirmation_cache=args.confirmation_cache,
        newflow_confirmation_cache=args.newflow_confirmation_cache,
        confirmation_count=args.confirmation_count,
        final_training_seeds=args.final_training_seeds,
    )
