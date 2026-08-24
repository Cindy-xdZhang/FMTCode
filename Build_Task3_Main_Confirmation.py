"""Build fresh Task3 main-experiment confirmation caches and labels."""

from __future__ import annotations

import argparse

from Build_Channel_Killing_Cache import build as build_channel
from Build_Task2_Universality_Cache import build_dataset
from Build_Task3_Universality_Labels import build as build_labels
from DeepUtils.utils import EasyConfig


CONFIGS = {
    "old8": {
        "cache": "config/mainExp_Task3_3D_3.1_confirmation_old8.yaml",
        "labels": "config/mainExp_Task3_3D_3.1_labels_old8.yaml",
    },
    "new2": {
        "cache": "config/mainExp_Task3_3D_3.1_confirmation_new2.yaml",
        "labels": "config/mainExp_Task3_3D_3.1_labels_new2.yaml",
    },
}


def build_cache(group, dataset, overwrite=False):
    config_path = CONFIGS[group]["cache"]
    config = EasyConfig(config_path)
    available = [
        str(item["id"] if isinstance(item, dict) else item.id)
        for item in config.datasets
    ]
    datasets = available if dataset == "all" else [dataset]
    unknown = set(datasets) - set(available)
    if unknown:
        raise ValueError(f"unknown {group} datasets: {sorted(unknown)}")
    for name in datasets:
        if name == "channel":
            build_channel(config, overwrite=overwrite)
        else:
            build_dataset(config, name, overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--stage", choices=("cache", "labels"), required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "cache":
        build_cache(args.group, args.dataset, args.overwrite)
    else:
        if args.dataset != "all":
            raise ValueError("label stage currently processes the complete group")
        build_labels(CONFIGS[args.group]["labels"], overwrite=args.overwrite)


if __name__ == "__main__":
    main()
