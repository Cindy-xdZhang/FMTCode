"""Screen high-Re pathline horizon and primitive-offset choices on common seeds."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from Build_Task2_Universality_Cache import build_dataset
from DeepUtils.utils import EasyConfig
from FMT_Utils.RawPathline_3D import raw_pathline_representation
from Run_Task2_Universality import _fit_cluster, _load_slices, _prepare


def _cfg(payload):
    config = EasyConfig(); config.update(payload); return config


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def _load_common_records(cache_root, variant_ids, variant_id, dataset):
    records = _load_slices(Path(cache_root) / variant_id / dataset)
    paths = {name: sorted((Path(cache_root) / name / dataset).glob("slice_*.npz"))
             for name in variant_ids}
    if any(len(values) != len(records) for values in paths.values()):
        raise RuntimeError(f"incomplete variant cache for {dataset}")
    for ordinal, record in enumerate(records):
        masks, source_indices = [], []
        for name in variant_ids:
            with np.load(paths[name][ordinal]) as data:
                masks.append(np.asarray(data["valid_mask"], dtype=bool))
                source_indices.append(json.loads(str(data["metadata_json"]))["source_start_index"])
        if len(set(source_indices)) != 1:
            raise RuntimeError(f"source index mismatch at {dataset} slice {ordinal}")
        common = np.logical_and.reduce(masks)
        with np.load(paths[variant_id][ordinal]) as data:
            current = np.asarray(data["valid_mask"], dtype=bool)
        selector = common[current]
        for key in ("raw", "fmt", "reference"):
            record[key] = record[key][selector]
        record["metadata"] = dict(record["metadata"])
        record["metadata"]["common_valid_fraction"] = float(common.mean())
    return records


def _score(records, representation, fit_slices, eval_slices, config):
    train_lengths = [len(record["reference"]) for record in records[:fit_slices]]
    selected = list(range(fit_slices)) + list(eval_slices)
    if representation == "fmt":
        values = np.concatenate([records[index]["fmt"] for index in selected])
        train, evaluate = _prepare(
            values, train_lengths, float(config.task2.fmt_neighbor_weight)
        )
    else:
        transformed = [raw_pathline_representation(records[index]["raw"], representation)
                       for index in selected]
        values = np.concatenate(transformed)
        train, evaluate = _prepare(values, train_lengths)
    reference = np.concatenate([records[index]["reference"] for index in eval_slices])
    with threadpool_limits(limits=1):
        _, cluster, metrics = _fit_cluster(
            train, evaluate, reference, config, already_scaled=True
        )
    return cluster, metrics


def _variant_config(source, spec, variant, dataset):
    payload = copy.deepcopy(source)
    payload["experiment"] = spec["experiment"]
    payload["sampling"]["fixed_time_indices"] = spec["fixed_time_indices"][dataset]
    payload["sampling"]["max_spatial_dim"] = int(variant["max_spatial_dim"])
    payload["sampling"]["seed_grid_shape"] = list(variant["seed_grid_shape"])
    payload["pathlines"].update({
        "dt_scale": float(variant["dt_scale"]),
        "integration_steps": int(variant["integration_steps"]),
        "sampled_steps": int(variant["sampled_steps"]),
        "offset_grid_scale": float(variant["offset_grid_scale"]),
        "offset_mode": str(variant["offset_mode"]),
    })
    payload["output"] = {
        "cache_dir": str(Path(spec["output_dir"]) / "cache" / variant["id"]),
        "result_dir": str(Path(spec["output_dir"]) / "unused_results" / variant["id"]),
    }
    return _cfg(payload)


def run(spec_path, phase="all", overwrite=False):
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    source = yaml.safe_load(Path(spec["source_config"]).read_text(encoding="utf-8"))
    root = Path(spec["output_dir"]); root.mkdir(parents=True, exist_ok=True)
    if phase in {"build", "all"}:
        for variant in spec["variants"]:
            for dataset in spec["datasets"]:
                print(f"build {variant['id']}/{dataset}", flush=True)
                build_dataset(_variant_config(source, spec, variant, dataset), dataset, overwrite)
    if phase in {"direct", "all"}:
        config = _cfg(source)
        if "kmeans_n_init" in spec:
            config.task2.kmeans_n_init = int(spec["kmeans_n_init"])
        variant_ids = [value["id"] for value in spec["variants"]]
        rows = []
        for variant in spec["variants"]:
            for dataset in spec["datasets"]:
                if spec.get("comparison_population", "common") == "common":
                    records = _load_common_records(root / "cache", variant_ids,
                                                   variant["id"], dataset)
                    valid_fraction = float(np.mean([
                        value["metadata"]["common_valid_fraction"] for value in records
                    ]))
                else:
                    records = _load_slices(root / "cache" / variant["id"] / dataset)
                    valid_fraction = float(np.mean([
                        value["metadata"]["valid_primitives"]
                        / value["metadata"]["total_primitives"] for value in records
                    ]))
                for scope, fit_slices, eval_slices in (
                    ("validation", 6, [6, 7]), ("heldout_test", 8, [8, 9])
                ):
                    for representation in spec["representations"]:
                        cluster, metrics = _score(
                            records, representation, fit_slices, eval_slices, config
                        )
                        rows.append({
                            "variant": variant["id"], "dataset": dataset,
                            "scope": scope, "representation": representation,
                            "cluster_as_vortex": cluster,
                            "valid_fraction": valid_fraction,
                            "comparison_population": spec.get(
                                "comparison_population", "common"
                            ),
                            "dt_scale": variant["dt_scale"],
                            "integration_steps": variant["integration_steps"],
                            "sampled_steps": variant["sampled_steps"],
                            "physical_horizon_frames": (
                                float(variant["dt_scale"]) * int(variant["integration_steps"])
                            ),
                            "offset_mode": variant["offset_mode"],
                            "offset_grid_scale": variant["offset_grid_scale"],
                            "max_spatial_dim": variant["max_spatial_dim"],
                            "seed_grid_shape": "x".join(
                                str(value) for value in variant["seed_grid_shape"]
                            ),
                            **metrics,
                        })
                        print(f"{scope} {variant['id']}/{dataset}/{representation}: "
                              f"F1={metrics['f1']:.4f}", flush=True)
        _write(root / "direct_scores.csv", rows)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_HighReSampling3D_1.1.yaml")
    parser.add_argument("--phase", choices=("build", "direct", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); run(args.config, args.phase, args.overwrite)
