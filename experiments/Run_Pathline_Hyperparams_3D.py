"""Build and evaluate pathline hyperparameters for 3D Task2."""

from __future__ import annotations
import argparse, copy, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits
from DeepUtils.utils import EasyConfig
from Build_Task2_Universality_Cache import build_dataset
from Build_Channel_Killing_Cache import build as build_channel
from FMT_Utils.NetCDF_window_3D import inspect_netcdf_3d, interior_time_indices
from Run_Task2_Universality import _fit_cluster, _load_slices, _prepare
from Train_3DFMT_VAE import _train_vae


def _cfg(payload):
    value = EasyConfig(); value.update(payload); return value


def _load_common_slices(cache_root, variant_ids, variant_id, dataset):
    """Load one variant restricted to seeds valid for every compared variant."""
    records = _load_slices(Path(cache_root) / variant_id / dataset)
    paths_by_variant = {
        other_id: sorted((Path(cache_root) / other_id / dataset).glob("slice_*.npz"))
        for other_id in variant_ids
    }
    current_paths = paths_by_variant[variant_id]
    if len(current_paths) != len(records):
        raise RuntimeError(f"cache file count changed while loading {variant_id}/{dataset}")
    for ordinal, (record, current_path) in enumerate(zip(records, current_paths)):
        masks, references, source_indices = [], [], []
        for other_id in variant_ids:
            matches = paths_by_variant[other_id]
            if len(matches) != len(records):
                raise RuntimeError(f"expected {len(records)} slices for {other_id}/{dataset}")
            with np.load(matches[ordinal]) as data:
                masks.append(np.asarray(data["valid_mask"], dtype=bool))
                references.append(np.asarray(data["reference"], dtype=bool))
                metadata = json.loads(str(data["metadata_json"]))
                source_indices.append(metadata["source_start_index"])
        if len(set(source_indices)) != 1:
            raise RuntimeError(f"source-time mismatch for {dataset} slice {ordinal}: {source_indices}")
        common_mask = np.logical_and.reduce(masks)
        with np.load(current_path) as data:
            current_mask = np.asarray(data["valid_mask"], dtype=bool)
        if common_mask.shape != current_mask.shape:
            raise RuntimeError(f"valid-mask shape mismatch for {variant_id}/{dataset} slice {ordinal}")
        selector = common_mask[current_mask]
        if int(selector.sum()) < 100:
            raise RuntimeError(f"only {int(selector.sum())} common seeds for {dataset} slice {ordinal}")
        for key in ("raw", "fmt", "reference"):
            record[key] = record[key][selector]
        common_references = [reference[common_mask[mask]]
                             for reference, mask in zip(references, masks)]
        if not all(np.array_equal(common_references[0], other)
                   for other in common_references[1:]):
            raise RuntimeError(f"IVD reference mismatch for {dataset} slice {ordinal}")
        record["metadata"] = dict(record["metadata"])
        record["metadata"]["common_valid_primitives"] = int(selector.sum())
        record["metadata"]["common_valid_fraction"] = float(common_mask.mean())
    return records


def _screen(config, dataset, seed, cache_root, variant_ids, variant_id):
    started = time.perf_counter()
    records = _load_common_slices(cache_root, variant_ids, variant_id, dataset)
    n_train = int(config.task2.train_slice_count)
    lengths = [len(r["reference"]) for r in records[:n_train]]
    reference = np.concatenate([r["reference"] for r in records[n_train:]])
    values = np.concatenate([r["fmt"] for r in records])
    train_x, test_x = _prepare(values, lengths, float(config.task2.fmt_neighbor_weight))
    with threadpool_limits(limits=1):
        _, _, direct = _fit_cluster(train_x, test_x, reference, config, already_scaled=True)
    batches = math.ceil(len(train_x) / int(config.task2.batch_size))
    settings = config.task2.dict(); settings["epochs"] = math.ceil(
        int(config.task2.target_optimizer_steps) / batches)
    vae_config = EasyConfig(); vae_config.update({"vae": settings, "evaluation": {
        "kmeans_seed": int(config.task2.kmeans_seed),
        "kmeans_n_init": int(config.task2.kmeans_n_init)}})
    train_mu, test_mu, _, losses = _train_vae(
        train_x, test_x, vae_config, int(seed),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with threadpool_limits(limits=1):
        _, _, vae = _fit_cluster(train_mu, test_mu, reference, config)
    return {
        "fmt_direct_f1": direct["f1"],
        "fmt_vae_f1": vae["f1"],
        "reconstruction": losses["reconstruction"],
        "training_seed": int(seed),
        "train_samples": int(sum(lengths)),
        "test_samples": int(len(reference)),
        "common_valid_fraction": float(np.mean([
            record["metadata"]["common_valid_fraction"] for record in records
        ])),
        "train_seconds": time.perf_counter() - started,
    }


def _direct(config, dataset, cache_root, variant_ids, variant_id):
    """Deterministically recompute direct FMT clustering on the common population."""
    started = time.perf_counter()
    records = _load_common_slices(cache_root, variant_ids, variant_id, dataset)
    n_train = int(config.task2.train_slice_count)
    lengths = [len(record["reference"]) for record in records[:n_train]]
    reference = np.concatenate([record["reference"] for record in records[n_train:]])
    values = np.concatenate([record["fmt"] for record in records])
    train_x, test_x = _prepare(values, lengths, float(config.task2.fmt_neighbor_weight))
    with threadpool_limits(limits=1):
        _, _, score = _fit_cluster(train_x, test_x, reference, config, already_scaled=True)
    return {"fmt_direct_f1": score["f1"], "train_samples": int(sum(lengths)),
            "test_samples": int(len(reference)),
            "direct_seconds": time.perf_counter() - started}


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def run(spec_path, phase, only_variants=None, only_datasets=None, overwrite_cache=False):
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    source = yaml.safe_load(Path(spec["source_config"]).read_text(encoding="utf-8"))
    if overwrite_cache and phase != "build":
        raise ValueError("--overwrite-cache is only allowed with --phase build")
    if phase == "all":
        for subphase in ("build", "train", "direct", "confirm"):
            run(spec_path, subphase, only_variants, only_datasets, overwrite_cache=False)
        return
    root = Path(spec["output_dir"]); rows = []
    variant_ids = [variant["id"] for variant in spec["variants"]]
    selected_variants = [variant for variant in spec["variants"]
                         if not only_variants or variant["id"] in only_variants]
    unknown = set(only_variants or ()) - set(variant_ids)
    if unknown:
        raise ValueError(f"unknown variant ids: {sorted(unknown)}")
    selected_datasets = [dataset for dataset in spec["datasets"]
                         if not only_datasets or dataset in only_datasets]
    unknown_datasets = set(only_datasets or ()) - set(spec["datasets"])
    if unknown_datasets:
        raise ValueError(f"unknown dataset ids: {sorted(unknown_datasets)}")
    for variant in selected_variants:
        cache_root = root / "cache" / variant["id"]
        result_root = root / "results_common" / variant["id"]
        payload = copy.deepcopy(source)
        payload["experiment"] = spec["experiment"]
        payload["pathlines"].update({k: variant[k] for k in
                                     ("dt_scale", "integration_steps", "sampled_steps")})
        payload["task2"]["training_seeds"] = [spec["screening_seed"]]
        payload["output"] = {"cache_dir": str(cache_root), "result_dir": str(result_root)}
        for dataset in selected_datasets:
            entry = next(item for item in source["datasets"] if item["id"] == dataset)
            total_frames = (int(source["channel_observer"]["total_frames"])
                            if dataset == "channel" else
                            int(inspect_netcdf_3d(entry["path"])["shape"]["t"]))
            max_future = max(math.ceil(float(v["dt_scale"]) * int(v["integration_steps"]))
                             for v in spec["variants"])
            fixed = interior_time_indices(
                total_frames, int(source["sampling"]["timeslices"]),
                float(source["sampling"]["begin_fraction"]),
                float(source["sampling"]["end_fraction"]), required_future_frames=max_future + 1,
            ).tolist()
            dataset_payload = copy.deepcopy(payload)
            dataset_payload["sampling"]["fixed_time_indices"] = fixed
            config = _cfg(dataset_payload)
            if phase == "build":
                if dataset == "channel": build_channel(config, overwrite=overwrite_cache)
                else: build_dataset(config, dataset, overwrite=overwrite_cache)
            if phase == "train":
                result_path = result_root / dataset / "screen.json"
                if result_path.exists(): result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = _screen(config, dataset, spec["screening_seed"],
                                     root / "cache", variant_ids, variant["id"])
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                rows.append({"variant": variant["id"], "dataset": dataset,
                    "dt_scale": variant["dt_scale"], "integration_steps": variant["integration_steps"],
                    "sampled_steps": variant["sampled_steps"],
                    **result})
            if phase == "direct":
                result_path = result_root / dataset / "direct.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = _direct(config, dataset, root / "cache", variant_ids, variant["id"])
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                rows.append({"variant": variant["id"], "dataset": dataset,
                    "dt_scale": variant["dt_scale"],
                    "integration_steps": variant["integration_steps"],
                    "sampled_steps": variant["sampled_steps"], **result})
            if phase == "confirm":
                for seed in spec["confirmation_seeds"]:
                    result_path = result_root / dataset / f"seed_{int(seed)}.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    else:
                        result = _screen(config, dataset, int(seed), root / "cache",
                                         variant_ids, variant["id"])
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    rows.append({"variant": variant["id"], "dataset": dataset,
                        "dt_scale": variant["dt_scale"],
                        "integration_steps": variant["integration_steps"],
                        "sampled_steps": variant["sampled_steps"], **result})
    if rows:
        base_name = {"train": "screening", "confirm": "confirmation",
                     "direct": "direct", "all": "all_runs"}[phase]
        suffix = "" if not only_variants else "." + "-".join(sorted(only_variants))
        if only_datasets:
            suffix += ".datasets-" + "-".join(sorted(only_datasets))
        name = f"{base_name}{suffix}.csv"
        _write_csv(root / name, rows)
        print(f"wrote {len(rows)} rows to {root / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_PathlineHyperparams3D_1.1.yaml")
    parser.add_argument("--phase", choices=("build", "train", "confirm", "direct", "all"),
                        default="all")
    parser.add_argument("--variant", action="append",
                        help="run only this variant; repeat to select more than one")
    parser.add_argument("--dataset", action="append",
                        help="run only this dataset; repeat to select more than one")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="rebuild cache files; only valid with --phase build")
    args = parser.parse_args()
    run(args.config, args.phase, args.variant, args.dataset, args.overwrite_cache)
