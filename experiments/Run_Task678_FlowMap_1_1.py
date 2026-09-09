"""Paired Task6/7 decoders and actual Task8 composition; never save checkpoints."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from FMT_Utils.FlowMapData_3D import affine_query, trajectory_metrics, sha256, write_json
from FMT_Utils.FlowMapModels_3D import (TokenTransform, training_arrays, train_decoder,
                                      predict, compose)
from experiments.Build_Task678_FlowMap_1_1 import DEFAULT_CONFIG, provenance


def load_role(spec, dataset, role):
    root = Path(spec["output_root"]) / "cache" / dataset
    records = []
    for i in spec["splits"][role]:
        path = root / f"window_{i:02d}.npz"
        fp = root / f"window_{i:02d}_features.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        fm = json.loads(fp.with_suffix(".json").read_text())
        if meta["cache_sha256"] != sha256(path) or fm["feature_sha256"] != sha256(fp):
            raise ValueError("Changed cached data")
        if fm["cache_sha256"] != meta["cache_sha256"] or meta["role"] != role:
            raise ValueError("Changed source identities/splits")
        if meta["config_sha256"] != spec["_config_sha256"] or fm["config_sha256"] != spec["_config_sha256"]:
            raise ValueError("Cache config mismatch")
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files if k != "metadata_json"}
        records.append(dict(data=data, feature_file=fp, metadata=meta))
    return records


def stack_data(records):
    return {k: np.concatenate([r["data"][k] for r in records]) for k in records[0]["data"]}


def source_features(records, arm):
    blocks = {k: [] for k in ("support0", "support1", "context")}
    for r in records:
        with np.load(r["feature_file"], allow_pickle=False) as a:
            for k in blocks:
                blocks[k].append(a[f"{arm}__{k}"])
    return {k: np.concatenate(v) for k, v in blocks.items()}


def fit_transform(train_features, arm, task, width, seed, spec, device):
    if task == "Task6":
        rows = np.concatenate([train_features["support0"], train_features["support1"]])
    else:
        rows = train_features["context"].reshape(-1, train_features["context"].shape[-1])
    return TokenTransform(width, seed, arm, device).fit(rows, spec["vae"])


def tokens(transform, features, task):
    result = {}
    for k in (("support0", "support1") if task == "Task6" else ("context",)):
        shape = features[k].shape
        result[k] = transform.transform(features[k].reshape(-1, shape[-1])).reshape(*shape[:-1], transform.width)
    return result


def interpolation(data, task):
    tau = np.linspace(0, 1, data["target0"].shape[2])
    query = (data["target0"][:, :, 0] - data["origin0"][:, None]) / data["radius0"][:, None, None]
    if task in ("Task6", "Task8"):
        first = affine_query(data["support0"], data["origin0"], data["radius0"], query, tau)
        if task == "Task6":
            q1 = (data["target1"][:, :, 0] - data["origin1"][:, None]) / data["radius1"][:, None, None]
            second = affine_query(data["support1"], data["origin1"], data["radius1"], q1, tau)
            return np.concatenate([first, second])
        query_second = (first[:, :, -1] - data["origin1"][:, None]) / data["radius1"][:, None, None]
        second = affine_query(data["support1"], data["origin1"], data["radius1"], query_second, tau)
        return np.concatenate([first, second[:, :, 1:]], 2)
    points = data["target0"][:, :, 0]
    predictions, weights = [], []
    for s in range(6):
        origin = data["context_origins"][:, s]
        q = (points - origin[:, None]) / data["radius0"][:, None, None]
        predictions.append(affine_query(data["context"][:, s], origin, data["radius0"], q, tau))
        weights.append(1. / np.maximum(np.linalg.norm(points - origin[:, None], axis=-1), 1e-12) ** 2)
    weight = np.stack(weights, 0)
    weight /= weight.sum(0)
    return (np.stack(predictions, 0) * weight[:, :, :, None, None]).sum(0)


def truth_radius(data, task):
    if task == "Task6":
        return np.concatenate([data["target0"], data["target1"]]), np.concatenate([data["radius0"], data["radius1"]])
    return data["target_long" if task == "Task8" else "target0"], data["radius0"]


def save_evaluation(directory, task, arm, prediction, data, records, common, extra):
    # Score exactly the durable numeric representation audited later.
    prediction = np.asarray(prediction, np.float32)
    truth, radii = truth_radius(data, task)
    metrics = trajectory_metrics(prediction, truth, radii)
    n = len(data["origin0"])
    counts = [len(r["data"]["origin0"]) for r in records]
    per_window, offset = [], 0
    for r, count in zip(records, counts):
        ix = np.arange(offset, offset + count)
        if task == "Task6":
            ix = np.concatenate([ix, n + ix])
        per_window.append({"ordinal": r["metadata"]["ordinal"],
            "metrics": trajectory_metrics(prediction[ix], truth[ix], radii[ix])})
        offset += count
    path = directory / f"{task}_{arm}_predictions.npz"
    np.savez_compressed(path, prediction=prediction.astype(np.float32))
    value = {**common, **extra, "task": task, "arm": arm, "metrics": metrics,
             "per_window": per_window, "prediction_file": path.name, "prediction_sha256": sha256(path),
             "test_sources": [{"ordinal": r["metadata"]["ordinal"], "cache_sha256": r["metadata"]["cache_sha256"]} for r in records]}
    write_json(directory / f"{task}_{arm}.json", value)
    print(f"RESULT {task}/{arm} position={metrics['position_nrmse']:.6g} shape={metrics['centered_shape_error']:.6g}", flush=True)
    return value


def run(spec, config, dataset, seed, width, device="cuda"):
    root = Path(spec["output_root"])
    spec["_config_sha256"] = sha256(config)
    torch.set_num_threads(int(os.getenv("SLURM_CPUS_PER_TASK", "4")))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Production comparison requires the allocated GPU")
    build = json.loads((root / "build" / f"{dataset}.json").read_text())
    if build["status"] != "PASS" or build["config_sha256"] != sha256(config):
        raise ValueError("Data build must pass before training")
    base = root / "shards" / dataset / f"seed{seed}_width{width}"
    attempt = 0
    while (base / f"attempt_{attempt}").exists():
        if (base / f"attempt_{attempt}" / "completed.json").exists():
            raise FileExistsError("Already completed; do not reopen test for a duplicate run")
        attempt += 1
    directory = base / f"attempt_{attempt}"
    directory.mkdir(parents=True)
    started = {**provenance(config), "experiment": spec["experiment"], "dataset": dataset,
               "seed": seed, "token_width": width, "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
               "torch": torch.__version__, "numpy": np.__version__}
    write_json(directory / "started.json", started)
    try:
        # No test arrays are opened until all train-only fits are done for an arm.
        train = load_role(spec, dataset, "train")
        validation = load_role(spec, dataset, "validation")
        train_data, validation_data = stack_data(train), stack_data(validation)
        results = []
        for arm in spec["arms"]:
            feature_train = source_features(train, arm)
            feature_val = source_features(validation, arm)
            fitted = {}
            for task in ("Task6", "Task7"):
                transform = fit_transform(feature_train, arm, task, width, seed, spec, device)
                arrays = training_arrays(train_data, tokens(transform, feature_train, task), task)
                model, training = train_decoder(arrays, width, spec["decoder"], seed + (7 if task == "Task7" else 6), device)
                val_arrays = training_arrays(validation_data, tokens(transform, feature_val, task), task)
                val_prediction = predict(model, val_arrays, device)
                vt, vr = truth_radius(validation_data, task)
                val_metrics = trajectory_metrics(val_prediction, vt, vr)
                fitted[task] = (transform, model, {**transform.metadata(), **training,
                    "validation_metrics": val_metrics, "selection": "fixed steps; validation not used to select epoch or arm"})
                print(f"TRAINED {dataset}/{seed}/{width}/{task}/{arm}", flush=True)
            test = load_role(spec, dataset, "test")
            test_data = stack_data(test)
            ft = source_features(test, arm)
            for task in ("Task6", "Task7"):
                transform, model, info = fitted[task]
                begin = time.perf_counter()
                test_tokens = tokens(transform, ft, task)
                arrays = training_arrays(test_data, test_tokens, task)
                prediction = predict(model, arrays, device)
                context_count = 1 if task == "Task6" else 6
                extra = {**info, "query_seconds": time.perf_counter() - begin,
                         "context_tokens": context_count,
                         "token_payload_bytes_per_query_region": info["token_bytes"] * context_count,
                         "common_metadata_bytes_per_query_region": 4 * (5 + (18 if task == "Task7" else 0)),
                         "decoder_bytes": info["parameter_count"] * 4, "checkpoint_files": 0}
                results.append(save_evaluation(directory, task, arm, prediction, test_data, test, started, extra))
                if task == "Task6":
                    begin = time.perf_counter()
                    long_prediction, composition_info = compose(model, arrays, device)
                    results.append(save_evaluation(directory, "Task8", arm, long_prediction, test_data, test, started,
                        {**extra, **composition_info, "query_seconds": time.perf_counter() - begin,
                         "context_tokens": 2, "token_payload_bytes_per_query_region": info["token_bytes"] * 2,
                         "common_metadata_bytes_per_query_region": 40,
                         "training_source": "same in-memory Task6 decoder; no extra long-target training"}))
            del fitted
            if device == "cuda":
                torch.cuda.empty_cache()
        # Actual-storage interpolation baselines, evaluated once per paired shard.
        for task in ("Task6", "Task7", "Task8"):
            begin = time.perf_counter()
            prediction = interpolation(test_data, task)
            c = {"Task6": 1, "Task7": 6, "Task8": 2}[task]
            results.append(save_evaluation(directory, task, "affine_interp7", prediction, test_data, test, started,
                {"parameter_count": 0, "token_bytes": 7 * 32 * 3 * 4, "context_tokens": c,
                 "token_payload_bytes_per_query_region": c * 7 * 32 * 3 * 4,
                 "common_metadata_bytes_per_query_region": {"Task6": 20, "Task7": 92, "Task8": 40}[task],
                 "query_seconds": time.perf_counter() - begin, "budget_note": "raw support storage, not dimension matched"}))
        write_json(directory / "completed.json", {**started, "status": "PASS", "completed": provenance(config),
                   "metric_records": len(results), "checkpoint_files": 0, "arms": spec["arms"] + ["affine_interp7"]})
    except Exception:
        write_json(directory / "failed.json", {**started, "status": "FAILED", "traceback": traceback.format_exc()})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--index", type=int)
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    spec = json.loads(Path(args.config).read_text())
    matrix = [(d, seed, width) for d in spec["datasets"] for seed in spec["seeds"] for width in spec["token_dimensions"]]
    d, seed, width = matrix[args.index] if args.index is not None else (args.dataset, args.seed, args.width)
    if (d, seed, width) not in matrix:
        raise ValueError("Run is not part of the frozen comparison matrix")
    run(spec, args.config, d, seed, width, args.device)


if __name__ == "__main__":
    main()
