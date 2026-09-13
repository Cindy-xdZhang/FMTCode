"""Prepare, train, summarize and deploy the versioned Channel binary experiment."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

DEFAULT_CONFIG = "config/mainExp_Task4C_HairpinBinary_2.1.json"
SPLITS = ("train", "validation", "test")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def identity(config):
    sources = [__file__, "FMT_Utils/Task4C_HairpinBinary_2_1.py", "FMT_Utils/DFT_FMT_3D.py"]
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {str(Path(p).name): sha(p) for p in sources}, "config_sha256": sha(config),
            "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
            "array_index": os.environ.get("SLURM_ARRAY_TASK_ID")}


def prepare(spec, config, input_root=None):
    import torch
    from FMT_Utils.Task4C_HairpinBinary_2_1 import (
        load_channel, read_dataset, gt_instance_bounds, sample_gt, trace_cross, fmt_features,
    )
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f"{split}.npz").exists() for split in SPLITS):
        raise FileExistsError("A prepared or partial cache exists; do not overwrite it")
    root = Path(input_root or spec["input_root"])
    flow_path, gt_path = root / spec["flow"], root / spec["gt"]
    for key, path in (("flow", flow_path), ("gt", gt_path)):
        if sha(path) != spec[f"{key}_sha256"]:
            raise ValueError(f"{key} differs from the user-supplied frozen Channel source")
    print("Reading native Channel geometry and GT", flush=True)
    axes, scalar, interpolator = load_channel(flow_path, spec["curves"]["vector"])
    gt = read_dataset(gt_path)
    instances, gt_low, gt_high, _ = gt_instance_bounds(gt)
    xmin, xmax = float(axes[0][0]), float(axes[0][-1])
    cuts = xmin + np.array(spec["split"]["x_fractions"]) * (xmax-xmin)
    assignment = np.full(len(instances), -1, np.int8)
    for split in range(3):
        inside = (gt_low[:, 0] >= cuts[split]) & (gt_high[:, 0] <= cuts[split+1])
        assignment[inside] = split
    owner = dict(zip(instances.tolist(), assignment.tolist()))
    wall_span = float(axes[2][-1]-axes[2][0])
    options = spec["curves"]
    step = wall_span * options["step_wall_span"]
    neighbor = wall_span * options["neighbor_radius_wall_span"]
    support = neighbor + options["steps_per_direction"] * step
    cell_axes = [(axis[1:]+axis[:-1])/2 for axis in axes]
    margin = support + np.array([np.diff(a).max() for a in axes])
    report = {"version": spec["version"], "identity": identity(config), "config": spec,
              "source_paths": {"flow": str(flow_path), "gt": str(gt_path)},
              "cuts_x": cuts.tolist(), "primitive_support_radius": support, "seed_margin_xyz": margin.tolist(),
              "native_dimensions_xyz": [len(a) for a in axes], "instances": [
                  {"id": int(i), "low": low.tolist(), "high": high.tolist(),
                   "split": SPLITS[int(s)] if s >= 0 else "excluded_crossing_boundary"}
                  for i, low, high, s in zip(instances, gt_low, gt_high, assignment)], "splits": {}}
    locator = None
    ranges = []
    for split, name in enumerate(SPLITS):
        rng = np.random.default_rng(spec["sampling"]["seed"] + split)
        mx = (cell_axes[0] > cuts[split]+margin[0]) & (cell_axes[0] < cuts[split+1]-margin[0])
        my = (cell_axes[1] > axes[1][0]+margin[1]) & (cell_axes[1] < axes[1][-1]-margin[1])
        mz = (cell_axes[2] > axes[2][0]+margin[2]) & (cell_axes[2] < axes[2][-1]-margin[2])
        available = (scalar < spec["lambda2_threshold"]) & mx[None, None, :] & my[None, :, None] & mz[:, None, None]
        candidates = np.flatnonzero(available)
        population = len(candidates)
        cap = spec["sampling"]["scan_cap_per_split"]
        if len(candidates) > cap:
            candidates = np.sort(rng.choice(candidates, cap, replace=False))
        iz, iy, ix = np.unravel_index(candidates, scalar.shape)
        points = np.column_stack((cell_axes[0][ix], cell_axes[1][iy], cell_axes[2][iz]))
        ids, locator = sample_gt(gt, points, locator)
        # Whole instances crossing either fixed cut are excluded, never relabeled.
        permitted = np.array([i < 0 or owner[int(i)] == split for i in ids])
        excluded = int((~permitted).sum())
        candidates, points, ids = candidates[permitted], points[permitted], ids[permitted]
        cap = spec["sampling"]["caps"][name]
        if len(candidates) > cap:
            take = np.sort(rng.choice(len(candidates), cap, replace=False))
            candidates, points, ids = candidates[take], points[take], ids[take]
        labels = (ids >= 0).astype(np.int8)
        if np.bincount(labels, minlength=2).min() < spec["sampling"]["minimum_class_count"]:
            raise ValueError(f"{name} lacks enough classes under the frozen spatial split")
        print(f"{name}: selected {len(points)} of {population} vortex cells", flush=True)
        chunks, all_valid, all_low, all_high = [], [], [], []
        for start in range(0, len(points), options["chunk_size"]):
            selected = points[start:start+options["chunk_size"]]
            curves, valid, low, high = trace_cross(interpolator, selected, step, neighbor, options["steps_per_direction"])
            if valid.any():
                if low[valid, 0].min() <= cuts[split] or high[valid, 0].max() >= cuts[split+1]:
                    raise ValueError("An integration query crossed the fixed spatial partition")
            chunks.append(curves)
            all_valid.append(valid)
            all_low.append(low)
            all_high.append(high)
        geometry, valid = np.concatenate(chunks), np.concatenate(all_valid)
        low, high = np.concatenate(all_low), np.concatenate(all_high)
        if valid.mean() < options["minimum_valid_fraction"]:
            write(out / f"{name}_preparation_failure.json", {"selected": len(valid), "valid": int(valid.sum())})
            raise ValueError("Too many invalid primitives; preserve the failed preparation rather than silently dropping them")
        features = np.concatenate([fmt_features(geometry[valid][i:i+1024], spec["fmt"]).numpy()
                                   for i in range(0, int(valid.sum()), 1024)])
        # Separate split files prevent accidental test loading during training/selection.
        path = out / f"{name}.npz"
        np.savez_compressed(path, geometry=geometry[valid], fmt=features, labels=labels[valid],
            source=candidates[valid], points=points[valid], instance=ids[valid],
            query_low=low[valid], query_high=high[valid])
        np.savez_compressed(out / f"{name}_candidates.npz", source=candidates, points=points, labels=labels,
                            instance=ids, valid=valid)
        lower_node = int(np.searchsorted(axes[0], low[valid, 0].min(), side="right")-1)
        upper_node = int(np.searchsorted(axes[0], high[valid, 0].max(), side="left"))
        ranges.append((lower_node, upper_node))
        report["splits"][name] = {"sha256": sha(path), "candidate_sha256": sha(out / f"{name}_candidates.npz"),
            "vortex_population_after_support_margin": population, "sampled_crossing_instance_points_excluded": excluded,
            "selected": len(points), "valid": int(valid.sum()), "class_counts": np.bincount(labels[valid], minlength=2).tolist(),
            "selected_class_counts": np.bincount(labels, minlength=2).tolist(),
            "instance_ids": sorted(int(i) for i in np.unique(ids[valid]) if i >= 0),
            "native_query_x_nodes": [lower_node, upper_node]}
    if not (ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0]):
        raise ValueError("Native interpolation nodes overlap across splits")
    sets = [set(report["splits"][s]["instance_ids"]) for s in SPLITS]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("A complete GT instance appears in more than one split")
    report.update(spatial_isolation_verified=True, completed_at=datetime.now(timezone.utc).isoformat())
    write(out / "preparation.json", report)
    print("Preparation completed with disjoint native interpolation supports and instance IDs", flush=True)


def load_split(out, name, manifest):
    path = Path(out) / f"{name}.npz"
    if sha(path) != manifest["splits"][name]["sha256"]:
        raise ValueError(f"Frozen {name} cache changed")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def choose_threshold(labels, probability):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, probability)
    f1 = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)
    # Deterministic ties choose the largest threshold (fewer positive predictions).
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def binary_metrics(labels, probability, threshold):
    from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score, confusion_matrix
    predicted = probability >= threshold
    return {"f1": float(f1_score(labels, predicted, zero_division=0)),
            "average_precision": float(average_precision_score(labels, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
            "class_counts": np.bincount(labels, minlength=2).tolist()}


def run_plan(spec):
    return [(method, seed) for seed in spec["training"]["seeds"] for method in spec["methods"]]


def train(spec, config, index):
    import torch
    from torch import nn
    from sklearn.metrics import average_precision_score
    from FMT_Utils.Task4C_HairpinBinary_2_1 import FMTClassifier, Conv3DClassifier, geometry_voxels
    out = Path(spec["output"])
    manifest = json.loads((out / "preparation.json").read_text())
    if manifest["identity"]["config_sha256"] != sha(config) or not manifest["spatial_isolation_verified"]:
        raise ValueError("Preparation is not certified for this config")
    method, seed = run_plan(spec)[index]
    destination = out / "runs" / f"{method}_seed{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    started = time.time()
    training, validation = load_split(out, "train", manifest), load_split(out, "validation", manifest)
    options = spec["training"]
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and options["require_cuda_on_ibex"] and device.type != "cuda":
        raise RuntimeError("Allocated training job cannot access CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = (FMTClassifier(options["dropout"]) if method == "fmt_mlp" else Conv3DClassifier(options["dropout"])).to(device)
    mean = training["fmt"].mean(0, dtype=np.float64)
    std = training["fmt"].std(0, dtype=np.float64)
    std[std < 1e-8] = 1
    np.savez_compressed(destination / "train_feature_statistics.npz", mean=mean, std=std)
    for data in (training, validation):
        data["standardized_fmt"] = ((data["fmt"]-mean)/std).astype(np.float32)
    count = np.bincount(training["labels"], minlength=2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(count[0]/count[1], device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, options["epochs"], eta_min=options["minimum_learning_rate"])
    batch_size = options["batch_size"]

    def features(data, chosen):
        if method == "fmt_mlp":
            return torch.as_tensor(data["standardized_fmt"][chosen], device=device)
        geometry = torch.as_tensor(data["geometry"][chosen], device=device)
        return geometry_voxels(geometry, spec["conv3d"]["voxel_resolution"], spec["conv3d"]["segment_subdivisions"])

    def predict(data):
        model.eval()
        probability = []
        with torch.no_grad():
            for offset in range(0, len(data["labels"]), batch_size):
                chosen = slice(offset, offset+batch_size)
                probability.append(model(features(data, chosen)).sigmoid().cpu().numpy())
        return np.concatenate(probability)

    best_ap, best_epoch, best_state, history = -1., 0, None, []
    for epoch in range(1, options["epochs"]+1):
        model.train()
        order = rng.permutation(len(training["labels"]))
        total_loss = 0.
        for offset in range(0, len(order), batch_size):
            ids = order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features(training, ids))
            loss = loss_fn(logits, torch.as_tensor(training["labels"][ids], device=device, dtype=torch.float32))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"])
            optimizer.step()
            total_loss += float(loss.detach())*len(ids)
        scheduler.step()
        probabilities = predict(validation)
        ap = float(average_precision_score(validation["labels"], probabilities))
        history.append({"epoch": epoch, "training_loss": total_loss/len(order), "validation_average_precision": ap})
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"method": method, "seed": seed, **history[-1]}), flush=True)
        if epoch-best_epoch >= options["patience"]:
            break
    model.load_state_dict(best_state)
    validation_probability = predict(validation)
    threshold = choose_threshold(validation["labels"], validation_probability)
    # First test read occurs strictly AFTER model and decision threshold selection.
    test = load_split(out, "test", manifest)
    test["standardized_fmt"] = ((test["fmt"]-mean)/std).astype(np.float32)
    probabilities = predict(test)
    metrics = binary_metrics(test["labels"], probabilities, threshold)
    np.savez_compressed(destination / "predictions.npz", labels=test["labels"], probability=probabilities,
        source=test["source"], instance=test["instance"], points=test["points"],
        validation_labels=validation["labels"], validation_probability=validation_probability)
    result = {"version": spec["version"], "method": method, "seed": seed, "identity": identity(config),
        "config": spec, "preparation_sha256": sha(out / "preparation.json"),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.time()-started,
        "selected_epoch": best_epoch, "threshold": threshold, "best_validation_average_precision": best_ap,
        "validation": binary_metrics(validation["labels"], validation_probability, threshold),
        "test": metrics, "history": history, "predictions_sha256": sha(destination / "predictions.npz"),
        "class_names": ["non_hairpin", "hairpin"], "weights_written": 0,
        "scope": "GT-membership on valid local primitives at sampled vortex-cell centers; single Channel crop"}
    write(destination / "result.json", result)
    print(f"Completed {method} seed {seed}; test metrics recorded for complete-array aggregation", flush=True)


def merge(spec, config):
    out = Path(spec["output"])
    records = []
    for method, seed in run_plan(spec):
        folder = out / "runs" / f"{method}_seed{seed}"
        record = json.loads((folder / "result.json").read_text())
        if record["identity"]["config_sha256"] != sha(config) or sha(folder / "predictions.npz") != record["predictions_sha256"]:
            raise ValueError("Run config or prediction bytes changed")
        with np.load(folder / "predictions.npz") as data:
            recomputed = binary_metrics(data["labels"], data["probability"], record["threshold"])
            if choose_threshold(data["validation_labels"], data["validation_probability"]) != record["threshold"]:
                raise ValueError("Decision threshold differs from validation selection")
        if recomputed != record["test"]:
            raise ValueError("Saved predictions disagree with reported metrics")
        records.append({"method": method, "seed": seed, "parameters": record["parameters"],
                        "selected_epoch": record["selected_epoch"], "seconds": record["seconds"], **record["test"]})
    summary = {}
    for method in spec["methods"]:
        selected = [r for r in records if r["method"] == method]
        summary[method] = {key: {"mean": float(np.mean([r[key] for r in selected])),
                                "sample_std": float(np.std([r[key] for r in selected], ddof=1)) if len(selected) > 1 else None}
                           for key in ("f1", "average_precision", "balanced_accuracy")}
    paired = []
    for seed in spec["training"]["seeds"]:
        pair = {r["method"]: r for r in records if r["seed"] == seed}
        paired.append({"seed": seed, **{key: pair["fmt_mlp"][key]-pair["conv3d_mlp"][key]
                                      for key in ("f1", "average_precision")}})
    write(out / "results.json", {"version": spec["version"], "runs": records, "summary": summary,
            "paired_fmt_minus_conv3d": paired, "all_runs_complete": True, "prediction_consistency_checked": True})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "seed", "parameters", "selected_epoch", "seconds", "f1", "average_precision", "balanced_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary), flush=True)


def append_locked(path, message):
    import fcntl
    with Path(path).open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(message)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def runtime(spec, config, phase, state, code=None):
    out = Path(spec["output"])
    device = "CPU"
    if phase == "train":
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    row = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase, "state": state,
           "exit_code": code, "device": device, **identity(config)}
    append_locked(out / "runtime_events.jsonl", json.dumps(row) + "\n")
    append_locked("docs/ibex_run_registry.md", f'\n- Task4-c {spec["version"]} runtime: ' + json.dumps(row) + "\n")


def submit(spec, config):
    out = Path(spec["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    write(out / "config.frozen.json", spec)
    previous = None
    for phase, cpu, memory, limit in (("prepare", 8, 32, "01:00:00"), ("train", 4, 24, "02:00:00"), ("merge", 2, 8, "00:10:00")):
        command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpu}",
            f"--mem={memory}G", f"--time={limit}", f"--job-name=t4c21-{phase}",
            f"--output={out}/logs/{phase}_%A_%a.out", f"--error={out}/logs/{phase}_%A_%a.err"]
        if previous:
            command += [f"--dependency=afterok:{previous}", "--kill-on-invalid-dep=yes"]
        if phase == "train":
            command += [f"--array=0-{len(run_plan(spec))-1}%3", "--gres=gpu:1", "--constraint=a100|v100", "--exclude=gpu203-02-r"]
        command += ["ibex_bash/task4c_hairpin_binary_2p1.sh", phase, config]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        row = {"job_id": job, "version": spec["version"], "phase": phase, "command": command,
               "submitted_at_utc": datetime.now(timezone.utc).isoformat(), "expected_device": "A100/V100" if phase == "train" else "CPU",
               "dependency": previous, **identity(config)}
        append_locked(out / "submissions.jsonl", json.dumps(row) + "\n")
        append_locked("docs/ibex_run_registry.md", "\n- Task4-c SUBMITTED " + json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        previous = job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "train", "merge", "submit", "runtime"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input-root")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--runtime-phase")
    parser.add_argument("--state")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.phase == "prepare":
        prepare(spec, args.config, args.input_root)
    elif args.phase == "train":
        train(spec, args.config, args.index)
    elif args.phase == "runtime":
        runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else:
        {"merge": merge, "submit": submit}[args.phase](spec, args.config)


if __name__ == "__main__":
    main()
