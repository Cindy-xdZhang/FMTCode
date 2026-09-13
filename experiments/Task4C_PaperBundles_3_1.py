"""Prepare Channel+TBL paper bundles, encode, compare FMT/Conv3D, and deploy."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from experiments.Task4C_HairpinBinary_2_1 import (
    sha, write, append_locked, choose_threshold, binary_metrics, run_plan,
)

DEFAULT_CONFIG = "config/mainExp_Task4C_PaperBundles_3.1.json"
SPLITS = ("train", "validation", "test")


def identity(config):
    sources = ["experiments/Task4C_PaperBundles_3_1.py", "FMT_Utils/Task4C_PaperBundles_3_1.py",
               "experiments/Task4C_HairpinBinary_2_1.py", "FMT_Utils/Task4C_HairpinBinary_2_1.py",
               "FMT_Utils/Task4C_Bundles_1_1.py", "FMT_Utils/Task4B_CrossFlow_3D.py",
               "FMT_Utils/DFT_FMT_3D.py"]
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: sha(p) for p in sources}, "config_sha256": sha(config),
            "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
            "array_index": os.environ.get("SLURM_ARRAY_TASK_ID")}


def build_head_catalog(axes, head_cells, gt, spec):
    from scipy import ndimage
    from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt, gt_instance_bounds
    from FMT_Utils.Task4C_PaperBundles_3_1 import cell_centers, native_partition
    components, count = ndimage.label(head_cells)  # face-connected on native cells
    flat = np.flatnonzero(head_cells)
    component = components.ravel()[flat]
    order = np.argsort(component, kind="stable")
    flat, component = flat[order], component[order]
    starts = np.r_[0, np.flatnonzero(np.diff(component))+1, len(flat)]
    points = cell_centers(flat, head_cells.shape, axes)
    labels, _ = sample_gt(gt, points)
    ids, low, high, _ = gt_instance_bounds(gt)
    cuts = axes[0][0] + np.array(spec["split"]["x_fractions"])*(axes[0][-1]-axes[0][0])
    owner = {int(i): -1 for i in ids}
    for split in range(3):
        for i in ids[(low[:, 0] >= cuts[split]) & (high[:, 0] <= cuts[split+1])]:
            owner[int(i)] = split
    catalog, excluded = [], []
    for first, stop in zip(starts[:-1], starts[1:]):
        region = int(component[first])
        cell_ids, instance = flat[first:stop], labels[first:stop]
        row = {"head_component": region, "head_cell_count": int(stop-first)}
        if stop-first < spec["bundles"]["minimum_head_cells"]:
            excluded.append({**row, "reason": "fewer_than_10_head_cells"})
            continue
        _, _, x = np.unravel_index(cell_ids, head_cells.shape)
        split_id = -1
        for s in range(3):
            left, right, _ = native_partition(axes, cuts, s, 1)
            if x.min() >= left and x.max() < right:
                split_id = s
                break
        if split_id < 0:
            excluded.append({**row, "reason": "whole_head_crosses_spatial_partition_or_buffer"})
            continue
        positive = instance[instance >= 0]
        instances, counts = np.unique(positive, return_counts=True)
        if any(owner[int(i)] != split_id for i in instances):
            excluded.append({**row, "reason": "GT_instance_crosses_partition", "instances": instances.tolist()})
            continue
        fraction = len(positive)/len(instance)
        dominant = int(instances[counts.argmax()]) if len(instances) else -1
        majority = float(counts.max()/len(instance)) if len(counts) else 0.
        if len(positive) == 0:
            target = 0
        elif majority >= spec["labels"]["positive_minimum_single_instance_fraction"]:
            target = 1
        else:
            excluded.append({**row, "reason": "ambiguous_partial_GT_overlap", "head_GT_fraction": fraction})
            continue
        catalog.append({**row, "split": SPLITS[split_id], "label": target, "instance": dominant,
                        "all_instances": instances.tolist(), "head_GT_fraction": fraction,
                        "dominant_instance_fraction": majority, "cell_ids": cell_ids})
    return catalog, {"head_components": count, "eligible_labeled_heads": len(catalog),
                     "excluded_heads": excluded, "x_cuts": cuts.tolist(),
                     "crossing_GT_instances": sorted(i for i, s in owner.items() if s < 0)}


def sample_views(pools, spec, destination, split, rng):
    from FMT_Utils.Task4C_PaperBundles_3_1 import normalize_view
    total = spec["sampling"]["per_flow_counts"][split]
    if not pools:
        raise ValueError(f"No cleaned heads for {split}")
    selected = rng.permutation(len(pools))[:total].tolist()
    # Repeated samples must be different uniform line subsets, never copies of
    # an undersized padded bundle. Multiple views are NOT independent vortices.
    augmentable = [i for i, pool in enumerate(pools) if len(pool["lines"]) > spec["bundles"]["maximum_lines"]]
    if len(selected) < total and not augmentable:
        raise ValueError("Not enough distinct bundle views; do not duplicate small bundles")
    while len(selected) < total:
        selected.extend(rng.permutation(augmentable)[:total-len(selected)].tolist())
    rng.shuffle(selected)
    folder = destination / split
    folder.mkdir()
    geometry = np.lib.format.open_memmap(folder / "geometry.npy", mode="w+", dtype="float32", shape=(total, 256, 32, 3))
    seeds = np.lib.format.open_memmap(folder / "seeds.npy", mode="w+", dtype="float32", shape=(total, 256, 3))
    counts, labels, regions, instances, centroids, radii, selections = [], [], [], [], [], [], []
    fingerprints = set()
    for j, base in enumerate(selected):
        pool = pools[base]
        available = len(pool["lines"])
        for _ in range(50):
            chosen = np.sort(rng.choice(available, min(256, available), replace=False))
            signature = (pool["head_component"], chosen.tobytes())
            if signature not in fingerprints:
                fingerprints.add(signature)
                break
        else:
            raise ValueError("Duplicate bundle subset; never fill the quota with identical samples")
        g, s, center, radius = normalize_view(pool["lines"][chosen], pool["seeds"][chosen])
        geometry[j], seeds[j] = g, s
        counts.append(len(chosen)); labels.append(pool["label"]); regions.append(pool["head_component"])
        instances.append(pool["instance"]); centroids.append(center); radii.append(radius)
        padded = np.full(256, -1, np.int32); padded[:len(chosen)] = pool["seed_indices"][chosen]
        selections.append(padded)
    geometry.flush(); seeds.flush()
    np.savez_compressed(folder / "metadata.npz", counts=np.asarray(counts, np.int16), labels=np.asarray(labels, np.int8),
        head_component=np.asarray(regions, np.int32), instance=np.asarray(instances, np.int32),
        centroid=np.asarray(centroids), radius=np.asarray(radii), seed_indices=np.asarray(selections))
    classes = np.bincount(labels, minlength=2)
    if classes.min() < spec["sampling"]["minimum_class_count"]:
        raise ValueError(f"Insufficient {split} class counts: {classes.tolist()}")
    return {"samples": total, "class_counts_negative_positive": classes.tolist(),
            "independent_head_regions": len(set(regions)),
            "positive_GT_instances": sorted(set(i for i in instances if i >= 0)),
            "valid_line_count_min_max": [int(min(counts)), int(max(counts))],
            "distinct_subsets_checked": len(fingerprints),
            "files": {p.name: sha(p) for p in folder.iterdir()}}


def prepare(spec, config, index, input_root=None):
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset
    from FMT_Utils.Task4C_PaperBundles_3_1 import (
        load_flow, native_partition, extract_vortex_grid, sample_head_seeds, trace_clean_lines,
    )
    flow = spec["flows"][index]
    out = Path(spec["output"]) / flow["name"]
    out.mkdir(parents=True, exist_ok=False)
    root = Path(input_root or spec["input_root"])
    for role in ("flow", "gt"):
        if sha(root / flow[role]) != flow[role+"_sha256"]:
            raise ValueError(f"Frozen {flow['name']} {role} source changed")
    axes, vortex, heads, oyf, grid, metadata = load_flow(root / flow["flow"], flow["lambda2_threshold"])
    catalog, report = build_head_catalog(axes, heads, read_dataset(root / flow["gt"]), spec)
    report.update(version=spec["version"], identity=identity(config), flow=flow,
                  source=metadata, started_at_utc=datetime.now(timezone.utc).isoformat(), splits={}, line_cleaning=[])
    write(out / "candidate_manifest.json", report)
    cuts = np.asarray(report["x_cuts"])
    options = spec["bundles"]
    wall_span = float(axes[2][-1]-axes[2][0])
    ranges, instance_sets = [], []
    for split_id, split in enumerate(SPLITS):
        left, right, support = native_partition(axes, cuts, split_id, int(metadata["vorticity_source"].startswith("second")))
        ranges.append(support)
        trace_grid = extract_vortex_grid(grid, vortex, left, right)
        selected = [r for r in catalog if r["split"] == split]
        pools, accepted, rejected = [], [], []
        for start in range(0, len(selected), options["heads_per_trace_chunk"]):
            rows = selected[start:start+options["heads_per_trace_chunk"]]
            head_seeds = []
            for row in rows:
                rng = np.random.default_rng(spec["sampling"]["seed"]+index*100000+row["head_component"])
                head_seeds.append(sample_head_seeds(row["cell_ids"], heads.shape, axes, oyf,
                                                  options["seed_pool_per_head"], rng))
            starts = np.concatenate(head_seeds)
            curves, valid, low, high, cleaning = trace_clean_lines(trace_grid, starts, options, wall_span)
            report["line_cleaning"].append({"split": split, "chunk_start": start, **cleaning})
            for j, row in enumerate(rows):
                offset = j*options["seed_pool_per_head"]
                sl = slice(offset, offset+options["seed_pool_per_head"])
                good = valid[sl]
                record = {k: v for k, v in row.items() if k != "cell_ids"}
                if good.sum() < options["minimum_valid_lines"]:
                    rejected.append({**record, "reason": "fewer_than_10_clean_lines", "valid_lines": int(good.sum())})
                    continue
                lo, hi = low[sl][good].min(0), high[sl][good].max(0)
                # Avoid retaining a shape truncated by an artificial split wall.
                guard = 2*options["initial_step_wall_span"]*wall_span
                if lo[0] <= axes[0][left]+guard or hi[0] >= axes[0][right]-guard:
                    rejected.append({**record, "reason": "line_bundle_touches_partition_buffer"})
                    continue
                accepted.append({**record, "valid_lines": int(good.sum()), "physical_bounds": [lo.tolist(), hi.tolist()]})
                pools.append({**record, "lines": curves[sl][good], "seeds": starts[sl][good],
                              "seed_indices": np.flatnonzero(good).astype(np.int32)})
            print(json.dumps({"flow": flow["name"], "split": split, "heads_processed": min(start+len(rows), len(selected)),
                              "heads_total": len(selected), "accepted": len(pools), "rejected": len(rejected)}), flush=True)
        # Compact line pool supports reproducibility without thousands of files.
        pool_offsets = np.r_[0, np.cumsum([len(p["lines"]) for p in pools])]
        if not pools:
            write(out / f"{split}_failure.json", {"rejected": rejected})
            raise ValueError(f"No valid pools in {flow['name']} {split}")
        np.savez_compressed(out / f"{split}_line_pool.npz", offsets=pool_offsets,
            head_component=np.array([p["head_component"] for p in pools]),
            lines=np.concatenate([p["lines"] for p in pools]), seeds=np.concatenate([p["seeds"] for p in pools]),
            seed_indices=np.concatenate([p["seed_indices"] for p in pools]))
        split_report = sample_views(pools, spec, out, split,
                                   np.random.default_rng(spec["sampling"]["seed"]+index*10+split_id))
        split_report.update(accepted_heads=accepted, rejected_after_tracing=rejected,
                            native_source_x_nodes_including_curl_stencil=support,
                            line_pool_sha256=sha(out / f"{split}_line_pool.npz"))
        report["splits"][split] = split_report
        instance_sets.append(set(i for p in pools for i in p["all_instances"]))
        write(out / "preparation.partial.json", report)
    if not (ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0]):
        raise ValueError("Original data support overlaps between splits")
    if any(instance_sets[i] & instance_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("A GT instance occurs across splits")
    report.update(spatial_isolation_verified=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        sampling_scope="Multiple distinct seed-line subsets per disjoint head; sample count is not independent vortex count")
    write(out / "preparation.json", report)


def encode(spec, config):
    import torch
    from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_fmt, bundle_voxels
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise RuntimeError("Bundle encoding requires the allocated GPU")
    out = Path(spec["output"])
    report = {"version": spec["version"], "identity": identity(config), "device": str(device), "splits": {}}
    for flow_id, flow in enumerate(spec["flows"]):
        root = out / flow["name"]
        manifest = json.loads((root / "preparation.json").read_text())
        if not manifest["spatial_isolation_verified"] or manifest["identity"]["config_sha256"] != sha(config):
            raise ValueError("Preparation has not passed this config's checks")
        for split in SPLITS:
            folder = root / split
            for name, digest in manifest["splits"][split]["files"].items():
                if sha(folder / name) != digest:
                    raise ValueError("Prepared geometry changed")
            geometry = np.load(folder / "geometry.npy", mmap_mode="r")
            seeds = np.load(folder / "seeds.npy", mmap_mode="r")
            with np.load(folder / "metadata.npz") as data:
                counts = data["counts"]
            fmt = np.lib.format.open_memmap(folder / "fmt.npy", mode="w+", dtype="float32", shape=(len(geometry), 322))
            voxels = np.lib.format.open_memmap(folder / "voxels.npy", mode="w+", dtype="float16", shape=(len(geometry), 4, 24, 24, 24))
            batch_size = spec["encoding"]["batch_size"]
            for offset in range(0, len(geometry), batch_size):
                sl = slice(offset, offset+batch_size)
                g = torch.tensor(np.asarray(geometry[sl]), device=device)
                s = torch.tensor(np.asarray(seeds[sl]), device=device)
                n = torch.tensor(counts[sl].astype(np.int64), device=device)
                fmt[sl] = bundle_fmt(g, s, n).cpu().numpy()
                voxels[sl] = bundle_voxels(g, n).cpu().numpy().astype(np.float16)
                if offset % (batch_size*100) == 0:
                    print(f"Encoded {flow['name']} {split} {offset}/{len(geometry)}", flush=True)
            fmt.flush(); voxels.flush()
            report["splits"][f"{flow['name']}/{split}"] = {"samples": len(geometry), "files": {
                "fmt.npy": sha(folder / "fmt.npy"), "voxels.npy": sha(folder / "voxels.npy"),
                "metadata.npz": sha(folder / "metadata.npz")}}
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(out / "encoding.json", report)


def load_encoded(spec, split, method, manifest):
    """Separate calls keep all test reads after model and threshold selection."""
    inputs, labels, heads, instances, flows = [], [], [], [], []
    filename = "fmt.npy" if method == "fmt_mlp" else "voxels.npy"
    for index, flow in enumerate(spec["flows"]):
        key = f"{flow['name']}/{split}"
        root = Path(spec["output"]) / key
        for name in (filename, "metadata.npz"):
            if sha(root / name) != manifest["splits"][key]["files"][name]:
                raise ValueError(f"Encoded source changed: {key}/{name}")
        inputs.append(np.load(root / filename))
        with np.load(root / "metadata.npz") as data:
            labels.append(data["labels"]); heads.append(data["head_component"]); instances.append(data["instance"])
            flows.append(np.full(len(data["labels"]), index, np.int8))
    return {"features": np.concatenate(inputs), "labels": np.concatenate(labels), "head_component": np.concatenate(heads),
            "instance": np.concatenate(instances), "flow_index": np.concatenate(flows)}


def grouped_metrics(data, probability, threshold, spec):
    result = {"pooled": binary_metrics(data["labels"], probability, threshold), "per_flow": {}, "per_head": {}}
    for index, flow in enumerate(spec["flows"]):
        mask = data["flow_index"] == index
        result["per_flow"][flow["name"]] = binary_metrics(data["labels"][mask], probability[mask], threshold)
        heads = np.unique(data["head_component"][mask])
        target, score = [], []
        for head in heads:
            selected = mask & (data["head_component"] == head)
            y = np.unique(data["labels"][selected])
            if len(y) != 1:
                raise ValueError("A head has inconsistent view labels")
            target.append(y[0]); score.append(probability[selected].mean())
        result["per_head"][flow["name"]] = {"head_count": len(heads), **binary_metrics(np.asarray(target), np.asarray(score), threshold)}
    return result


def train(spec, config, index):
    import torch
    from torch import nn
    from sklearn.metrics import average_precision_score
    from FMT_Utils.Task4C_PaperBundles_3_1 import FMTBundleClassifier, Conv3DClassifier
    method, seed = run_plan(spec)[index]
    out = Path(spec["output"])
    manifest = json.loads((out / "encoding.json").read_text())
    if manifest["identity"]["config_sha256"] != sha(config):
        raise ValueError("Encoding belongs to a different config")
    folder = out / "runs" / f"{method}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    options = spec["training"]
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise RuntimeError("Training cannot access its allocated GPU")
    started = time.time()
    training = load_encoded(spec, "train", method, manifest)
    validation = load_encoded(spec, "validation", method, manifest)
    mean, std = None, None
    if method == "fmt_mlp":
        mean = training["features"].mean(0, dtype=np.float64)
        std = training["features"].std(0, dtype=np.float64)
        std[std < 1e-8] = 1
        np.savez_compressed(folder / "train_feature_statistics.npz", mean=mean, std=std)
        for data in (training, validation):
            data["features"] = ((data["features"]-mean)/std).astype(np.float32)
    model = (FMTBundleClassifier(options["dropout"]) if method == "fmt_mlp" else Conv3DClassifier(options["dropout"])).to(device)
    counts = np.bincount(training["labels"], minlength=2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counts[0]/counts[1], device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, options["epochs"], eta_min=1e-6)
    batch_size = options["batch_size"]

    def predict(data):
        model.eval()
        probability = []
        with torch.no_grad():
            for start in range(0, len(data["labels"]), batch_size):
                x = torch.as_tensor(data["features"][start:start+batch_size], device=device, dtype=torch.float32)
                probability.append(model(x).sigmoid().cpu().numpy())
        return np.concatenate(probability)

    best, best_epoch, state, history = -1., 0, None, []
    for epoch in range(1, options["epochs"]+1):
        model.train()
        order = rng.permutation(len(training["labels"]))
        total = 0.
        for start in range(0, len(order), batch_size):
            chosen = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(training["features"][chosen], device=device, dtype=torch.float32)
            y = torch.as_tensor(training["labels"][chosen], device=device, dtype=torch.float32)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(chosen)
        schedule.step()
        score = float(average_precision_score(validation["labels"], predict(validation)))
        history.append({"epoch": epoch, "training_loss": total/len(order), "validation_average_precision": score})
        if score > best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"method": method, "seed": seed, **history[-1]}), flush=True)
        if epoch-best_epoch >= options["patience"]:
            break
    model.load_state_dict(state)
    val_probability = predict(validation)
    threshold = choose_threshold(validation["labels"], val_probability)
    # First test read, strictly after epoch and threshold selection.
    test = load_encoded(spec, "test", method, manifest)
    if method == "fmt_mlp":
        test["features"] = ((test["features"]-mean)/std).astype(np.float32)
    probability = predict(test)
    np.savez_compressed(folder / "predictions.npz", probability=probability, labels=test["labels"],
        head_component=test["head_component"], instance=test["instance"], flow_index=test["flow_index"],
        validation_probability=val_probability, validation_labels=validation["labels"])
    write(folder / "result.json", {"version": spec["version"], "identity": identity(config), "config": spec,
        "method": method, "seed": seed, "selected_epoch": best_epoch, "threshold": threshold,
        "parameters": sum(p.numel() for p in model.parameters()), "history": history,
        "test": grouped_metrics(test, probability, threshold, spec),
        "validation": grouped_metrics(validation, val_probability, threshold, spec),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "torch": torch.__version__,
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.time()-started,
        "predictions_sha256": sha(folder / "predictions.npz"), "weights_written": 0,
        "scope": "Two snapshots; disjoint spatial heads/instances, multiple line-subset views; region-derived binary labels"})


def merge(spec, config):
    import csv
    out = Path(spec["output"])
    records, rows = [], []
    for method, seed in run_plan(spec):
        folder = out / "runs" / f"{method}_seed{seed}"
        record = json.loads((folder / "result.json").read_text())
        if record["identity"]["config_sha256"] != sha(config) or sha(folder / "predictions.npz") != record["predictions_sha256"]:
            raise ValueError("Run identity changed")
        with np.load(folder / "predictions.npz") as data:
            if choose_threshold(data["validation_labels"], data["validation_probability"]) != record["threshold"]:
                raise ValueError("Threshold was not selected on validation")
            if grouped_metrics(data, data["probability"], record["threshold"], spec) != record["test"]:
                raise ValueError("Reported test metrics disagree with saved predictions")
        records.append(record)
        for scope, values in [("pooled", record["test"]["pooled"]), *record["test"]["per_flow"].items()]:
            rows.append({"method": method, "seed": seed, "scope": scope, "parameters": record["parameters"],
                         **{k: values[k] for k in ("f1", "average_precision", "balanced_accuracy")}})
    summary = {}
    for method in spec["methods"]:
        summary[method] = {}
        for scope in ("pooled", *[f["name"] for f in spec["flows"]]):
            selected = [r for r in rows if r["method"] == method and r["scope"] == scope]
            summary[method][scope] = {key: {"mean": float(np.mean([r[key] for r in selected])),
                "sample_std": float(np.std([r[key] for r in selected], ddof=1)) if len(selected)>1 else None}
                for key in ("f1", "average_precision", "balanced_accuracy")}
    write(out / "results.json", {"version": spec["version"], "summary": summary,
        "runs": [{k: v for k, v in r.items() if k not in ("history", "config")} for r in records],
        "all_runs_complete": True, "prediction_consistency_checked": True})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary), flush=True)


def runtime(spec, config, phase, state, code=None):
    out = Path(spec["output"])
    device = "CPU"
    if phase in ("encode", "train"):
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    row = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase, "state": state,
           "exit_code": code, "device": device, **identity(config)}
    append_locked(out / "runtime_events.jsonl", json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md", "\n- Task4-c 3.1 runtime: "+json.dumps(row)+"\n")


def submit(spec, config):
    out = Path(spec["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    write(out / "config.frozen.json", spec)
    previous = None
    for phase, cpu, memory, limit in (("prepare", 8, 48, "03:00:00"), ("encode", 4, 32, "02:00:00"),
                                      ("train", 4, 32, "02:00:00"), ("merge", 2, 8, "00:10:00")):
        command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpu}",
            f"--mem={memory}G", f"--time={limit}", f"--job-name=t4c31-{phase}",
            f"--output={out}/logs/{phase}_%A_%a.out", f"--error={out}/logs/{phase}_%A_%a.err"]
        if previous:
            command += [f"--dependency=afterok:{previous}", "--kill-on-invalid-dep=yes"]
        if phase == "prepare":
            command += [f"--array=0-{len(spec['flows'])-1}%2"]
        if phase in ("encode", "train"):
            command += ["--gres=gpu:1", "--constraint=a100|v100", "--exclude=gpu203-02-r"]
        if phase == "train":
            command += [f"--array=0-{len(run_plan(spec))-1}%3"]
        command += ["ibex_bash/task4c_paper_bundles_3p1.sh", phase, config]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        row = {"job_id": job, "version": spec["version"], "phase": phase, "command": command,
               "submitted_at_utc": datetime.now(timezone.utc).isoformat(), "expected_device": "A100/V100" if phase in ("encode", "train") else "CPU",
               "dependency": previous, **identity(config)}
        append_locked(out / "submissions.jsonl", json.dumps(row)+"\n")
        append_locked("docs/ibex_run_registry.md", "\n- Task4-c SUBMITTED "+json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        previous = job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "encode", "train", "merge", "submit", "runtime"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input-root")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--runtime-phase")
    parser.add_argument("--state")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.phase == "prepare":
        prepare(spec, args.config, args.index, args.input_root)
    elif args.phase == "train":
        train(spec, args.config, args.index)
    elif args.phase == "runtime":
        runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else:
        {"encode": encode, "merge": merge, "submit": submit}[args.phase](spec, args.config)


if __name__ == "__main__":
    main()
