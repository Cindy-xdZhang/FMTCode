"""Multiscale Task4-c: sealed test, development-only regularization search."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.Task4C_HairpinBinary_2_1 import sha, write, append_locked, choose_threshold, binary_metrics
from experiments.Task4C_PaperBundles_3_1 import (
    build_head_catalog, load_encoded, grouped_metrics, identity as previous_identity, encode as previous_encode,
)

DEFAULT_CONFIG = "config/mainExp_Task4C_Multiscale_4.1.json"
SPLITS = ("train", "validation", "test")


def identity(config):
    result = previous_identity(config)
    for p in ("experiments/Task4C_Multiscale_4_1.py", "FMT_Utils/Task4C_Multiscale_4_1.py"):
        result["source_sha256"][p] = sha(p)
    return result


def save_split(rows, folder, spec):
    folder.mkdir()
    geometry = np.lib.format.open_memmap(folder / "geometry.npy", mode="w+", dtype=np.float32, shape=(len(rows), 27, 32, 3))
    seeds = np.lib.format.open_memmap(folder / "seeds.npy", mode="w+", dtype=np.float32, shape=(len(rows), 27, 3))
    for i, row in enumerate(rows):
        geometry[i], seeds[i] = row.pop("geometry"), row.pop("normalized_seeds")
    geometry.flush(); seeds.flush()
    metadata = {"counts": np.array([r["line_count"] for r in rows], np.int16),
                "labels": np.array([r["label"] for r in rows], np.int8)}
    for name in ("head_component", "instance", "scale_id", "center_number", "source_cell"):
        metadata[name] = np.array([r[name] for r in rows], np.int32)
    for name in ("center", "centroid", "radius", "bounds", "neighbor_distance", "seed_rms_distance", "measured_mean_arc_length"):
        metadata[name] = np.array([r[name] for r in rows])
    signature = np.stack([metadata[k] for k in ("head_component", "center_number", "scale_id")], axis=1)
    if len(np.unique(signature, axis=0)) != len(rows):
        raise ValueError("Duplicated physical center/scale sample")
    counts = np.bincount(metadata["labels"], minlength=2)
    if counts.min() < spec["sampling"]["minimum_class_count"]:
        raise ValueError("The frozen candidate population has too few positive or negative samples")
    np.savez_compressed(folder / "metadata.npz", **metadata)
    return {"samples": len(rows), "class_counts_negative_positive": counts.tolist(),
            "independent_head_regions": len(np.unique(metadata["head_component"])),
            "positive_GT_instances": sorted(int(i) for i in np.unique(metadata["instance"]) if i >= 0),
            "distinct_center_scale_signatures": len(rows),
            "valid_line_count_min_max": [int(metadata["counts"].min()), int(metadata["counts"].max())],
            "files": {p.name: sha(p) for p in folder.iterdir()},
            "scales": {str(s): {"samples": int(np.sum(metadata["scale_id"] == s)),
                "class_counts_negative_positive": np.bincount(metadata["labels"][metadata["scale_id"] == s], minlength=2).tolist(),
                "neighbor_distance_mean": float(metadata["neighbor_distance"][metadata["scale_id"] == s].mean()),
                "actual_seed_rms_distance_mean": float(metadata["seed_rms_distance"][metadata["scale_id"] == s].mean()),
                "measured_arc_length_mean": float(metadata["measured_mean_arc_length"][metadata["scale_id"] == s].mean())}
                for s in np.unique(metadata["scale_id"])}}


def prepare(spec, config, index, input_root=None):
    from scipy import ndimage
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow, native_partition
    from FMT_Utils.Task4C_Multiscale_4_1 import extract_native_partition, center_and_neighbors, trace_primitive_batch
    flow = spec["flows"][index]
    root = Path(input_root or spec["input_root"])
    out = Path(spec["output"]) / flow["name"]
    out.mkdir(parents=True, exist_ok=False)
    for role in ("flow", "gt"):
        if sha(root / flow[role]) != flow[role+"_sha256"]:
            raise ValueError("Frozen source changed")
    axes, vortex, heads, oyf, grid, source = load_flow(root / flow["flow"], flow["lambda2_threshold"])
    components, _ = ndimage.label(heads)
    catalog, details = build_head_catalog(axes, heads, read_dataset(root / flow["gt"]), spec)
    report = {"version": spec["version"], "identity": identity(config), "source": source,
              "catalog": details, "splits": {}, "flow": flow, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write(out / "candidate_manifest.json", report)
    wall_span = float(axes[2][-1]-axes[2][0])
    cuts = np.asarray(details["x_cuts"])
    supports, groups, instance_sets = [], [], []
    for split_id, split in enumerate(SPLITS):
        left, right, support = native_partition(axes, cuts, split_id, int(source["vorticity_source"].startswith("second")))
        supports.append(support)
        mesh = extract_native_partition(grid, axes, left, right)
        eligible = [r for r in catalog if r["split"] == split]
        if not eligible:
            raise ValueError("No eligible heads in split")
        all_rows, all_rejected, cleaning = [], [], []
        scales = spec["scale_sets"][split]
        total = spec["sampling"]["per_flow_counts"][split]
        for scale_id, scale in enumerate(scales):
            target = total//len(scales)+int(scale_id < total % len(scales))
            rows, pending, rejected = [], [], []
            rng = np.random.default_rng(spec["sampling"]["seed"]+index*1000+split_id*100+scale_id)
            attempts, valid_over_quota = 0, 0

            def flush():
                nonlocal pending, valid_over_quota
                if not pending:
                    return
                valid, failed, stats = trace_primitive_batch(mesh, pending, scale, spec, wall_span, (axes[0][left], axes[0][right]))
                cleaning.append({"scale_id": scale_id, **stats})
                remaining = max(0, target-len(rows))
                valid_over_quota += max(0, len(valid)-remaining)
                rows.extend(valid[:remaining])
                rejected.extend(failed)
                pending = []

            for center_number in range(spec["sampling"]["maximum_center_rounds"]):
                for chosen in rng.permutation(len(eligible)):
                    row = eligible[chosen]
                    sample = center_and_neighbors(row, components, axes, oyf, center_number, scale,
                                                  spec["sampling"]["seed"]+index*100000)
                    attempts += 1
                    if sample is None or len(sample["seeds"]) < spec["bundles"]["minimum_valid_lines"]:
                        rejected.append({"reason": "fewer_than_10_head_seeds", "head_component": row["head_component"], "label": row["label"]})
                        continue
                    sample.update(head_component=row["head_component"], label=row["label"], instance=row["instance"],
                                  all_instances=row["all_instances"], scale_id=scale_id, center_number=center_number)
                    pending.append(sample)
                    if len(pending) >= spec["sampling"]["trace_batch_primitives"]:
                        flush()
                    if len(rows) >= target:
                        break
                flush()
                if len(rows) >= target:
                    break
            all_rejected.append({"scale_id": scale_id, "attempts": attempts, "accepted": len(rows), "valid_over_quota": valid_over_quota,
                                 "rejection_counts": {reason: sum(r["reason"] == reason for r in rejected)
                                                       for reason in sorted(set(r["reason"] for r in rejected))},
                                 "rejected_class_counts": np.bincount([r["label"] for r in rejected], minlength=2).tolist()})
            write(out / f"{split}_progress.json", {"scales": all_rejected})
            if len(rows) != target:
                raise ValueError(f"Insufficient valid {flow['name']} {split} {scale['name']}: {len(rows)}/{target}")
            all_rows.extend(rows)
            print(json.dumps({"flow": flow["name"], "split": split, "scale": scale["name"],
                              "valid": len(rows), "attempts": attempts}), flush=True)
        # Shuffle sample order only; never move a head or a scale between splits.
        rng.shuffle(all_rows)
        groups.append(set(r["head_component"] for r in all_rows))
        instance_sets.append(set(i for r in all_rows for i in r["all_instances"]))
        split_report = save_split(all_rows, out / split, spec)
        split_report.update(native_source_x_nodes_including_curl_stencil=support,
                            generation=all_rejected, line_cleaning=cleaning)
        report["splits"][split] = split_report
        write(out / "preparation.partial.json", report)
    if not (supports[0][1] < supports[1][0] and supports[1][1] < supports[2][0]):
        raise ValueError("Source supports overlap")
    if any(groups[i] & groups[j] or instance_sets[i] & instance_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("A head or GT instance crosses splits")
    report.update(spatial_isolation_verified=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Multiple positions and actual center-neighbor/integration scales; no extra independent snapshots")
    write(out / "preparation.json", report)


def encode(spec, config):
    # Frozen 3.1 encoders already accept masked variable line counts.
    previous_encode(spec, config)
    path = Path(spec["output"]) / "encoding.json"
    data = json.loads(path.read_text())
    data["identity"] = identity(config)
    data["test_sealed_for_development"] = True
    write(path, data)


def development_plan(spec):
    return [(method, candidate, spec["training"]["development_seed"])
            for candidate in spec["candidates"] for method in spec["methods"]]


def confirmation_plan(spec):
    rank = json.loads((Path(spec["output"]) / "development_rank.json").read_text())
    return [(method, candidate, seed) for method in spec["methods"] for candidate in rank["top_two"][method]
            for seed in spec["training"]["confirmation_seeds"]]


def final_plan(spec):
    selected = json.loads((Path(spec["output"]) / "selection.json").read_text())
    if not selected["validation_gate_passed"]:
        raise ValueError("Final test remains sealed: validation target has not been reached")
    return [(method, selected["selected"][method]["candidate"], seed)
            for method in spec["methods"] for seed in spec["training"]["final_seeds"]]


def run_folder(spec, phase, method, candidate, seed):
    return Path(spec["output"]) / phase / f"{candidate['name']}_{method}_seed{seed}"


def load_data(spec, split, method, manifest):
    data = load_encoded(spec, split, method, manifest)
    scale = []
    for flow in spec["flows"]:
        with np.load(Path(spec["output"]) / flow["name"] / split / "metadata.npz") as meta:
            scale.append(meta["scale_id"])
    data["scale_id"] = np.concatenate(scale)
    return data


def all_metrics(data, probability, threshold, spec):
    result = grouped_metrics(data, probability, threshold, spec)
    result["per_scale"] = {str(s): binary_metrics(data["labels"][data["scale_id"] == s],
        probability[data["scale_id"] == s], threshold) for s in np.unique(data["scale_id"])}
    return result


def train(spec, config, index, phase):
    import torch
    from torch import nn
    from sklearn.metrics import average_precision_score
    from FMT_Utils.Task4C_Multiscale_4_1 import make_model, transform_fmt
    plan = {"search": development_plan, "confirm": confirmation_plan, "final": final_plan}[phase](spec)
    method, candidate, seed = plan[index]
    folder = run_folder(spec, phase, method, candidate, seed)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(spec["output"]) / "encoding.json").read_text())
    if manifest["identity"]["config_sha256"] != sha(config):
        raise ValueError("Cache belongs to another experiment")
    options = spec["training"]
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise RuntimeError("Training cannot access allocated CUDA device")
    started = time.time()
    training = load_data(spec, "train", method, manifest)
    validation = load_data(spec, "validation", method, manifest)
    mean, std = None, None

    def standardize(data, fit=False):
        nonlocal mean, std
        if method != "fmt_mlp":
            return
        x = transform_fmt(data["features"], candidate["signed_log_fmt"])
        if fit:
            mean, std = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)
            std[std < 1e-8] = 1
        x = ((x-mean)/std).astype(np.float32)
        if candidate["fmt_clip_std"] is not None:
            x = np.clip(x, -candidate["fmt_clip_std"], candidate["fmt_clip_std"])
        data["features"] = x

    standardize(training, True); standardize(validation)
    group = np.column_stack([training["flow_index"], training["head_component"]])
    _, inv, number = np.unique(group, axis=0, return_inverse=True, return_counts=True)
    sampling_probability = 1/number[inv].astype(float)
    sampling_probability /= sampling_probability.sum()
    positive_rate = float(np.dot(sampling_probability, training["labels"]))
    model = make_model(method, candidate).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-positive_rate)/positive_rate, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    batch_size = options["batch_size"]

    def predict(data):
        model.eval()
        probability = []
        with torch.no_grad():
            for start in range(0, len(data["labels"]), batch_size):
                x = torch.as_tensor(data["features"][start:start+batch_size], device=device, dtype=torch.float32)
                probability.append(model(x).sigmoid().cpu().numpy())
        return np.concatenate(probability)

    best, best_epoch, state, history = (-1., -1.), 0, None, []
    for epoch in range(1, options["epochs"]+1):
        warm = options["warmup_epochs"]
        factor = epoch/warm if epoch <= warm else .5*(1+np.cos(np.pi*(epoch-warm)/(options["epochs"]-warm)))
        learning_rate = max(1e-6, candidate["learning_rate"]*factor)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        order = rng.choice(len(training["labels"]), len(training["labels"]), p=sampling_probability, replace=True)
        total = 0.
        for start in range(0, len(order), batch_size):
            chosen = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(training["features"][chosen], device=device, dtype=torch.float32)
            y = torch.as_tensor(training["labels"][chosen], device=device, dtype=torch.float32)
            y = y*(1-candidate["label_smoothing"])+candidate["label_smoothing"]/2
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            total += float(loss.detach())*len(chosen)
        probability = predict(validation)
        threshold = choose_threshold(validation["labels"], probability)
        values = binary_metrics(validation["labels"], probability, threshold)
        score = (values["f1"], values["average_precision"])
        history.append({"epoch": epoch, "training_loss": total/len(order), "learning_rate": learning_rate,
                        "validation_f1": score[0], "validation_average_precision": score[1], "threshold": threshold})
        if score > best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"phase": phase, "method": method, "candidate": candidate["name"], "seed": seed, **history[-1]}), flush=True)
        if epoch-best_epoch >= options["patience"] and epoch >= warm:
            break
    model.load_state_dict(state)
    val_probability = predict(validation)
    threshold = choose_threshold(validation["labels"], val_probability)
    diagnostic = rng.choice(len(training["labels"]), min(2048, len(training["labels"])), replace=False)
    train_subset = {k: v[diagnostic] for k, v in training.items()}
    result = {"version": spec["version"], "phase": phase, "identity": identity(config), "candidate": candidate,
              "method": method, "seed": seed, "threshold": threshold, "selected_epoch": best_epoch,
              "validation": all_metrics(validation, val_probability, threshold, spec),
              "training_subset": binary_metrics(train_subset["labels"], predict(train_subset), threshold),
              "history": history, "parameters": sum(p.numel() for p in model.parameters()), "weights_written": 0,
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "torch": torch.__version__}
    probabilities = {"validation_probability": val_probability, "validation_labels": validation["labels"],
                     "validation_head_component": validation["head_component"], "validation_flow_index": validation["flow_index"],
                     "validation_scale_id": validation["scale_id"]}
    if phase == "final":
        # First test load follows an immutable config selection and all fitting.
        test = load_data(spec, "test", method, manifest)
        standardize(test)
        probability = predict(test)
        result["test"] = all_metrics(test, probability, threshold, spec)
        probabilities.update(probability=probability, labels=test["labels"], head_component=test["head_component"],
                             instance=test["instance"], flow_index=test["flow_index"], scale_id=test["scale_id"])
    else:
        result["test_loaded"] = False
    np.savez_compressed(folder / "predictions.npz", **probabilities)
    result.update(started_at_utc=datetime.fromtimestamp(started, timezone.utc).isoformat(),
                  finished_at_utc=datetime.now(timezone.utc).isoformat(), seconds=time.time()-started,
                  predictions_sha256=sha(folder / "predictions.npz"))
    write(folder / "result.json", result)


def read_run(spec, phase, method, candidate, seed):
    folder = run_folder(spec, phase, method, candidate, seed)
    result = json.loads((folder / "result.json").read_text())
    if sha(folder / "predictions.npz") != result["predictions_sha256"]:
        raise ValueError("Prediction file changed")
    if (result["version"], result["phase"], result["method"], result["candidate"], result["seed"]) != (spec["version"], phase, method, candidate, seed):
        raise ValueError("Run identity does not match its frozen plan")
    encoded = json.loads((Path(spec["output"]) / "encoding.json").read_text())["identity"]
    for key in ("config_sha256", "source_sha256"):
        if result["identity"][key] != encoded[key]:
            raise ValueError("Run source/config differs from encoded data")
    with np.load(folder / "predictions.npz") as p:
        threshold = choose_threshold(p["validation_labels"], p["validation_probability"])
        if threshold != result["threshold"]:
            raise ValueError("Threshold was not selected on validation predictions")
        for split, label_key, probability_key in (("validation", "validation_labels", "validation_probability"), ("test", "labels", "probability")):
            if split == "test" and phase != "final":
                if "probability" in p or result.get("test_loaded") is not False:
                    raise ValueError("Development accessed test data")
                continue
            recomputed = binary_metrics(p[label_key], p[probability_key], threshold)
            for metric in ("f1", "average_precision", "balanced_accuracy"):
                if not np.isclose(recomputed[metric], result[split]["pooled"][metric], rtol=0, atol=1e-12):
                    raise ValueError("Stored metric disagrees with saved probabilities")
    return result


def rank(spec, config):
    records = [read_run(spec, "search", *item) for item in development_plan(spec)]
    top = {}
    for method in spec["methods"]:
        rows = sorted([r for r in records if r["method"] == method],
                      key=lambda r: (r["validation"]["pooled"]["f1"], r["validation"]["pooled"]["average_precision"]), reverse=True)
        top[method] = [r["candidate"] for r in rows[:2]]
    write(Path(spec["output"]) / "development_rank.json", {"identity": identity(config), "top_two": top,
        "rows": [{"method": r["method"], "candidate": r["candidate"], "seed": r["seed"],
                  "validation": r["validation"], "training_subset": r["training_subset"]} for r in records], "test_loaded": False})


def select(spec, config):
    ranking = json.loads((Path(spec["output"]) / "development_rank.json").read_text())
    selected, candidates = {}, []
    for method in spec["methods"]:
        for candidate in ranking["top_two"][method]:
            records = [read_run(spec, "search", method, candidate, spec["training"]["development_seed"])]
            records += [read_run(spec, "confirm", method, candidate, seed) for seed in spec["training"]["confirmation_seeds"]]
            f1 = [r["validation"]["pooled"]["f1"] for r in records]
            ap = [r["validation"]["pooled"]["average_precision"] for r in records]
            row = {"method": method, "candidate": candidate, "validation_f1_mean": float(np.mean(f1)),
                   "validation_f1_std": float(np.std(f1, ddof=1)), "validation_ap_mean": float(np.mean(ap)),
                   "seeds": [r["seed"] for r in records], "per_seed_f1": f1}
            candidates.append(row)
        chosen = max([r for r in candidates if r["method"] == method], key=lambda r: (r["validation_f1_mean"], r["validation_ap_mean"]))
        selected[method] = chosen
    gate = all(r["validation_f1_mean"] >= spec["target"]["f1"] for r in selected.values())
    report = {"version": spec["version"], "identity": identity(config), "selected": selected,
              "candidates": candidates, "validation_gate_passed": gate, "test_loaded": False,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write(Path(spec["output"]) / "selection.json", report)
    print(json.dumps(report), flush=True)
    if gate and os.environ.get("SLURM_JOB_ID"):
        submit_final(spec, config)


def merge(spec, config):
    records = [read_run(spec, "final", *item) for item in final_plan(spec)]
    summary = {}
    for method in spec["methods"]:
        rows = [r for r in records if r["method"] == method]
        summary[method] = {key: {"mean": float(np.mean([r["test"]["pooled"][key] for r in rows])),
                                 "sample_std": float(np.std([r["test"]["pooled"][key] for r in rows], ddof=1))}
                           for key in ("f1", "average_precision", "balanced_accuracy")}
    write(Path(spec["output"]) / "results.json", {"version": spec["version"], "identity": identity(config),
        "summary": summary, "runs": [{k: v for k, v in r.items() if k != "history"} for r in records],
        "target_reached": all(s["f1"]["mean"] >= spec["target"]["f1"] for s in summary.values())})


def runtime(spec, config, phase, state, code=None):
    device = "CPU"
    if phase in ("encode", "search", "confirm", "final"):
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    row = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase, "state": state, "exit_code": code,
           "device": device, **identity(config)}
    append_locked(Path(spec["output"]) / "runtime_events.jsonl", json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md", "\n- Task4-c 4.1 runtime: "+json.dumps(row)+"\n")


def submit_job(spec, config, phase, array=None, previous=None):
    gpu = phase in ("encode", "search", "confirm", "final")
    out = Path(spec["output"]).resolve()
    command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=8" if phase == "prepare" else "--cpus-per-task=4",
               "--mem=48G" if phase == "prepare" else "--mem=32G", "--time=04:00:00" if phase == "prepare" else "--time=02:00:00",
               f"--job-name=t4c41-{phase}", f"--output={out}/logs/{phase}_%A_%a.out", f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:
        command += [f"--dependency=afterok:{previous}", "--kill-on-invalid-dep=yes"]
    if gpu:
        command += ["--gres=gpu:1", "--constraint=a100|v100", "--exclude=gpu203-02-r"]
    if array is not None:
        command += [f"--array=0-{array-1}%{2 if phase == 'prepare' else 6}"]
    command += ["ibex_bash/task4c_multiscale_4p1.sh", phase, config]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    row = {"job_id": job, "version": spec["version"], "phase": phase, "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
           "command": command, "expected_device": "A100/V100" if gpu else "CPU", "dependency": previous, **identity(config)}
    append_locked(out / "submissions.jsonl", json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md", "\n- Task4-c 4.1 SUBMITTED "+json.dumps(row)+"\n")
    print(json.dumps(row), flush=True)
    return job


def submit(spec, config):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False); (out / "logs").mkdir()
    write(out / "config.frozen.json", spec)
    previous = None
    for phase, count in (("prepare", 2), ("encode", None), ("search", len(development_plan(spec))),
                         ("rank", None), ("confirm", 8), ("select", None)):
        previous = submit_job(spec, config, phase, count, previous)


def submit_final(spec, config):
    plan = final_plan(spec)
    marker = Path(spec["output"]) / "final_submission.lock"
    with marker.open("x") as handle:
        handle.write(datetime.now(timezone.utc).isoformat())
    job = submit_job(spec, config, "final", len(plan))
    submit_job(spec, config, "merge", previous=job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "encode", "search", "rank", "confirm", "select", "final", "merge", "submit", "runtime"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input-root")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--runtime-phase"); parser.add_argument("--state"); parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if args.phase == "prepare":
        prepare(spec, args.config, args.index, args.input_root)
    elif args.phase in ("search", "confirm", "final"):
        train(spec, args.config, args.index, args.phase)
    elif args.phase == "runtime":
        runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else:
        {"encode": encode, "rank": rank, "select": select, "merge": merge, "submit": submit}[args.phase](spec, args.config)


if __name__ == "__main__":
    main()
