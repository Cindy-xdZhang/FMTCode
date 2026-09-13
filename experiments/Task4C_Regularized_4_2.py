"""Task4-c 4.2: fixed multiscale data, wall-relative FMT and physical regularization."""
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

DEFAULT_CONFIG = "config/mainExp_Task4C_Regularized_4.2.json"
SPLITS = ("train", "validation", "test")


def identity(config):
    result = previous_identity(config)
    for p in ("experiments/Task4C_Multiscale_4_1.py", "FMT_Utils/Task4C_Multiscale_4_1.py",
              "experiments/Task4C_Regularized_4_2.py", "FMT_Utils/Task4C_Regularized_4_2.py"):
        result["source_sha256"][p] = sha(p)
    return result


def encode_partition(spec, config, split, report):
    import torch
    from FMT_Utils.Task4C_Regularized_4_2 import extended_fmt, augment_bundle, bundle_voxels
    parent, out = Path(spec["parent_output"]), Path(spec["output"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise ValueError("Allocated CUDA device unavailable")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    base_split = "train" if split == "train_augmented" else split
    for flow_index, flow in enumerate(spec["flows"]):
        key = flow["name"]+"/"+split
        folder, source = out/key, parent/flow["name"]/base_split
        folder.mkdir(parents=True, exist_ok=False)
        preparation = json.loads((source.parent/"preparation.json").read_text())
        if not preparation["spatial_isolation_verified"] or preparation["identity"]["config_sha256"] != spec["parent_config_sha256"]:
            raise ValueError("Parent data differs from frozen 4.1")
        for name in ("geometry.npy", "seeds.npy", "metadata.npz"):
            if sha(source/name) != preparation["splits"][base_split]["files"][name]:
                raise ValueError("Parent source changed")
        geometry, seeds = np.load(source/"geometry.npy", mmap_mode="r"), np.load(source/"seeds.npy", mmap_mode="r")
        with np.load(source/"metadata.npz") as original:
            metadata = {k:original[k] for k in original.files}
        n = len(geometry)
        original_index = np.arange(n)
        kind = np.zeros(n, np.int8)
        if split == "train_augmented":
            positive = np.flatnonzero(metadata["labels"] == 1)
            original_index = np.r_[original_index, positive, positive]
            kind = np.r_[kind, np.ones(len(positive), np.int8), np.full(len(positive), 2, np.int8)]
        rng = np.random.default_rng(spec["augmentation"]["seed"]+flow_index)
        angles = rng.uniform(-15,15,n).astype(np.float32)*np.pi/180
        actual_angles = np.where(kind == 2, angles[original_index], 0).astype(np.float32)
        expanded = {k:v[original_index] for k,v in metadata.items()}
        expanded.update(original_sample_index=original_index, augmentation_type=kind, tilt_radians=actual_angles)
        np.savez_compressed(folder/"metadata.npz", **expanded)
        fmt = np.lib.format.open_memmap(folder/"fmt.npy", mode="w+", dtype="float32", shape=(len(kind),610))
        voxels = np.lib.format.open_memmap(folder/"voxels.npy", mode="w+", dtype="float16", shape=(len(kind),4,24,24,24))
        for offset in range(0,len(kind),32):
            sl = slice(offset,offset+32)
            index = original_index[sl]
            g = torch.tensor(np.asarray(geometry[index]), device=device)
            s = torch.tensor(np.asarray(seeds[index]), device=device)
            for code, name in ((1,"mirror"),(2,"tilt")):
                selected = torch.as_tensor(kind[sl] == code, device=device)
                if selected.any():
                    angle = torch.as_tensor(actual_angles[sl], device=device)[selected]
                    g[selected],s[selected] = augment_bundle(g[selected],s[selected],name,angle)
            counts = torch.as_tensor(expanded["counts"][sl].astype(np.int64),device=device)
            fmt[sl] = extended_fmt(g,s,counts).cpu().numpy()
            voxels[sl] = bundle_voxels(g,counts).cpu().numpy().astype(np.float16)
            if offset%3200 == 0: print("Encoded",key,offset,len(kind),flush=True)
        fmt.flush(); voxels.flush()
        report["splits"][key] = {"samples":len(kind),"physical_original_samples":n,
            "augmentation_counts":np.bincount(kind,minlength=3).tolist(),
            "class_counts":np.bincount(expanded["labels"],minlength=2).tolist(),
            "files":{name:sha(folder/name) for name in ("fmt.npy","voxels.npy","metadata.npz")}}


def encode(spec, config):
    report = {"version":spec["version"],"identity":identity(config),"splits":{},"test_encoded":False,
              "parent_encoding_sha256":sha(Path(spec["parent_output"])/"encoding.json")}
    for split in ("train","validation","train_augmented"):
        encode_partition(spec,config,split,report)
    write(Path(spec["output"])/"encoding.json",report)


def encode_test(spec, config):
    selected = json.loads((Path(spec["output"])/"selection.json").read_text())
    if not selected["validation_gate_passed"]:
        raise ValueError("Test remains sealed")
    report = json.loads((Path(spec["output"])/"encoding.json").read_text())
    encode_partition(spec,config,"test",report)
    report["test_encoded"] = True
    write(Path(spec["output"])/"encoding.json",report)


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


def load_data(spec, split, method, manifest, candidate):
    if split == "train" and candidate["augmentation"]:
        split = "train_augmented"
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
    from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt
    from FMT_Utils.Task4C_Regularized_4_2 import make_model, classifier_loss, classifier_probability
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
    training = load_data(spec, "train", method, manifest, candidate)
    validation = load_data(spec, "validation", method, manifest, candidate)
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
    positive_weight = torch.tensor((1-positive_rate)/positive_rate, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    batch_size = options["batch_size"]

    def predict(data):
        model.eval()
        probability = []
        with torch.no_grad():
            for start in range(0, len(data["labels"]), batch_size):
                x = torch.as_tensor(data["features"][start:start+batch_size], device=device, dtype=torch.float32)
                probability.append(classifier_probability(model(x)).cpu().numpy())
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
            loss = classifier_loss(model(x), y, candidate, positive_weight)
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
              "history": history, "training_examples":len(training["labels"]), "parameters": sum(p.numel() for p in model.parameters()), "weights_written": 0,
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", "torch": torch.__version__}
    probabilities = {"validation_probability": val_probability, "validation_labels": validation["labels"],
                     "validation_head_component": validation["head_component"], "validation_flow_index": validation["flow_index"],
                     "validation_scale_id": validation["scale_id"]}
    if phase == "final":
        # First test load follows an immutable config selection and all fitting.
        test = load_data(spec, "test", method, manifest, candidate)
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
    if phase in ("encode", "encode_test", "search", "confirm", "final"):
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    row = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase, "state": state, "exit_code": code,
           "device": device, **identity(config)}
    append_locked(Path(spec["output"]) / "runtime_events.jsonl", json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md", "\n- Task4-c 4.2 runtime: "+json.dumps(row)+"\n")


def submit_job(spec, config, phase, array=None, previous=None):
    gpu = phase in ("encode", "encode_test", "search", "confirm", "final")
    out = Path(spec["output"]).resolve()
    limit = "00:15:00" if phase in ("encode", "encode_test") else "00:45:00" if gpu else "00:05:00"
    command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=4",
               "--mem=32G", f"--time={limit}",
               f"--job-name=t4c42-{phase}", f"--output={out}/logs/{phase}_%A_%a.out", f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:
        command += [f"--dependency=afterok:{previous}", "--kill-on-invalid-dep=yes"]
    if gpu:
        command += ["--gres=gpu:1", "--constraint=a100|v100|p100|rtx2080ti", "--exclude=gpu203-02-r"]
    if array is not None:
        command += [f"--array=0-{array-1}%6"]
    command += ["ibex_bash/task4c_regularized_4p2.sh", phase, config]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    row = {"job_id": job, "version": spec["version"], "phase": phase, "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
           "command": command, "expected_device": "A100/V100/P100/RTX2080Ti" if gpu else "CPU", "dependency": previous, **identity(config)}
    append_locked(out / "submissions.jsonl", json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md", "\n- Task4-c 4.2 SUBMITTED "+json.dumps(row)+"\n")
    print(json.dumps(row), flush=True)
    return job


def submit(spec, config):
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=False); (out / "logs").mkdir()
    write(out / "config.frozen.json", spec)
    previous = None
    for phase, count in (("encode", None), ("search", len(development_plan(spec))),
                         ("rank", None), ("confirm", 8), ("select", None)):
        previous = submit_job(spec, config, phase, count, previous)


def submit_final(spec, config):
    plan = final_plan(spec)
    marker = Path(spec["output"]) / "final_submission.lock"
    with marker.open("x") as handle:
        handle.write(datetime.now(timezone.utc).isoformat())
    encoded = submit_job(spec, config, "encode_test")
    job = submit_job(spec, config, "final", len(plan), encoded)
    submit_job(spec, config, "merge", previous=job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["encode", "encode_test", "search", "rank", "confirm", "select", "final", "merge", "submit", "runtime"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input-root")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--runtime-phase"); parser.add_argument("--state"); parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if args.phase in ("search", "confirm", "final"):
        train(spec, args.config, args.index, args.phase)
    elif args.phase == "runtime":
        runtime(spec, args.config, args.runtime_phase, args.state, args.exit_code)
    else:
        {"encode": encode, "encode_test": encode_test, "rank": rank, "select": select, "merge": merge, "submit": submit}[args.phase](spec, args.config)


if __name__ == "__main__":
    main()
