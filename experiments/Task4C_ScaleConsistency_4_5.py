"""Task4-c 4.5: paired physical scales with prediction-consistency regularization."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from experiments import Task4C_ManifoldMixup_4_4 as pipeline

DEFAULT_CONFIG = "config/mainExp_Task4C_ScaleConsistency_4.5.json"
FROZEN_IDENTITY = pipeline.identity


def identity(config):
    result = FROZEN_IDENTITY(config)
    for name in ("experiments/Task4C_ScaleConsistency_4_5.py", "ibex_bash/task4c_scale_consistency_4p5.sh"):
        result["source_sha256"][name] = pipeline.sha(name)
    return result


def encode(spec, config):
    """Copy verified development caches byte-for-byte; never copy test data."""
    source = Path(spec["cache_source"]["output"])
    manifest_path = source / "encoding.json"
    if pipeline.sha(manifest_path) != spec["cache_source"]["encoding_sha256"]:
        raise ValueError("Frozen development cache manifest changed")
    original = json.loads(manifest_path.read_text())
    if original["identity"]["config_sha256"] != spec["cache_source"]["config_sha256"] or original["test_encoded"]:
        raise ValueError("Unexpected cache source or previously opened test")
    for name, expected in original["identity"]["source_sha256"].items():
        if pipeline.sha(name) != expected:
            raise ValueError(f"Frozen encoder dependency changed: {name}")
    report = {"version":spec["version"], "identity":identity(config), "splits":{}, "test_encoded":False,
              "cache_source_encoding_sha256":pipeline.sha(manifest_path), "copy_mode":"byte_identical_development_only"}
    for split in ("train", "validation"):
        for flow in spec["flows"]:
            key = f"{flow['name']}/{split}"
            destination = Path(spec["output"]) / key
            destination.mkdir(parents=True, exist_ok=False)
            entry = original["splits"][key]
            for name in ("fmt.npy", "voxels.npy", "metadata.npz"):
                if pipeline.sha(source/key/name) != entry["files"][name]:
                    raise ValueError(f"Source cache changed: {key}/{name}")
                shutil.copyfile(source/key/name, destination/name)
                if pipeline.sha(destination/name) != entry["files"][name]:
                    raise ValueError(f"Cache copy failed: {key}/{name}")
            report["splits"][key] = copy.deepcopy(entry)
            print("Copied frozen cache", key, entry["samples"], flush=True)
    pipeline.write(Path(spec["output"])/"encoding.json", report)


def center_index(flow, head, center_number, centers, labels, scales):
    """Index exact same-center views; refuse mixed labels or duplicate scale IDs."""
    keys = np.column_stack((flow, head, center_number))
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    ranks = np.empty(len(order), np.int64)
    ranks[order] = np.arange(len(order)) - offsets[inverse[order]]
    for start, end in zip(offsets[:-1], offsets[1:]):
        ids = order[start:end]
        if not np.all(centers[ids] == centers[ids[0]]) or not np.all(labels[ids] == labels[ids[0]]):
            raise ValueError("A same-center group has inconsistent coordinates or labels")
        if len(np.unique(scales[ids])) != len(ids):
            raise ValueError("Duplicate scale in a same-center group")
    return {"inverse":inverse, "counts":counts, "order":order, "offsets":offsets, "ranks":ranks}


def partner_indices(index, chosen, rng):
    """Uniformly choose a different scale, or self for singleton centers."""
    group = index["inverse"][chosen]
    count = index["counts"][group]
    shift = rng.integers(1, np.maximum(count, 2))
    rank = (index["ranks"][chosen] + shift) % count
    partner = index["order"][index["offsets"][group]+rank]
    return partner, count > 1


def paired_loss(logits, labels, valid_pairs, candidate, positive_weight, epoch, ramp_epochs):
    """Supervise both views; consistency is zero for non-physical singleton pairs."""
    import torch
    from FMT_Utils.Task4C_Regularized_4_2 import classifier_loss
    pairs = len(valid_pairs)
    if len(logits) != 2*pairs or not torch.equal(labels[:pairs], labels[pairs:]):
        raise ValueError("Expected two equally labeled views of every training center")
    supervised = classifier_loss(logits, labels, candidate, positive_weight)
    probability = logits.softmax(-1)
    per_pair = (probability[:pairs]-probability[pairs:]).square().mean(-1)
    valid = valid_pairs.to(per_pair.dtype)
    consistency = (per_pair*valid).sum()/valid.sum().clamp_min(1)
    ramp = 0. if epoch == 1 else float(np.exp(-5*(1-min((epoch-1)/ramp_epochs, 1.))**2))
    weight = candidate["consistency_weight"]*ramp
    return supervised+weight*consistency, supervised, consistency, weight


def train(spec, config, index, phase):
    import torch
    from torch import nn
    from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt
    from FMT_Utils.Task4C_LinePooling_4_3 import make_model, classifier_probability
    plan = {"search":pipeline.development_plan, "confirm":pipeline.confirmation_plan, "final":pipeline.final_plan}[phase](spec)
    method, candidate, seed = plan[index]
    folder = pipeline.run_folder(spec, phase, method, candidate, seed)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(spec["output"])/"encoding.json").read_text())
    if manifest["identity"]["config_sha256"] != pipeline.sha(config):
        raise ValueError("Cache belongs to another experiment")
    options = spec["training"]
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng, pair_rng = np.random.default_rng(seed), np.random.default_rng(seed+300000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise RuntimeError("Training cannot access allocated CUDA device")
    started = time.time()
    training = pipeline.load_data(spec, "train", method, manifest, candidate)
    validation = pipeline.load_data(spec, "validation", method, manifest, candidate)
    center_numbers, centers = [], []
    for flow in spec["flows"]:
        with np.load(Path(spec["output"])/flow["name"]/"train"/"metadata.npz") as meta:
            center_numbers.append(meta["center_number"]); centers.append(meta["center"])
    groups = center_index(training["flow_index"], training["head_component"], np.concatenate(center_numbers),
                          np.concatenate(centers), training["labels"], training["scale_id"])
    mean, std = None, None

    def standardize(data, fit=False):
        nonlocal mean, std
        if method != "fmt_mlp":
            return
        mask = data["features"][..., -1:]
        x = transform_fmt(data["features"][..., :-1], candidate["signed_log_fmt"])
        if fit:
            valid = x[mask[...,0]>.5]
            mean, std = valid.mean(0, dtype=np.float64), valid.std(0, dtype=np.float64)
            std[std < 1e-8] = 1
        x = np.clip(((x-mean)/std).astype(np.float32), -candidate["fmt_clip_std"], candidate["fmt_clip_std"])
        data["features"] = np.concatenate((x*mask,mask),-1)

    standardize(training, True); standardize(validation)
    heads = np.column_stack((training["flow_index"], training["head_component"]))
    _, inv, number = np.unique(heads, axis=0, return_inverse=True, return_counts=True)
    sampling_probability = 1/number[inv].astype(float)
    sampling_probability /= sampling_probability.sum()
    positive_rate = float(np.dot(sampling_probability, training["labels"]))
    positive_weight = torch.tensor((1-positive_rate)/positive_rate, device=device, dtype=torch.float32)
    model = make_model(method, candidate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    batch_size = options["batch_size"]
    if batch_size % 2 or len(training["labels"]) % 2:
        raise ValueError("Paired training requires even batch and training-view counts")

    def predict(data):
        model.eval()
        probability = []
        with torch.no_grad():
            for start in range(0,len(data["labels"]),batch_size):
                x = torch.as_tensor(data["features"][start:start+batch_size], device=device, dtype=torch.float32)
                probability.append(classifier_probability(model(x)).cpu().numpy())
        return np.concatenate(probability)

    best, best_epoch, state, history = (-1.,-1.), 0, None, []
    for epoch in range(1,options["epochs"]+1):
        warm = options["warmup_epochs"]
        factor = epoch/warm if epoch <= warm else .5*(1+np.cos(np.pi*(epoch-warm)/(options["epochs"]-warm)))
        learning_rate = max(1e-6,candidate["learning_rate"]*factor)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        order = rng.choice(len(training["labels"]),len(training["labels"])//2,p=sampling_probability,replace=True)
        total, ce_total, consistency_total, physical_pairs = 0.,0.,0.,0
        for start in range(0,len(order),batch_size//2):
            chosen = order[start:start+batch_size//2]
            partner, valid = partner_indices(groups,chosen,pair_rng)
            selected = np.r_[chosen,partner]
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(training["features"][selected],device=device,dtype=torch.float32)
            y = torch.as_tensor(training["labels"][selected],device=device,dtype=torch.float32)
            loss, ce, consistency, weight = paired_loss(model(x),y,torch.as_tensor(valid,device=device),candidate,
                positive_weight,epoch,options["consistency_ramp_epochs"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite paired training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            total += float(loss.detach())*len(chosen)
            ce_total += float(ce.detach())*len(chosen)
            consistency_total += float(consistency.detach())*int(valid.sum())
            physical_pairs += int(valid.sum())
        probability = predict(validation)
        threshold = pipeline.choose_threshold(validation["labels"],probability)
        values = pipeline.binary_metrics(validation["labels"],probability,threshold)
        score = (values["f1"],values["average_precision"])
        history.append({"epoch":epoch,"training_loss":total/len(order),"supervised_loss":ce_total/len(order),
            "consistency_mse":consistency_total/max(1,physical_pairs),"consistency_weight":weight,
            "physical_pair_fraction":physical_pairs/len(order),"training_views":2*len(order),
            "learning_rate":learning_rate,"validation_f1":score[0],"validation_average_precision":score[1],"threshold":threshold})
        if score > best:
            best,best_epoch,state = score,epoch,copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"phase":phase,"method":method,"candidate":candidate["name"],"seed":seed,**history[-1]}),flush=True)
        if epoch-best_epoch >= options["patience"] and epoch >= warm:
            break
    model.load_state_dict(state)
    val_probability = predict(validation)
    threshold = pipeline.choose_threshold(validation["labels"],val_probability)
    diagnostic = rng.choice(len(training["labels"]),min(2048,len(training["labels"])),replace=False)
    train_subset = {k:v[diagnostic] for k,v in training.items()}
    result = {"version":spec["version"],"phase":phase,"identity":identity(config),"candidate":candidate,"method":method,
        "seed":seed,"threshold":threshold,"selected_epoch":best_epoch,
        "validation":pipeline.all_metrics(validation,val_probability,threshold,spec),
        "training_subset":pipeline.binary_metrics(train_subset["labels"],predict(train_subset),threshold),
        "history":history,"training_examples":len(training["labels"]),"parameters":sum(p.numel() for p in model.parameters()),
        "training_unique_centers":len(groups["counts"]),"training_multiscale_centers":int((groups["counts"]>1).sum()),
        "weights_written":0,"device":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU","torch":torch.__version__}
    probabilities = {"validation_probability":val_probability,"validation_labels":validation["labels"],
        "validation_head_component":validation["head_component"],"validation_flow_index":validation["flow_index"],
        "validation_scale_id":validation["scale_id"]}
    if phase == "final":
        test = pipeline.load_data(spec,"test",method,manifest,candidate)
        standardize(test)
        probability = predict(test)
        result["test"] = pipeline.all_metrics(test,probability,threshold,spec)
        probabilities.update(probability=probability,labels=test["labels"],head_component=test["head_component"],
            instance=test["instance"],flow_index=test["flow_index"],scale_id=test["scale_id"])
    else:
        result["test_loaded"] = False
    np.savez_compressed(folder/"predictions.npz",**probabilities)
    result.update(started_at_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.time()-started,
        predictions_sha256=pipeline.sha(folder/"predictions.npz"))
    pipeline.write(folder/"result.json",result)


def runtime(spec, config, phase, state, code=None):
    device = "CPU"
    if phase in ("encode_test","search","confirm","final"):
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    row = {"time":datetime.now(timezone.utc).isoformat(),"phase":phase,"state":state,"exit_code":code,"device":device,**identity(config)}
    pipeline.append_locked(Path(spec["output"])/"runtime_events.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.5 runtime: "+json.dumps(row)+"\n")


def submit_job(spec, config, phase, array=None, previous=None):
    gpu = phase in ("encode_test","search","confirm","final")
    out = Path(spec["output"]).resolve()
    limit = "00:15:00" if phase in ("encode","encode_test") else "00:45:00" if gpu else "00:05:00"
    command = ["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=32G",f"--time={limit}",
        f"--job-name=t4c45-{phase}",f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:
        command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
    if gpu:
        command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
    if array is not None:
        command += [f"--array=0-{array-1}%6"]
    command += ["ibex_bash/task4c_scale_consistency_4p5.sh",phase,config]
    job = subprocess.check_output(command,text=True).strip().split(";")[0]
    row = {"job_id":job,"version":spec["version"],"phase":phase,"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
        "command":command,"expected_device":"A100/V100/P100/RTX2080Ti" if gpu else "CPU","dependency":previous,**identity(config)}
    pipeline.append_locked(out/"submissions.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.5 SUBMITTED "+json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
    return job


def configure_runtime():
    """Reuse frozen orchestration in this 4.5 process; no old files are modified."""
    pipeline.DEFAULT_CONFIG = DEFAULT_CONFIG
    pipeline.__doc__ = __doc__
    pipeline.identity = identity
    pipeline.encode = encode
    pipeline.train = train
    pipeline.runtime = runtime
    pipeline.submit_job = submit_job


if __name__ == "__main__":
    configure_runtime()
    pipeline.main()
