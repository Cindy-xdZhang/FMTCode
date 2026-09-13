"""Task4-c 4.8: paired-scale training with Euclidean SAM and AdamW.

Core algorithm: Foret et al., arXiv:2010.01412, equations 2-3.
This is an AdamW adaptation, not a reproduction of the paper's benchmarks.
"""
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
import torch

from experiments import Task4C_ScaleConsistency_4_5 as consistency

pipeline = consistency.pipeline
DEFAULT_CONFIG = "config/Ablation_Task4C_Sharpness_4.8.json"
FROZEN_LOAD_DATA = pipeline.load_data


def identity(config):
    result = consistency.FROZEN_IDENTITY(config)
    for name in ("experiments/Task4C_ScaleConsistency_4_5.py", "ibex_bash/task4c_scale_consistency_4p5.sh",
                 "experiments/Task4C_Sharpness_4_8.py", "ibex_bash/task4c_sharpness_4p8.sh"):
        result["source_sha256"][name] = pipeline.sha(name)
    return result


def load_data(spec, split, method, manifest, candidate):
    if split not in ("train", "validation"):
        raise ValueError("4.8 only reads train/validation")
    return FROZEN_LOAD_DATA(spec,split,method,manifest,candidate)


def sam_step(model, optimizer, closure, rho):
    """Two passes with one dropout draw; restore exact weights before AdamW.

    The closure returns (loss, supervised loss, consistency MSE, weight).
    No second-order differentiation and no weight decay in the perturbation.
    Rho zero deliberately uses two passes for a compute-matched control.
    """
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("SAM radius must be finite and nonnegative")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("Stateful BatchNorm is outside this frozen model protocol")
    parameters = [p for p in model.parameters() if p.requires_grad]
    device = parameters[0].device
    def rng_state():
        return (torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    def restore_rng(state):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1],device)
    before_rng = rng_state()
    optimizer.zero_grad(set_to_none=True)
    initial = closure()
    if not torch.isfinite(initial[0]):
        raise ValueError("Nonfinite original loss")
    initial[0].backward()
    after_rng = rng_state()
    active = [p for p in parameters if p.grad is not None]
    if not active:
        raise ValueError("SAM received no gradients")
    norm = torch.stack([p.grad.detach().norm(2) for p in active]).norm(2)
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite original gradient")
    original = [p.detach().clone() for p in active]
    metrics = {"loss":float(initial[0].detach()),"supervised":float(initial[1].detach()),
               "consistency":float(initial[2].detach()),"consistency_weight":float(initial[3]),
               "first_gradient_norm":float(norm.detach())}
    try:
        with torch.no_grad():
            factor = rho / norm.clamp_min(1e-12)
            for p in active:
                p.add_(p.grad * factor)
        optimizer.zero_grad(set_to_none=True)
        restore_rng(before_rng)
        perturbed = closure()
        if not torch.isfinite(perturbed[0]):
            raise ValueError("Nonfinite perturbed loss")
        perturbed[0].backward()
        metrics["perturbed_loss"] = float(perturbed[0].detach())
    finally:
        with torch.no_grad():
            for p, saved in zip(active,original):
                p.copy_(saved)
        restore_rng(after_rng)
    torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    return metrics


def train(spec, config, index, phase):
    import torch
    from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt
    from FMT_Utils.Task4C_LinePooling_4_3 import make_model, classifier_probability
    if phase not in ("search", "confirm"):
        raise ValueError("This development ablation cannot read test data")
    plan = {"search":pipeline.development_plan, "confirm":pipeline.confirmation_plan}[phase](spec)
    method, candidate, seed = plan[index]
    loss_candidate = {**candidate, "consistency_weight":candidate["consistency_weights"][method]}
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
    groups = consistency.center_index(training["flow_index"], training["head_component"], np.concatenate(center_numbers),
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
        total, ce_total, consistency_total, physical_pairs, perturbed_total = 0.,0.,0.,0,0.
        for start in range(0,len(order),batch_size//2):
            chosen = order[start:start+batch_size//2]
            partner, valid = consistency.partner_indices(groups,chosen,pair_rng)
            selected = np.r_[chosen,partner]
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(training["features"][selected],device=device,dtype=torch.float32)
            y = torch.as_tensor(training["labels"][selected],device=device,dtype=torch.float32)
            valid_tensor = torch.as_tensor(valid,device=device)
            def closure():
                return consistency.paired_loss(model(x),y,valid_tensor,loss_candidate,
                    positive_weight,epoch,options["consistency_ramp_epochs"])
            step = sam_step(model,optimizer,closure,candidate["sam_rho"])
            weight = step["consistency_weight"]
            total += step["loss"]*len(chosen)
            ce_total += step["supervised"]*len(chosen)
            consistency_total += step["consistency"]*int(valid.sum())
            perturbed_total += step["perturbed_loss"]*len(chosen)
            physical_pairs += int(valid.sum())
        probability = predict(validation)
        threshold = pipeline.choose_threshold(validation["labels"],probability)
        values = pipeline.binary_metrics(validation["labels"],probability,threshold)
        score = (values["f1"],values["average_precision"])
        history.append({"epoch":epoch,"training_loss":total/len(order),"supervised_loss":ce_total/len(order),
            "consistency_mse":consistency_total/max(1,physical_pairs),"consistency_weight":weight,
            "physical_pair_fraction":physical_pairs/len(order),"training_views":2*len(order),
            "forward_views":4*len(order),"perturbed_loss":perturbed_total/len(order),"sam_rho":candidate["sam_rho"],
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
        "optimizer_protocol":"two_pass_same_batch_same_dropout_SAM_AdamW",
        "effective_consistency_weight":loss_candidate["consistency_weight"],"weights_written":0,"device":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU","torch":torch.__version__}
    probabilities = {"validation_probability":val_probability,"validation_labels":validation["labels"],
        "validation_head_component":validation["head_component"],"validation_flow_index":validation["flow_index"],
        "validation_scale_id":validation["scale_id"]}
    result["test_loaded"] = False
    np.savez_compressed(folder/"predictions.npz",**probabilities)
    result.update(started_at_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.time()-started,
        predictions_sha256=pipeline.sha(folder/"predictions.npz"))
    pipeline.write(folder/"result.json",result)

def rank(spec, config):
    records=[pipeline.read_run(spec,"search",*item) for item in pipeline.development_plan(spec)]
    top={}
    for method in spec["methods"]:
        rows=[r for r in records if r["method"] == method]
        control=[r for r in rows if r["candidate"]["sam_rho"] == 0]
        if len(control) != 1:
            raise ValueError("Exactly one matched AdamW control is required")
        best=max([r for r in rows if r["candidate"]["sam_rho"] > 0],
                 key=lambda r:(r["validation"]["pooled"]["f1"],r["validation"]["pooled"]["average_precision"]))
        top[method]=[control[0]["candidate"],best["candidate"]]
    pipeline.write(Path(spec["output"])/"development_rank.json",{"identity":identity(config),"top_two":top,
        "rule":"always confirm rho=0 control and validation-selected positive rho",
        "rows":[{k:r[k] for k in ("method","candidate","seed","validation","training_subset")} for r in records],
        "test_loaded":False})


def select(spec, config):
    ranking = json.loads((Path(spec["output"])/"development_rank.json").read_text())
    rows=[]
    for method in spec["methods"]:
        for candidate in ranking["top_two"][method]:
            records=[pipeline.read_run(spec,"search",method,candidate,spec["training"]["development_seed"])]
            records += [pipeline.read_run(spec,"confirm",method,candidate,s) for s in spec["training"]["confirmation_seeds"]]
            f1=[r["validation"]["pooled"]["f1"] for r in records]
            rows.append({"method":method,"candidate":candidate,"validation_f1_mean":float(np.mean(f1)),
                "validation_f1_std":float(np.std(f1,ddof=1)),"validation_ap_mean":float(np.mean([r["validation"]["pooled"]["average_precision"] for r in records])),
                "seeds":[r["seed"] for r in records],"per_seed_f1":f1})
    chosen={m:max([r for r in rows if r["method"]==m],key=lambda r:(r["validation_f1_mean"],r["validation_ap_mean"])) for m in spec["methods"]}
    report={"version":spec["version"],"identity":identity(config),"selected":chosen,"candidates":rows,
        "test_loaded":False,"validation_target_met":all(r["validation_f1_mean"]>=spec["target"]["f1"] for r in chosen.values()),
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),"test_status":"4.6 frozen benchmark not reopened by this ablation"}
    pipeline.write(Path(spec["output"])/"selection.json",report)
    print(json.dumps({"selected":chosen,"test_loaded":False}),flush=True)


def runtime(spec,config,phase,state,code=None):
    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    row={"time":datetime.now(timezone.utc).isoformat(),"phase":phase,"state":state,"exit_code":code,
         "device":device,**identity(config)}
    pipeline.append_locked(Path(spec["output"])/"runtime_events.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.8 runtime: "+json.dumps(row)+"\n")


def submit_job(spec,config,phase,array=None,previous=None):
    gpu=phase in ("search","confirm")
    out=Path(spec["output"]).resolve()
    limit="01:00:00" if gpu else "00:15:00"
    command=["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=32G",f"--time={limit}",
        f"--job-name=t4c48-{phase}",f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
    if gpu:command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
    if array is not None:command += [f"--array=0-{array-1}%4"]
    command += ["ibex_bash/task4c_sharpness_4p8.sh",phase,config]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row={"job_id":job,"phase":phase,"version":spec["version"],"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
         "command":command,"dependency":previous,"expected_device":"A100/V100/P100/RTX2080Ti" if gpu else "CPU",
         **identity(config)}
    pipeline.append_locked(out/"submissions.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.8 SUBMITTED "+json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
    return job


def submit(spec,config):
    out=Path(spec["output"]);out.mkdir(parents=True,exist_ok=False);(out/"logs").mkdir()
    pipeline.write(out/"config.frozen.json",spec)
    previous=None
    for phase,count in (("encode",None),("search",len(pipeline.development_plan(spec))),
                        ("rank",None),("confirm",8),("select",None)):
        previous=submit_job(spec,config,phase,count,previous)


def configure_runtime():
    pipeline.identity=identity
    consistency.identity=identity
    pipeline.load_data=load_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("encode","search","rank","confirm","select","submit","runtime"))
    p.add_argument("--config",default=DEFAULT_CONFIG);p.add_argument("--index",type=int,default=0)
    p.add_argument("--runtime-phase");p.add_argument("--state");p.add_argument("--exit-code",type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text());configure_runtime()
    if a.phase in ("search","confirm"):train(spec,a.config,a.index,a.phase)
    elif a.phase=="runtime":runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{"encode":consistency.encode,"rank":rank,"select":select,"submit":submit}[a.phase](spec,a.config)


if __name__ == "__main__":main()
