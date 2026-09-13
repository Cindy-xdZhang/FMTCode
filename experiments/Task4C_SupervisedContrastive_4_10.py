"""Task4-c 4.10: auxiliary supervised contrastive loss on paired physical scales.

Uses the anchor-mean form of Khosla et al. (2020), equation 2. This is joint
classification/contrastive training, not their two-stage ImageNet reproduction.
The frozen 4.3 classifier and its inference path are preserved.
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
from torch import nn
from torch.nn import functional as F

from experiments import Task4C_Sharpness_4_8 as sharpness
from FMT_Utils import Task4C_LinePooling_4_3 as networks

consistency = sharpness.consistency
pipeline = sharpness.pipeline
FROZEN_IDENTITY = sharpness.identity
DEFAULT_CONFIG = "config/Ablation_Task4C_SupervisedContrastive_4.10.json"


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    if "base_config" not in definition:
        return definition  # Explicit full engineering fixtures.
    if pipeline.sha(definition["base_config"]) != definition["base_config_sha256"]:
        raise ValueError("Frozen base configuration changed")
    if "candidates" in definition and "training" in definition:
        return definition
    spec = json.loads(Path(definition["base_config"]).read_text())
    for key in ("version", "output", "contrastive", "target", "base_config", "base_config_sha256"):
        spec[key] = definition[key]
    spec["training"].update(definition["training_overrides"])
    base = dict(spec["candidates"][0])
    base.pop("sam_rho")
    spec["candidates"] = [dict(base, name=f"candidate_{i:02d}", contrastive_weight=weight,
        contrastive_temperature=definition["contrastive"]["temperature"])
        for i, weight in enumerate(definition["contrastive"]["weights"])]
    spec["regularization"] = {"type": "physical_scale_consistency_plus_supervised_contrastive",
        "inference": "one_original_primitive_without_projection_or_voting"}
    return spec


def identity(config):
    result = FROZEN_IDENTITY(config)
    for name in ("experiments/Task4C_SupervisedContrastive_4_10.py",
                 "ibex_bash/task4c_supervised_contrastive_4p10.sh"):
        result["source_sha256"][name] = pipeline.sha(name)
    definition = json.loads(Path(config).read_text())
    if "base_config" in definition:
        result["source_sha256"][definition["base_config"]] = pipeline.sha(definition["base_config"])
    return result


class ContrastiveClassifier(nn.Module):
    """Frozen classifier topology plus a training-only 64->128->128 projector."""
    def __init__(self, method, candidate):
        super().__init__()
        if method not in ("fmt_mlp", "conv3d_mlp") or candidate["architecture"] != "learned_pool_wider_conv_4.3":
            raise ValueError("Only the frozen 4.3 models are supported")
        self.method = method
        self.model = networks.make_model(method, candidate)
        # Projector construction must not change subsequent classifier dropout draws.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, x, with_projection=False):
        if self.method == "fmt_mlp":
            mask = x[..., -1] > .5
            line = self.model.line(x[..., :-1])
            mean = (line * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            maximum = line.masked_fill(~mask[..., None], -torch.inf).amax(1)
            embedding = self.model.head[:-1](torch.cat((mean, maximum), -1))
            logits = self.model.head[-1](embedding)
        else:
            embedding = self.model.network[:-1](x)
            logits = self.model.network[-1](embedding)
        if not with_projection:
            return logits
        projected = F.normalize(self.projection(F.normalize(embedding, dim=-1)), dim=-1)
        return logits, projected


def supervised_contrastive(projected, labels, temperature):
    """Mean equation-2 loss; all other same-class entries are positives.

    Self-comparisons are excluded from both numerator and denominator. Anchors
    without positives contribute zero; physical paired batches have positives.
    Temperature/base-temperature scaling is one (both use the same value).
    """
    if projected.ndim != 2 or labels.ndim != 1 or len(projected) != len(labels):
        raise ValueError("Expected BxD projections and B labels")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if len(labels) < 2:
        return projected.sum() * 0
    scores = projected @ projected.T / temperature
    other = ~torch.eye(len(labels), dtype=torch.bool, device=projected.device)
    positive = (labels[:, None] == labels[None, :]) & other
    count = positive.sum(-1)
    log_denominator = scores.masked_fill(~other, -torch.inf).logsumexp(-1, keepdim=True)
    log_probability = scores - log_denominator
    loss = -torch.where(positive, log_probability, 0.).sum(-1) / count.clamp_min(1)
    return loss.mean()


def combined_loss(logits, projected, labels, valid, candidate, positive_weight, epoch, ramp_epochs):
    loss, ce, scale_loss, scale_weight = consistency.paired_loss(
        logits, labels, valid, candidate, positive_weight, epoch, ramp_epochs)
    weight = candidate["contrastive_weight"]
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Contrastive weight must be finite and nonnegative")
    contrastive = loss.new_zeros(())
    if weight:
        if projected is None:
            raise ValueError("Positive contrastive weight requires projections")
        contrastive = supervised_contrastive(projected, labels, candidate["contrastive_temperature"])
        loss = loss + weight * contrastive
    return loss, ce, scale_loss, scale_weight, contrastive


def train(spec, config, index, phase):
    import torch
    from torch import nn
    from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt
    from FMT_Utils.Task4C_LinePooling_4_3 import classifier_probability
    if phase not in ("search", "confirm"):
        raise ValueError("4.10 only reads training and validation")
    plan = {"search":pipeline.development_plan, "confirm":pipeline.confirmation_plan}[phase](spec)
    method, candidate, seed = plan[index]
    loss_candidate = dict(candidate, consistency_weight=candidate["consistency_weights"][method])
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
    model = ContrastiveClassifier(method, candidate).to(device)
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
        total, ce_total, consistency_total, physical_pairs, contrastive_total = 0.,0.,0.,0,0.
        for start in range(0,len(order),batch_size//2):
            chosen = order[start:start+batch_size//2]
            partner, valid = consistency.partner_indices(groups,chosen,pair_rng)
            selected = np.r_[chosen,partner]
            optimizer.zero_grad(set_to_none=True)
            x = torch.as_tensor(training["features"][selected],device=device,dtype=torch.float32)
            y = torch.as_tensor(training["labels"][selected],device=device,dtype=torch.float32)
            if candidate["contrastive_weight"]:
                logits, projected = model(x, with_projection=True)
            else:
                logits, projected = model(x), None
            loss, ce, scale_loss, weight, contrastive = combined_loss(logits, projected, y,
                torch.as_tensor(valid, device=device), loss_candidate, positive_weight, epoch,
                options["consistency_ramp_epochs"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite paired training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            total += float(loss.detach())*len(chosen)
            ce_total += float(ce.detach())*len(chosen)
            consistency_total += float(scale_loss.detach())*int(valid.sum())
            contrastive_total += float(contrastive.detach())*len(chosen)
            physical_pairs += int(valid.sum())
        probability = predict(validation)
        threshold = pipeline.choose_threshold(validation["labels"],probability)
        values = pipeline.binary_metrics(validation["labels"],probability,threshold)
        score = (values["f1"],values["average_precision"])
        history.append({"epoch":epoch,"training_loss":total/len(order),"supervised_loss":ce_total/len(order),
            "consistency_mse":consistency_total/max(1,physical_pairs),"consistency_weight":weight,
            "physical_pair_fraction":physical_pairs/len(order),"training_views":2*len(order),
            "contrastive_loss":contrastive_total/len(order),"contrastive_weight":candidate["contrastive_weight"],
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
        "inference_parameters":sum(p.numel() for p in model.model.parameters()),
        "projection_parameters":sum(p.numel() for p in model.projection.parameters()),
        "effective_consistency_weight":loss_candidate["consistency_weight"],
        "optimizer_protocol":"one_pass_AdamW_joint_classification_scale_consistency_supervised_contrastive",
        "weights_written":0,"device":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU","torch":torch.__version__}
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
        control=[r for r in rows if r["candidate"]["contrastive_weight"] == 0]
        if len(control) != 1:
            raise ValueError("Exactly one zero-weight classifier control is required")
        best=max([r for r in rows if r["candidate"]["contrastive_weight"] > 0],
                 key=lambda r:(r["validation"]["pooled"]["f1"],r["validation"]["pooled"]["average_precision"]))
        top[method]=[control[0]["candidate"],best["candidate"]]
    pipeline.write(Path(spec["output"])/"development_rank.json",{"identity":identity(config),"top_two":top,
        "rule":"always confirm zero-weight control and validation-selected positive contrastive weight",
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
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.10 runtime: "+json.dumps(row)+"\n")


def submit_job(spec,config,phase,array=None,previous=None):
    gpu=phase in ("search","confirm")
    out=Path(spec["output"]).resolve()
    limit="01:00:00" if gpu else "00:15:00"
    command=["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=32G",f"--time={limit}",
        f"--job-name=t4c410-{phase}",f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
    if gpu:command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
    if array is not None:command += [f"--array=0-{array-1}%4"]
    command += ["ibex_bash/task4c_supervised_contrastive_4p10.sh",phase,config]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row={"job_id":job,"phase":phase,"version":spec["version"],"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
         "command":command,"dependency":previous,"expected_device":"A100/V100/P100/RTX2080Ti" if gpu else "CPU",
         **identity(config)}
    pipeline.append_locked(out/"submissions.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.10 SUBMITTED "+json.dumps(row)+"\n")
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
    pipeline.load_data=sharpness.load_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("encode","search","rank","confirm","select","submit","runtime"))
    p.add_argument("--config",default=DEFAULT_CONFIG);p.add_argument("--index",type=int,default=0)
    p.add_argument("--runtime-phase");p.add_argument("--state");p.add_argument("--exit-code",type=int)
    a=p.parse_args();spec=load_spec(a.config);configure_runtime()
    if a.phase in ("search","confirm"):train(spec,a.config,a.index,a.phase)
    elif a.phase=="runtime":runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{"encode":consistency.encode,"rank":rank,"select":select,"submit":submit}[a.phase](spec,a.config)


if __name__ == "__main__":main()

