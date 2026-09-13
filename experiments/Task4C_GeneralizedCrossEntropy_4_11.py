"""Task4-c 4.11: weighted, smoothed generalized cross entropy without pruning.

Core loss: Zhang and Sabuncu, NeurIPS 2018, equation 6. Class weighting and
soft targets retain the frozen 4.5 protocol; no label-noise theorem is claimed
for these data or this adaptation. The original training loop remains frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from experiments import Task4C_Sharpness_4_8 as sharpness
from FMT_Utils.Task4C_Regularized_4_2 import classifier_loss as frozen_classifier_loss

consistency = sharpness.consistency
pipeline = sharpness.pipeline
FROZEN_IDENTITY = sharpness.identity
FROZEN_PAIRED_LOSS = consistency.paired_loss
DEFAULT_CONFIG = "config/Ablation_Task4C_GeneralizedCrossEntropy_4.11.json"


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    if "base_config" not in definition:
        return definition  # Explicit full engineering fixtures.
    if pipeline.sha(definition["base_config"]) != definition["base_config_sha256"]:
        raise ValueError("Frozen base configuration changed")
    if "candidates" in definition and "training" in definition:
        return definition
    spec = json.loads(Path(definition["base_config"]).read_text())
    for key in ("version", "output", "generalized_loss", "target", "base_config", "base_config_sha256"):
        spec[key] = definition[key]
    spec["training"].update(definition["training_overrides"])
    base = dict(spec["candidates"][0])
    base.pop("sam_rho")
    spec["candidates"] = [dict(base, name=f"candidate_{i:02d}", loss_q=q)
        for i, q in enumerate(definition["generalized_loss"]["q_candidates"])]
    spec["regularization"] = {"type": "generalized_cross_entropy_plus_physical_scale_consistency",
        "inference": "unchanged_single_original_primitive_no_voting"}
    spec["sources"] = ["https://papers.nips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf"]
    return spec


def identity(config):
    result = FROZEN_IDENTITY(config)
    for name in ("experiments/Task4C_GeneralizedCrossEntropy_4_11.py",
                 "ibex_bash/task4c_generalized_cross_entropy_4p11.sh"):
        result["source_sha256"][name] = pipeline.sha(name)
    definition = json.loads(Path(config).read_text())
    if "base_config" in definition:
        result["source_sha256"][definition["base_config"]] = pipeline.sha(definition["base_config"])
    return result


def generalized_cross_entropy(logits, labels, candidate, positive_weight):
    """Sum weighted smoothed targets times Lq, divided by hard-target weights.

    The denominator matches PyTorch weighted cross entropy with label smoothing.
    For q=0, call the original implementation exactly. For q>0, expm1 avoids
    cancellation near q=0. Every example remains in the objective.
    """
    q = candidate["loss_q"]
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Generalized cross entropy requires q in [0, 1]")
    if candidate["classifier"] != "linear":
        raise ValueError("This ablation only supports the frozen linear classifier head")
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("Expected Bx2 logits and B labels")
    if q == 0:
        return frozen_classifier_loss(logits, labels, candidate, positive_weight)
    labels = labels.long()
    smooth = candidate["label_smoothing"]
    if not 0 <= smooth <= 1:
        raise ValueError("Invalid label smoothing")
    weights = torch.stack((positive_weight.new_ones(()), positive_weight))
    targets = F.one_hot(labels, num_classes=2).to(logits.dtype)*(1-smooth)+smooth/2
    terms = -torch.expm1(q*F.log_softmax(logits, dim=-1))/q
    return (terms*targets*weights).sum()/weights[labels].sum()


def paired_loss(logits, labels, valid, candidate, positive_weight, epoch, ramp_epochs):
    original = FROZEN_PAIRED_LOSS(logits, labels, valid, candidate, positive_weight, epoch, ramp_epochs)
    if candidate["loss_q"] == 0:
        return original
    supervised = generalized_cross_entropy(logits, labels, candidate, positive_weight)
    _, _, scale_loss, scale_weight = original
    return supervised+scale_weight*scale_loss, supervised, scale_loss, scale_weight


def train(spec, config, index, phase):
    """Adapt only the loss inside one invocation of the frozen 4.5 trainer.

    Each Slurm task is a separate process; the callback is restored even if
    training fails. No frozen source files or model definitions are edited.
    """
    if phase not in ("search", "confirm"):
        raise ValueError("4.11 only reads training and validation")
    plan = {"search":pipeline.development_plan, "confirm":pipeline.confirmation_plan}[phase](spec)
    method, candidate, seed = plan[index]
    configured = dict(candidate, consistency_weight=candidate["consistency_weights"][method])
    previous = consistency.paired_loss
    try:
        consistency.paired_loss = lambda logits, labels, valid, unused, positive_weight, epoch, ramp_epochs: paired_loss(
            logits, labels, valid, configured, positive_weight, epoch, ramp_epochs)
        consistency.train(spec, config, index, phase)
    finally:
        consistency.paired_loss = previous
    folder = pipeline.run_folder(spec, phase, method, candidate, seed)
    result = json.loads((folder/"result.json").read_text())
    result["supervised_loss_protocol"] = "weighted_smoothed_untruncated_GCE_Eq6_q0_exact_CE"
    result["effective_consistency_weight"] = configured["consistency_weight"]
    pipeline.write(folder/"result.json", result)


def rank(spec, config):
    records=[pipeline.read_run(spec,"search",*item) for item in pipeline.development_plan(spec)]
    top={}
    for method in spec["methods"]:
        rows=[r for r in records if r["method"] == method]
        control=[r for r in rows if r["candidate"]["loss_q"] == 0]
        if len(control) != 1:
            raise ValueError("Exactly one matched cross-entropy control is required")
        best=max([r for r in rows if r["candidate"]["loss_q"] > 0],
                 key=lambda r:(r["validation"]["pooled"]["f1"],r["validation"]["pooled"]["average_precision"]))
        top[method]=[control[0]["candidate"],best["candidate"]]
    pipeline.write(Path(spec["output"])/"development_rank.json",{"identity":identity(config),"top_two":top,
        "rule":"always confirm q=0 control and validation-selected positive q",
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
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.11 runtime: "+json.dumps(row)+"\n")


def submit_job(spec,config,phase,array=None,previous=None):
    gpu=phase in ("search","confirm")
    out=Path(spec["output"]).resolve()
    limit="01:00:00" if gpu else "00:15:00"
    command=["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=32G",f"--time={limit}",
        f"--job-name=t4c411-{phase}",f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
    if gpu:command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
    if array is not None:command += [f"--array=0-{array-1}%4"]
    command += ["ibex_bash/task4c_generalized_cross_entropy_4p11.sh",phase,config]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row={"job_id":job,"phase":phase,"version":spec["version"],"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
         "command":command,"dependency":previous,"expected_device":"A100/V100/P100/RTX2080Ti" if gpu else "CPU",
         **identity(config)}
    pipeline.append_locked(out/"submissions.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.11 SUBMITTED "+json.dumps(row)+"\n")
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
