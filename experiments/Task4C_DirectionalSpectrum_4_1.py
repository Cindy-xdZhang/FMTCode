"""Development-only ablation of direction-preserving Fourier bundle features."""
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

from experiments.Task4C_HairpinBinary_2_1 import sha, write, append_locked, choose_threshold, binary_metrics
from experiments.Task4C_PaperBundles_3_1 import grouped_metrics

CONFIG = "config/Verify_Task4C_DirectionalSpectrum_4.1.json"


def identity(config, spec):
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "config_sha256": sha(config), "source_sha256": sha(__file__),
            "parent_encoding_sha256": sha(Path(spec["parent_output"])/"encoding.json"),
            "host": os.environ.get("SLURMD_NODENAME", os.environ.get("COMPUTERNAME")),
            "job": os.environ.get("SLURM_JOB_ID"), "array_index": os.environ.get("SLURM_ARRAY_TASK_ID")}


@torch.no_grad()
def spectrum_features(geometry, counts, frequencies, kind="signed"):
    """Keep signed xyz complex coefficients; pool only over valid curves."""
    mask = torch.arange(geometry.shape[1], device=geometry.device)[None] < counts[:, None]
    if (counts < 10).any() or not torch.isfinite(geometry).all():
        raise ValueError("Invalid cleaned bundle")
    if kind == "gram":
        from FMT_Utils.DFT_FMT_3D import dft_rotation_invariants_3d
        b,n,p,_ = geometry.shape
        coefficients = dft_rotation_invariants_3d(geometry.reshape(b*n,p,3)/(p**.5), frequencies, "gram", True).reshape(b,n,-1)
    else:
        coefficients = torch.view_as_real(torch.fft.rfft(geometry, dim=2, norm="ortho")[:, :, :frequencies]).flatten(2)
    mean = (coefficients*mask[..., None]).sum(1)/counts[:, None]
    variance = ((coefficients-mean[:, None]).square()*mask[..., None]).sum(1)/counts[:, None]
    maximum = coefficients.masked_fill(~mask[..., None], -torch.inf).amax(1)
    minimum = coefficients.masked_fill(~mask[..., None], torch.inf).amin(1)
    return torch.cat((mean, variance.sqrt(), maximum, minimum), dim=1)


def encode(spec, config):
    parent, out = Path(spec["parent_output"]), Path(spec["output"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise ValueError("Allocated CUDA device unavailable")
    manifest = {"identity": identity(config, spec), "splits": {}, "test_loaded": False}
    for flow in spec["flows"]:
        prep = json.loads((parent/flow["name"]/"preparation.json").read_text())
        if not prep["spatial_isolation_verified"] or prep["identity"]["config_sha256"] != spec["parent_config_sha256"]:
            raise ValueError("Parent preparation differs from frozen 4.1")
        for split in ("train", "validation"):
            key = flow["name"]+"/"+split
            source, folder = parent/key, out/key
            folder.mkdir(parents=True, exist_ok=False)
            for name in ("metadata.npz", "geometry.npy"):
                if sha(source/name) != prep["splits"][split]["files"][name]:
                    raise ValueError("Parent geometry/metadata changed")
            with np.load(source/"metadata.npz") as meta:
                counts = meta["counts"]
            geometry = np.load(source/"geometry.npy", mmap_mode="r")
            arrays = {r["name"]: np.lib.format.open_memmap(folder/(r["name"]+".npy"), mode="w+", dtype="float32",
                       shape=(len(geometry), 24*r["frequencies"] if r["kind"]=="signed" else 4*(4*r["frequencies"]-1))) for r in spec["representations"]}
            for start in range(0, len(geometry), 256):
                sl = slice(start, start+256)
                g = torch.as_tensor(np.array(geometry[sl]), device=device)
                n = torch.as_tensor(counts[sl].astype(np.int64), device=device)
                for representation in spec["representations"]:
                    arrays[representation["name"]][sl] = spectrum_features(g, n, representation["frequencies"], representation["kind"]).cpu().numpy()
            for array in arrays.values():
                array.flush()
            manifest["splits"][key] = {"features": {name: sha(folder/(name+".npy")) for name in arrays},
                                       "metadata_sha256": sha(source/"metadata.npz")}
            print("Encoded development spectrum:", key, flush=True)
    write(out/"encoding.json", manifest)


def load_split(spec, manifest, split, representation):
    if split not in ("train", "validation"):
        raise ValueError("This ablation has no test access")
    rows = []
    for index, flow in enumerate(spec["flows"]):
        key = flow["name"]+"/"+split
        features = Path(spec["output"])/key/(representation+".npy")
        metadata = Path(spec["parent_output"])/key/"metadata.npz"
        if sha(features) != manifest["splits"][key]["features"][representation] or sha(metadata) != manifest["splits"][key]["metadata_sha256"]:
            raise ValueError("Encoded development data changed")
        with np.load(metadata) as meta:
            row = {k: meta[k] for k in ("labels", "head_component", "instance", "scale_id")}
        row.update(features=np.load(features), flow_index=np.full(len(row["labels"]), index))
        rows.append(row)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def train(spec, config, index):
    candidate = spec["candidates"][index]
    options = spec["training"]
    seed = spec["seed"]
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise ValueError("Allocated CUDA device unavailable")
    folder = Path(spec["output"])/"runs"/candidate["name"]
    folder.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(spec["output"])/"encoding.json").read_text())
    current = identity(config, spec)
    for k in ("source_sha256", "config_sha256", "parent_encoding_sha256"):
        if current[k] != manifest["identity"][k]:
            raise ValueError("Changed diagnostic source/config")
    data = {s: load_split(spec, manifest, s, candidate["representation"]) for s in ("train", "validation")}
    # Same train-only signed-log, standardization and clipping as normalized 4.1.
    for row in data.values():
        row["features"] = np.sign(row["features"])*np.log1p(np.abs(row["features"]))
    mean, std = data["train"]["features"].mean(0, dtype=np.float64), data["train"]["features"].std(0, dtype=np.float64)
    std[std < 1e-8] = 1
    for row in data.values():
        row["features"] = np.clip((row["features"]-mean)/std, -8, 8).astype(np.float32)
    training, validation = data["train"], data["validation"]
    _, inverse, count = np.unique(np.column_stack((training["flow_index"], training["head_component"])), axis=0, return_inverse=True, return_counts=True)
    probability = 1/count[inverse].astype(float); probability /= probability.sum()
    positive = float(probability@training["labels"])
    width = training["features"].shape[1]
    p = candidate["dropout"]
    model = nn.Sequential(nn.Linear(width,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(p),
        nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(p),nn.Linear(128,1)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-positive)/positive, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(),lr=candidate["learning_rate"],weight_decay=candidate["weight_decay"])

    def predict(row):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.tensor(row["features"][i:i+options["batch_size"]], device=device)).squeeze(-1).sigmoid().cpu().numpy()
                                   for i in range(0,len(row["labels"]),options["batch_size"])])

    history, best, state, best_epoch = [], (-1.,-1.), None, 0
    started = time.time()
    for epoch in range(1,options["epochs"]+1):
        warm = options["warmup_epochs"]
        factor = epoch/warm if epoch <= warm else .5*(1+np.cos(np.pi*(epoch-warm)/(options["epochs"]-warm)))
        lr = max(1e-6,candidate["learning_rate"]*factor)
        for group in optimizer.param_groups: group["lr"] = lr
        model.train(); order=rng.choice(len(training["labels"]),len(training["labels"]),replace=True,p=probability); total=0.
        for i in range(0,len(order),options["batch_size"]):
            chosen=order[i:i+options["batch_size"]]; optimizer.zero_grad(set_to_none=True)
            x=torch.tensor(training["features"][chosen],device=device)
            y=torch.tensor(training["labels"][chosen],device=device,dtype=torch.float32)
            smoothing=candidate["label_smoothing"]
            loss=loss_fn(model(x).squeeze(-1),y*(1-smoothing)+smoothing/2)
            if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); total+=float(loss.detach())*len(chosen)
        val_probability=predict(validation); threshold=choose_threshold(validation["labels"],val_probability)
        metrics=binary_metrics(validation["labels"],val_probability,threshold); score=(metrics["f1"],metrics["average_precision"])
        history.append({"epoch":epoch,"training_loss":total/len(order),"validation_f1":score[0],"validation_ap":score[1],"threshold":threshold,"learning_rate":lr})
        if score>best: best,best_epoch,state=score,epoch,copy.deepcopy(model.state_dict())
        if epoch==1 or epoch%10==0: print(json.dumps({"candidate":candidate,**history[-1]}),flush=True)
        if epoch-best_epoch>=options["patience"] and epoch>=warm: break
    model.load_state_dict(state); prediction=predict(validation); threshold=choose_threshold(validation["labels"],prediction)
    ids=rng.choice(len(training["labels"]),min(2048,len(training["labels"])),replace=False)
    subset={k:v[ids] for k,v in training.items()}
    np.savez_compressed(folder/"predictions.npz",probability=prediction,labels=validation["labels"],flow_index=validation["flow_index"],
        head_component=validation["head_component"],scale_id=validation["scale_id"])
    result={"version":spec["version"],"identity":current,"candidate":candidate,"seed":seed,"selected_epoch":best_epoch,
        "threshold":threshold,"validation":grouped_metrics(validation,prediction,threshold,spec),"history":history,
        "training_subset":binary_metrics(subset["labels"],predict(subset),threshold),"test_loaded":False,"weights_written":0,
        "parameters":sum(p.numel() for p in model.parameters()),"seconds":time.time()-started,
        "device":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU","torch":torch.__version__,
        "predictions_sha256":sha(folder/"predictions.npz"),"finished_at_utc":datetime.now(timezone.utc).isoformat()}
    write(folder/"result.json",result)


def merge(spec, config):
    rows=[]
    for candidate in spec["candidates"]:
        folder=Path(spec["output"])/"runs"/candidate["name"]
        row=json.loads((folder/"result.json").read_text())
        if row["predictions_sha256"]!=sha(folder/"predictions.npz") or row["test_loaded"]: raise ValueError("Invalid evidence")
        rows.append({k:v for k,v in row.items() if k!="history"})
    rows.sort(key=lambda r:(r["validation"]["pooled"]["f1"],r["validation"]["pooled"]["average_precision"]),reverse=True)
    write(Path(spec["output"])/"results.json",{"identity":identity(config,spec),"rows":rows,"test_loaded":False})


def runtime(spec,config,phase,state,code=None):
    row={"time":datetime.now(timezone.utc).isoformat(),"phase":phase,"state":state,"exit_code":code,
         "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",**identity(config,spec)}
    append_locked(Path(spec["output"])/"runtime_events.jsonl",json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md","\n- Task4-c directional spectrum runtime: "+json.dumps(row)+"\n")


def submit(spec,config):
    out=Path(spec["output"]).resolve(); out.mkdir(parents=True,exist_ok=False); (out/"logs").mkdir()
    write(out/"config.frozen.json",spec); previous=None
    for phase in ("encode","train","merge"):
        command=["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=16G","--time=00:10:00",
                 f"--job-name=t4c-dir-{phase}",f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
        if phase!="merge": command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
        if previous: command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
        if phase=="train": command += [f"--array=0-{len(spec['candidates'])-1}%4"]
        command += ["ibex_bash/task4c_directional_spectrum_4p1.sh",phase,config]
        previous=subprocess.check_output(command,text=True).strip().split(";")[0]
        row={"job_id":previous,"phase":phase,"version":spec["version"],"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
             "command":command,"expected_device":"CPU" if phase=="merge" else "A100/V100/P100/RTX2080Ti",**identity(config,spec)}
        append_locked(out/"submissions.jsonl",json.dumps(row)+"\n"); append_locked("docs/ibex_run_registry.md","\n- Task4-c directional spectrum SUBMITTED "+json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("encode","train","merge","submit","runtime")); parser.add_argument("--config",default=CONFIG)
    parser.add_argument("--index",type=int,default=0); parser.add_argument("--runtime-phase"); parser.add_argument("--state"); parser.add_argument("--exit-code",type=int)
    args=parser.parse_args(); spec=json.loads(Path(args.config).read_text())
    if args.phase=="train": train(spec,args.config,args.index)
    elif args.phase=="runtime": runtime(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    else: {"encode":encode,"merge":merge,"submit":submit}[args.phase](spec,args.config)


if __name__=="__main__": main()
