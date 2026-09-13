"""Training/validation ablation of signed FFT bandwidth and voxel resolution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn

import experiments.Task4C_LinePooling_4_3 as pipeline
import FMT_Utils.Task4C_LinePooling_4_3 as networks
from experiments.Task4C_HairpinBinary_2_1 import sha, write, append_locked
from experiments.Task4C_PaperBundles_3_1 import identity as base_identity
from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_voxels

DEFAULT_CONFIG = "config/Ablation_Task4C_RepresentationResolution_4.7.json"


def identity(config):
    result = base_identity(config)
    for name in (
        "experiments/Task4C_Multiscale_4_1.py", "FMT_Utils/Task4C_Multiscale_4_1.py",
        "FMT_Utils/Task4C_Regularized_4_2.py", "experiments/Task4C_LinePooling_4_3.py",
        "FMT_Utils/Task4C_LinePooling_4_3.py", "experiments/Task4C_RepresentationResolution_4_7.py",
        "ibex_bash/task4c_representation_resolution_4p7.sh",
    ):
        result["source_sha256"][name] = sha(name)
    return result


@torch.no_grad()
def full_line_fmt(geometry, seeds, counts):
    old = networks.line_fmt(geometry, seeds, counts)
    delta = geometry.diff(dim=2)
    tangent = delta/delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = torch.cat((tangent, tangent[:, :, -1:]), dim=2)
    sequence = torch.cat((geometry, tangent), -1)
    signed = torch.view_as_real(torch.fft.rfft(sequence, dim=2, norm="ortho")).flatten(2)
    signed *= old[..., -1:]
    if not torch.equal(signed[..., :72], old[..., 161:233]):
        raise ValueError("Low-frequency coefficients changed")
    result = torch.cat((old[..., :161], signed, old[..., -1:]), -1)
    if result.shape[-1] != 366 or not torch.isfinite(result).all():
        raise ValueError("Invalid full-frequency line representation")
    return result


class MappedFeatures:
    """Concatenated read-only arrays, loading only requested rows into memory."""
    def __init__(self, paths):
        self.arrays = [np.load(p, mmap_mode="r") for p in paths]
        self.ends = np.cumsum([len(a) for a in self.arrays])
        first = self.arrays[0]
        if any(a.shape[1:] != first.shape[1:] or a.dtype != first.dtype for a in self.arrays):
            raise ValueError("Incompatible feature partitions")
        self.shape = (int(self.ends[-1]), *first.shape[1:])
        self.ndim, self.dtype = len(self.shape), first.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        ids = np.arange(len(self))[index]
        scalar = np.ndim(ids) == 0
        ids = np.atleast_1d(ids)
        result = np.empty((len(ids), *self.shape[1:]), dtype=self.dtype)
        begin = 0
        for end, array in zip(self.ends, self.arrays):
            selected = (ids >= begin) & (ids < end)
            result[selected] = array[ids[selected]-begin]
            begin = end
        return result[0] if scalar else result


class ExpandedFMT(nn.Module):
    """Identical parameter count for six and seventeen signed frequency bins."""
    def __init__(self, dropout):
        super().__init__()
        self.line = nn.Sequential(nn.Linear(365,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128,128), nn.LayerNorm(128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(256,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128,64), nn.GELU(), nn.Linear(64,2))

    def forward(self, tokens):
        mask = tokens[..., -1] > .5
        values = self.line(tokens[..., :-1])
        mean = (values*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        maximum = values.masked_fill(~mask[..., None], -torch.inf).amax(1)
        return self.head(torch.cat((mean, maximum), -1))


def make_model(method, candidate):
    return ExpandedFMT(candidate["dropout"]) if method == "fmt_mlp" else networks.WiderConv3D(candidate["dropout"])


def load_data(spec, split, method, manifest, candidate):
    if split not in ("train", "validation"):
        raise ValueError("This ablation does not open test data")
    filename = "fmt.npy" if method == "fmt_mlp" else f"voxels{candidate['voxel_resolution']}.npy"
    paths, labels, heads, instances, flows, scales = [], [], [], [], [], []
    for index, flow in enumerate(spec["flows"]):
        key = flow["name"]+"/"+split
        root = Path(spec["output"])/key
        for name in (filename, "metadata.npz"):
            if sha(root/name) != manifest["splits"][key]["files"][name]:
                raise ValueError("Encoded data changed")
        paths.append(root/filename)
        with np.load(root/"metadata.npz") as data:
            labels.append(data["labels"]); heads.append(data["head_component"]); instances.append(data["instance"])
            flows.append(np.full(len(data["labels"]), index, np.int8)); scales.append(data["scale_id"])
    if method == "fmt_mlp":
        features = np.concatenate([np.load(p) for p in paths])
        if candidate["signed_frequency_bins"] == 6:
            features[..., 233:365] = 0
        elif candidate["signed_frequency_bins"] != 17:
            raise ValueError("Unregistered frequency count")
    else:
        features = MappedFeatures(paths)
    return {"features":features, "labels":np.concatenate(labels), "head_component":np.concatenate(heads),
            "instance":np.concatenate(instances), "flow_index":np.concatenate(flows), "scale_id":np.concatenate(scales)}


def encode(spec, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SLURM_JOB_ID") and device.type != "cuda":
        raise RuntimeError("Allocated GPU unavailable")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    report = {"version":spec["version"], "identity":identity(config), "splits":{}, "test_encoded":False}
    for split in ("train", "validation"):
        for flow in spec["flows"]:
            key = flow["name"]+"/"+split
            source, dest = Path(spec["parent_output"])/key, Path(spec["output"])/key
            prep = json.loads((source.parent/"preparation.json").read_text())
            if not prep["spatial_isolation_verified"] or prep["identity"]["config_sha256"] != spec["parent_config_sha256"]:
                raise ValueError("Physical data is not the frozen 4.1 data")
            for name in ("geometry.npy", "seeds.npy", "metadata.npz"):
                if sha(source/name) != prep["splits"][split]["files"][name]:
                    raise ValueError("Frozen physical data changed")
            dest.mkdir(parents=True, exist_ok=False)
            (dest/"metadata.npz").write_bytes((source/"metadata.npz").read_bytes())
            geometry = np.load(source/"geometry.npy", mmap_mode="r")
            seeds = np.load(source/"seeds.npy", mmap_mode="r")
            with np.load(source/"metadata.npz") as data:
                counts, labels = data["counts"], data["labels"]
            n = len(geometry)
            fmt = np.lib.format.open_memmap(dest/"fmt.npy", mode="w+", dtype="float32", shape=(n,27,366))
            grids = {resolution:np.lib.format.open_memmap(dest/f"voxels{resolution}.npy", mode="w+",
                       dtype="float16", shape=(n,4,resolution,resolution,resolution)) for resolution in (24,48)}
            for start in range(0,n,8):
                sl = slice(start,start+8)
                g = torch.tensor(np.array(geometry[sl]), device=device)
                se = torch.tensor(np.array(seeds[sl]), device=device)
                co = torch.tensor(counts[sl].astype(np.int64), device=device)
                fmt[sl] = full_line_fmt(g,se,co).cpu().numpy()
                for resolution, grid in grids.items():
                    grid[sl] = bundle_voxels(g,co,resolution=resolution).cpu().numpy().astype(np.float16)
                if start % 1600 == 0: print("Encoded",key,start,n,flush=True)
            fmt.flush()
            for grid in grids.values(): grid.flush()
            report["splits"][key] = {"samples":n, "class_counts":np.bincount(labels,minlength=2).tolist(),
                "files":{name:sha(dest/name) for name in ("fmt.npy","voxels24.npy","voxels48.npy","metadata.npz")}}
    write(Path(spec["output"])/"encoding.json",report)


def smoke(spec, config):
    """Real allocated-device forward/backward; no fitting or test scoring."""
    if not torch.cuda.is_available(): raise RuntimeError("GPU smoke check requires CUDA")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    manifest = json.loads((Path(spec["output"])/"encoding.json").read_text())
    records=[]
    for resolution in (24,48):
        candidate = next(c for c in spec["candidates"] if c["voxel_resolution"] == resolution)
        data = load_data(spec,"train","conv3d_mlp",manifest,candidate)
        x = torch.tensor(data["features"][:spec["training"]["batch_size"]], device="cuda", dtype=torch.float32)
        torch.manual_seed(95901)
        model = make_model("conv3d_mlp",candidate).cuda().train()
        torch.cuda.reset_peak_memory_stats()
        loss = model(x).square().mean();loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("GPU forward/backward is nonfinite")
        records.append({"resolution":resolution,"batch_size":len(x),"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
        del model,x,loss,data
        torch.cuda.empty_cache()
    write(Path(spec["output"])/"device_smoke.json",{"identity":identity(config),"device":torch.cuda.get_device_name(0),
         "rows":records,"test_loaded":False,"purpose":"memory and numerical check, not performance"})


def rank(spec, config):
    records=[pipeline.read_run(spec,"search",*item) for item in pipeline.development_plan(spec)]
    top={}
    for method in spec["methods"]:
        field="signed_frequency_bins" if method=="fmt_mlp" else "voxel_resolution"
        levels=(6,17) if method=="fmt_mlp" else (24,48)
        top[method]=[max([r for r in records if r["method"]==method and r["candidate"][field]==level],
            key=lambda r:(r["validation"]["pooled"]["f1"],r["validation"]["pooled"]["average_precision"]))["candidate"]
            for level in levels]
    write(Path(spec["output"])/"development_rank.json",{"identity":identity(config),"top_two":top,
        "rule":"best learning rate within each representation; confirm both representations",
        "rows":[{k:r[k] for k in ("method","candidate","seed","validation","training_subset")} for r in records],"test_loaded":False})


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
    write(Path(spec["output"])/"selection.json",report)
    print(json.dumps({"selected":chosen,"test_loaded":False}),flush=True)


def runtime(spec,config,phase,state,code=None):
    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    row={"time":datetime.now(timezone.utc).isoformat(),"phase":phase,"state":state,"exit_code":code,"device":device,**identity(config)}
    append_locked(Path(spec["output"])/"runtime_events.jsonl",json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.7 runtime: "+json.dumps(row)+"\n")


def submit_job(spec,config,phase,array=None,previous=None):
    gpu=phase in ("encode","smoke","search","confirm")
    out=Path(spec["output"]).resolve()
    limit="00:30:00" if phase=="encode" else "02:00:00" if phase in ("search","confirm") else "00:10:00"
    command=["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=32G",f"--time={limit}",
        f"--job-name=t4c47-{phase}",f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
    if gpu:command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
    if array is not None:command += [f"--array=0-{array-1}%6"]
    command += ["ibex_bash/task4c_representation_resolution_4p7.sh",phase,config]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row={"job_id":job,"phase":phase,"version":spec["version"],"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
        "command":command,"dependency":previous,"expected_device":"A100/V100/P100/RTX2080Ti" if gpu else "CPU",**identity(config)}
    append_locked(out/"submissions.jsonl",json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.7 SUBMITTED "+json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
    return job


def submit(spec,config):
    out=Path(spec["output"]);out.mkdir(parents=True,exist_ok=False);(out/"logs").mkdir()
    write(out/"config.frozen.json",spec)
    previous=None
    for phase,count in (("encode",None),("smoke",None),("search",len(pipeline.development_plan(spec))),
                        ("rank",None),("confirm",8),("select",None)):
        previous=submit_job(spec,config,phase,count,previous)


def configure_runtime():
    pipeline.identity=identity
    pipeline.load_data=load_data
    networks.make_model=make_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("encode","smoke","search","rank","confirm","select","submit","runtime"))
    p.add_argument("--config",default=DEFAULT_CONFIG);p.add_argument("--index",type=int,default=0)
    p.add_argument("--runtime-phase");p.add_argument("--state");p.add_argument("--exit-code",type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text());configure_runtime()
    if a.phase in ("search","confirm"):pipeline.train(spec,a.config,a.index,a.phase)
    elif a.phase=="runtime":runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{"encode":encode,"smoke":smoke,"rank":rank,"select":select,"submit":submit}[a.phase](spec,a.config)


if __name__=="__main__":main()
