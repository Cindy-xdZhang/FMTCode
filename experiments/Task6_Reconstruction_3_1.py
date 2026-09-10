"""Task6 signed-Fourier VAE recovery with train-only initialization and validation selection."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256,write_json
from FMT_Utils.PrimitiveVAE_3D import vae_loss,geometry_metrics
from FMT_Utils.Task6Recovery_3D import signed_fmt,initialize_vae
from experiments.Task6_PrimitiveVAE_2_1 import provenance,predict

DEFAULT_CONFIG="config/mainExp_Task6_Reconstruction_3.1.json"


def fit(spec,x,y,vx,vy,arm,seed,device,folder,fixed_steps=None):
    cfg=spec["vae"]
    torch.manual_seed(seed)
    if device.type=="cuda": torch.cuda.manual_seed_all(seed)
    model,initialization=initialize_vae(x,y,arm,cfg["latent_dim"],spec["frequencies"],cfg["width"],cfg["blocks"])
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=0.)
    xt,yt=torch.tensor(x,device=device),torch.tensor(y,device=device)
    batch=cfg["batch_size"]
    per_epoch=math.ceil(len(x)/batch)
    total=fixed_steps if fixed_steps is not None else cfg["epochs"]*per_epoch
    rng=np.random.default_rng(seed)
    probe=np.sort(rng.choice(len(x),min(4096,len(x)),replace=False))
    history=[]; best_state=None; best_value=float("inf"); best_step=None
    start=time.monotonic()
    def record(step,rec=None,kl=None):
        nonlocal best_state,best_value,best_step
        train_score=geometry_metrics(predict(model,x[probe],device),y[probe])["position_rmse_r"]
        validation_score=geometry_metrics(predict(model,vx,device),vy)["position_rmse_r"]
        row=dict(step=step,train_probe_rmse_r=train_score,validation_rmse_r=validation_score,
            geometry_loss=rec,kl=kl,seconds=time.monotonic()-start)
        history.append(row)
        with (folder/"curve.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(row,allow_nan=False)+"\n")
        if step>0 and validation_score<best_value:
            best_value=validation_score; best_step=step
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps(row),flush=True)
    record(0)
    for step in range(total):
        if step%per_epoch==0: order=rng.permutation(len(x))
        index=order[(step%per_epoch)*batch:((step%per_epoch)+1)*batch]
        index=torch.as_tensor(index,device=device)
        rate=cfg["learning_rate"]*(.1+.9*.5*(1+math.cos(math.pi*step/max(1,total-1))))
        for group in optimizer.param_groups: group["lr"]=rate
        model.train(); optimizer.zero_grad(set_to_none=True)
        output,mu,logvar=model(xt[index],sample=True)
        loss,reconstruction,kl=vae_loss(output,yt[index],mu,logvar,cfg["beta"])
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite VAE training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"])
        optimizer.step()
        if (step+1)%cfg["probe_every"]==0 or step+1==total:
            record(step+1,float(reconstruction.detach()),float(kl.detach()))
    assert best_state is not None and best_step>0
    model.load_state_dict(best_state)
    report=dict(initialization=initialization,curve=history,updates=total,selected_step=best_step,
        selected_validation_rmse_r=best_value,selection="minimum validation RMSE among positive-step probes; no test reads",
        epochs_exposed=total/per_epoch,train_examples_exposed=(total//per_epoch)*len(x)+min(len(x),(total%per_epoch)*batch),
        final_posterior_logvar=model.posterior_logvar.detach().cpu().numpy().tolist(),
        selected_train_probe_rmse_r=geometry_metrics(predict(model,x[probe],device),y[probe])["position_rmse_r"],
        elapsed_seconds=time.monotonic()-start)
    return model,report


def train(spec,config,index):
    jobs=[(d,a,s) for d in spec["datasets"] for a in spec["arms"] for s in spec["seeds"]]
    dataset,arm,seed=jobs[index]
    root=Path(spec["output_root"]); data_dir=root/"data"/dataset
    audit=json.loads((data_dir/"audit.json").read_text())
    assert audit["passed"] and audit["provenance"]["config_sha256"]==sha256(config)
    manifest=json.loads((data_dir/"manifest.json").read_text())
    assert audit["manifest_sha256"]==sha256(data_dir/"manifest.json")
    folder=root/"runs"/dataset/arm/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    if list(folder.iterdir()): raise FileExistsError(folder)
    if not torch.cuda.is_available(): raise RuntimeError("Main training requires GPU")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda")
    metadata=dict(experiment=spec["experiment"],dataset=dataset,arm=arm,seed=seed,
        provenance=provenance(config),device=torch.cuda.get_device_name(0),torch_version=torch.__version__,
        vae=spec["vae"],frequencies=spec["frequencies"],data_manifest_sha256=sha256(data_dir/"manifest.json"))
    write_json(folder/"started.json",metadata)
    def load(role):
        if role=="legacy_test":
            legacy=Path(spec["legacy_benchmark_cache"])/"data"/dataset
            source_manifest=json.loads((legacy/"manifest.json").read_text())
            path=legacy/"test.npz"
            assert sha256(path)==source_manifest["files"]["test"]["sha256"]
        else:
            path=data_dir/f"{role}.npz"
            assert sha256(path)==manifest["files"][role]["sha256"]
        with np.load(path) as cache:
            geometry=cache["geometry"].copy()
            if arm=="signed_fmt_vae": inputs=signed_fmt(geometry,spec["frequencies"]).astype(np.float32)
            elif arm=="raw_vae": inputs=geometry.reshape(len(geometry),672).copy()
            else: inputs=cache["token"].copy()
            return inputs,geometry,cache["scale_id"].copy()
    x,y,_=load("train"); vx,vy,_=load("validation")
    selected=np.sort(np.random.default_rng(spec["sampling"]["data_seed"]).choice(len(x),spec["vae"]["fit_samples"],replace=False))
    fit_dir=folder/"fit_check"; fit_dir.mkdir()
    model,small=fit(spec,x[selected],y[selected],x[selected],y[selected],arm,seed,device,fit_dir,spec["vae"]["fit_steps"])
    small["metric"]=geometry_metrics(predict(model,x[selected],device),y[selected])
    write_json(fit_dir/"result.json",{**metadata,**small,"validation_role":"same training subset, not held out"})
    del model; torch.cuda.empty_cache()
    model,training=fit(spec,x,y,vx,vy,arm,seed,device,folder)
    # Do not consume a fresh test split while the validation reconstruction is
    # still an unresolved engineering failure.
    if training["selected_validation_rmse_r"] >= spec["validation_gate_rmse_r"]:
        write_json(folder/"validation_gate_failed.json",{**metadata,"training":training})
        raise RuntimeError("Validation gate failed; test has not been read")
    rows=[]
    # Test is read for the first time only after validation selection has ended.
    for role in ("test","unseen_scale","legacy_test"):
        tx,ty,scale=load(role)
        prediction=predict(model,tx,device)
        np.savez(folder/f"{role}_predictions.npz",prediction=prediction)
        for sid in [-1]+sorted(np.unique(scale).tolist()):
            mask=np.ones(len(ty),bool) if sid==-1 else scale==sid
            rows.append(dict(role=role,scale_id=sid,**geometry_metrics(prediction[mask],ty[mask])))
    write_json(folder/"result.json",{**metadata,"training":training,"fit_check":small["metric"],"metrics":rows,
        "decoder_input":"latent only; no raw-geometry or inverse-Fourier output bypass",
        "model_files":"none; best state kept in memory only"})


def audit_results(spec,config):
    root=Path(spec["output_root"]); rows=[]; completed=[]
    for dataset in spec["datasets"]:
        for arm in spec["arms"]:
            for seed in spec["seeds"]:
                folder=root/"runs"/dataset/arm/str(seed)
                result=json.loads((folder/"result.json").read_text())
                assert result["provenance"]["config_sha256"]==sha256(config)
                assert result["training"]["selected_step"]>0 and result["vae"]["beta"]>0
                for role in ("test","unseen_scale","legacy_test"):
                    source=Path(spec["legacy_benchmark_cache"])/"data"/dataset/"test.npz" if role=="legacy_test" else root/"data"/dataset/f"{role}.npz"
                    with np.load(source) as cache: truth=cache["geometry"]; scale=cache["scale_id"]
                    with np.load(folder/f"{role}_predictions.npz") as data: prediction=data["prediction"]
                    for metric in [m for m in result["metrics"] if m["role"]==role]:
                        sid=metric["scale_id"]; mask=np.ones(len(truth),bool) if sid==-1 else scale==sid
                        fresh=geometry_metrics(prediction[mask],truth[mask])
                        # Independent scalar reduction of the primary metric.
                        diff=prediction[mask].astype(np.float64)[:,:,1:]-truth[mask].astype(np.float64)[:,:,1:]
                        separate=float(np.sqrt(np.sum(diff*diff)/(len(diff)*7*31)))
                        np.testing.assert_allclose(separate,fresh["position_rmse_r"],rtol=1e-12)
                        for key,value in fresh.items(): np.testing.assert_allclose(metric[key],value,rtol=1e-9,atol=1e-10)
                        rows.append(dict(dataset=dataset,arm=arm,seed=seed,**{k:v for k,v in metric.items() if not isinstance(v,list)}))
                completed.append(dict(dataset=dataset,arm=arm,seed=seed,selected_step=result["training"]["selected_step"],
                    validation_rmse_r=result["training"]["selected_validation_rmse_r"],
                    fit_rmse_r=result["fit_check"]["position_rmse_r"],device=result["device"]))
    for suffix in ("*.pt","*.pth","*.ckpt"): assert not any(root.rglob(suffix))
    with (root/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    primary=[r for r in rows if r["arm"]=="signed_fmt_vae" and r["role"]=="test" and r["scale_id"]==-1]
    failures=[r for r in primary if r["position_rmse_r"]>=1.]
    write_json(root/"final_audit.json",dict(passed=True,rows=len(rows),runs=len(completed),provenance=provenance(config),
        goal_all_signed_fmt_runs_below_one=not failures,failures=failures,runs_metadata=completed))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=["train","audit-results"])
    parser.add_argument("--config",default=DEFAULT_CONFIG); parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args(); spec=json.loads(Path(args.config).read_text())
    if args.phase=="train": train(spec,args.config,args.index)
    else: audit_results(spec,args.config)
