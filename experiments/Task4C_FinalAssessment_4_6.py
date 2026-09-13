"""Freeze validation-selected Task4-c methods, then assess the fixed test once."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from experiments.Task4C_HairpinBinary_2_1 import sha, write, append_locked, choose_threshold, binary_metrics
from experiments.Task4C_PaperBundles_3_1 import identity as base_identity
from experiments.Task4C_LinePooling_4_3 import encode_partition

DEFAULT_CONFIG = "config/mainExp_Task4C_FinalAssessment_4.6.json"
METHODS = ("fmt_mlp", "conv3d_mlp")


def identity(config):
    result = base_identity(config)
    for path in (
        "experiments/Task4C_Multiscale_4_1.py", "FMT_Utils/Task4C_Multiscale_4_1.py",
        "experiments/Task4C_Regularized_4_2.py", "FMT_Utils/Task4C_Regularized_4_2.py",
        "experiments/Task4C_LinePooling_4_3.py", "FMT_Utils/Task4C_LinePooling_4_3.py",
        "experiments/Task4C_ManifoldMixup_4_4.py", "FMT_Utils/Task4C_ManifoldMixup_4_4.py",
        "experiments/Task4C_ScaleConsistency_4_5.py", "ibex_bash/task4c_scale_consistency_4p5.sh",
        "experiments/Task4C_FinalAssessment_4_6.py", "ibex_bash/task4c_final_assessment_4p6.sh",
    ):
        result["source_sha256"][path] = sha(path)
    return result


def select_sources(spec):
    """Recompute the three development seeds; do not inspect any test result."""
    rows, sample_identity = [], None
    for source in spec["sources"]:
        root = Path(source["output"])
        selection = json.loads((root/"selection.json").read_text())
        if (selection["identity"]["commit"] != source["commit"]
                or selection["identity"]["config_sha256"] != source["config_sha256"]
                or selection["test_loaded"] or selection["validation_gate_passed"]):
            raise ValueError("Source is changed, unfinished, or has its own final evaluation")
        if sha(source["config"]) != source["config_sha256"]:
            raise ValueError("Source configuration changed")
        for method in METHODS:
            chosen = selection["selected"][method]
            if len(chosen["seeds"]) != 3 or len(set(chosen["seeds"])) != 3:
                raise ValueError("Require three completed development seeds")
            f1, ap = [], []
            for seed in chosen["seeds"]:
                name = f"{chosen['candidate']['name']}_{method}_seed{seed}"
                paths = [p for phase in ("search","confirm") for p in (root/phase).glob(name+"/result.json")]
                if len(paths) != 1:
                    raise ValueError("Missing or duplicate development run")
                run = json.loads(paths[0].read_text())
                predictions = paths[0].parent/"predictions.npz"
                if sha(predictions) != run["predictions_sha256"] or run["test_loaded"]:
                    raise ValueError("Development prediction identity failed")
                if (run["candidate"] != chosen["candidate"] or run["method"] != method or run["seed"] != seed
                        or any(run["identity"][key] != selection["identity"][key]
                               for key in ("commit","config_sha256","source_sha256"))):
                    raise ValueError("Development candidate identity failed")
                with np.load(predictions) as p:
                    current = np.column_stack((p["validation_labels"],p["validation_flow_index"],p["validation_head_component"],p["validation_scale_id"]))
                    if sample_identity is None:
                        sample_identity = current
                    if len(current) != spec["validation_samples"] or not np.array_equal(current,sample_identity):
                        raise ValueError("Sources do not share the fixed validation samples")
                    values = binary_metrics(p["validation_labels"],p["validation_probability"],run["threshold"])
                    if choose_threshold(p["validation_labels"],p["validation_probability"]) != run["threshold"]:
                        raise ValueError("Invalid validation threshold")
                f1.append(values["f1"]); ap.append(values["average_precision"])
            for key,value in (("validation_f1_mean",np.mean(f1)),("validation_f1_std",np.std(f1,ddof=1)),("validation_ap_mean",np.mean(ap))):
                if not np.isclose(value,chosen[key],rtol=0,atol=1e-12):
                    raise ValueError("Source validation aggregation disagrees with predictions")
            rows.append({**copy.deepcopy(chosen),"source":source,
                "source_selection_sha256":sha(root/"selection.json"),"source_encoding_sha256":sha(root/"encoding.json")})
    selected = {m:max([r for r in rows if r["method"]==m],key=lambda r:(r["validation_f1_mean"],r["validation_ap_mean"])) for m in METHODS}
    for row in selected.values():
        if row["source"]["module"] not in spec["supported_training_modules"]:
            raise ValueError("Selected method needs its faithful adapter; never substitute a lower-ranked method")
    return {"version":spec["version"],"selection_rule":"maximum three-seed validation F1, then AP",
        "selected":selected,"candidates":rows,"test_loaded":False,"selection_frozen":True,
        "target_test_f1":spec["target_f1"],"frozen_at_utc":datetime.now(timezone.utc).isoformat()}


def read_selection(spec):
    root = Path(spec["output"])
    frozen = json.loads((root/"selection_lock.json").read_text())
    if (sha(root/"config.frozen.json") != frozen["frozen_config_sha256"]
            or json.loads((root/"config.frozen.json").read_text()) != spec):
        raise ValueError("Final assessment configuration changed after freezing")
    if sha(root/"selection.json") != frozen["selection_sha256"]:
        raise ValueError("Final model selection changed after freezing")
    selection = json.loads((root/"selection.json").read_text())
    if not selection["selection_frozen"] or selection["target_test_f1"] != spec["target_f1"]:
        raise ValueError("Invalid final assessment selection")
    return selection


def prepare(spec, config):
    selection = read_selection(spec)
    out = Path(spec["output"])
    (out/"method_configs").mkdir()
    for method,row in selection["selected"].items():
        source = row["source"]
        source_root = Path(source["output"])
        if sha(source_root/"encoding.json") != row["source_encoding_sha256"]:
            raise ValueError("Source cache manifest changed")
        original = json.loads((source_root/"encoding.json").read_text())
        for path,expected in original["identity"]["source_sha256"].items():
            if sha(path) != expected:
                raise ValueError(f"Frozen training dependency changed: {path}")
        method_spec = json.loads(Path(source["config"]).read_text())
        method_spec["version"] = spec["version"]
        method_spec["output"] = str((out/"models"/method).resolve())
        method_spec["training"]["final_seeds"] = spec["final_seeds"]
        method_spec["assessment"] = {"config":str(Path(config).resolve()),"config_sha256":sha(config),
            "selection_sha256":sha(out/"selection.json"),"source_version":source["version"],"source_commit":source["commit"]}
        method_config = out/"method_configs"/(method+".json")
        write(method_config,method_spec)
        manifest = {"version":spec["version"],"identity":identity(str(method_config)),"splits":{},"test_encoded":False,
            "source_encoding_sha256":row["source_encoding_sha256"]}
        filename = "fmt.npy" if method=="fmt_mlp" else "voxels.npy"
        for split in ("train","validation"):
            for flow in method_spec["flows"]:
                key = f"{flow['name']}/{split}"
                destination = Path(method_spec["output"])/key
                destination.mkdir(parents=True)
                entry = original["splits"][key]
                for name in (filename,"metadata.npz"):
                    if sha(source_root/key/name) != entry["files"][name]:
                        raise ValueError("Source development cache changed")
                    shutil.copyfile(source_root/key/name,destination/name)
                    if sha(destination/name) != entry["files"][name]:
                        raise ValueError("Development cache copy failed")
                manifest["splits"][key] = copy.deepcopy(entry)
        write(Path(method_spec["output"])/"encoding.json",manifest)


def encode_test(spec, config):
    read_selection(spec)
    out = Path(spec["output"])
    temporary_spec = {**spec,"output":str(out/"shared_test_cache")}
    report = {"version":spec["version"],"identity":identity(config),"splits":{}}
    encode_partition(temporary_spec,config,"test",report)
    if sum(row["samples"] for row in report["splits"].values()) != spec["test_samples"]:
        raise ValueError("Fixed test count changed")
    write(out/"test_encoding.json",report)
    for method in METHODS:
        model_out = out/"models"/method
        manifest = json.loads((model_out/"encoding.json").read_text())
        filename = "fmt.npy" if method=="fmt_mlp" else "voxels.npy"
        for flow in spec["flows"]:
            key = f"{flow['name']}/test"
            destination = model_out/key
            destination.mkdir(parents=True)
            for name in (filename,"metadata.npz"):
                shutil.copyfile(out/"shared_test_cache"/key/name,destination/name)
                if sha(destination/name) != report["splits"][key]["files"][name]:
                    raise ValueError("Test cache copy failed")
            manifest["splits"][key] = copy.deepcopy(report["splits"][key])
        manifest["test_encoded"] = True
        write(model_out/"encoding.json",manifest)


def train(spec, config, index):
    selection = read_selection(spec)
    if index < 0 or index >= len(METHODS)*len(spec["final_seeds"]):
        raise ValueError("Invalid final run index")
    method_index,seed_index = divmod(index,len(spec["final_seeds"]))
    method = METHODS[method_index]
    row = selection["selected"][method]
    method_config = Path(spec["output"])/"method_configs"/(method+".json")
    method_spec = json.loads(method_config.read_text())
    if (method_spec["assessment"]["selection_sha256"] != sha(Path(spec["output"])/"selection.json")
            or method_spec["assessment"]["config_sha256"] != sha(config)
            or method_spec["training"]["final_seeds"] != spec["final_seeds"]):
        raise ValueError("Method configuration belongs to another selection")
    module = importlib.import_module(row["source"]["module"])
    plan = lambda _: [(method,row["candidate"],seed) for seed in spec["final_seeds"]]
    # Only orchestration is adapted in this isolated process. The selected
    # version's preprocessing, network, optimizer, loss, and trainer are reused.
    module.identity = identity
    module.final_plan = plan
    if hasattr(module,"pipeline"):
        module.pipeline.identity = identity
        module.pipeline.final_plan = plan
    module.train(method_spec,str(method_config),seed_index,"final")


def merge(spec, config):
    selection = read_selection(spec)
    out = Path(spec["output"])
    rows,reference = [],None
    for method in METHODS:
        row = selection["selected"][method]
        for seed in spec["final_seeds"]:
            folder = out/"models"/method/"final"/f"{row['candidate']['name']}_{method}_seed{seed}"
            result = json.loads((folder/"result.json").read_text())
            method_config = out/"method_configs"/(method+".json")
            expected_identity = identity(str(method_config))
            if (result["candidate"] != row["candidate"] or result["seed"] != seed
                    or result["method"] != method or result["weights_written"] != 0
                    or result["phase"] != "final" or "test" not in result
                    or any(result["identity"][key] != expected_identity[key]
                           for key in ("commit","config_sha256","source_sha256"))):
                raise ValueError("Final run differs from frozen selection")
            if sha(folder/"predictions.npz") != result["predictions_sha256"]:
                raise ValueError("Final prediction hash mismatch")
            with np.load(folder/"predictions.npz") as p:
                sample_ids = np.column_stack((p["labels"],p["flow_index"],p["head_component"],p["scale_id"]))
                if reference is None: reference=sample_ids
                if len(sample_ids) != spec["test_samples"] or not np.array_equal(sample_ids,reference):
                    raise ValueError("Final methods used different test samples")
                if choose_threshold(p["validation_labels"],p["validation_probability"]) != result["threshold"]:
                    raise ValueError("Threshold was not selected exclusively on validation")
                values = binary_metrics(p["labels"],p["probability"],result["threshold"])
                for key in ("f1","average_precision","balanced_accuracy"):
                    if not np.isclose(values[key],result["test"]["pooled"][key],atol=1e-12,rtol=0):
                        raise ValueError("Final saved prediction metrics disagree")
            rows.append({**{k:v for k,v in result.items() if k!='history'},"source_version":row["source"]["version"]})
    summary={m:{k:{"mean":float(np.mean([r["test"]["pooled"][k] for r in rows if r["method"]==m])),
                     "sample_std":float(np.std([r["test"]["pooled"][k] for r in rows if r["method"]==m],ddof=1))}
                for k in ("f1","average_precision","balanced_accuracy")} for m in METHODS}
    report={"version":spec["version"],"identity":identity(config),"selection_sha256":sha(out/"selection.json"),
        "summary":summary,"runs":rows,"target_reached":all(r['f1']['mean']>=spec['target_f1'] for r in summary.values()),
        "completed_at_utc":datetime.now(timezone.utc).isoformat()}
    write(out/"results.json",report)
    print(json.dumps({"summary":summary,"target_reached":report["target_reached"]}),flush=True)


def runtime(spec,config,phase,state,code=None):
    device="CPU"
    if phase in ("encode_test","train"):
        import torch
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"
    row={"time":datetime.now(timezone.utc).isoformat(),"phase":phase,"state":state,"exit_code":code,"device":device,**identity(config)}
    append_locked(Path(spec["output"])/"runtime_events.jsonl",json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.6 runtime: "+json.dumps(row)+"\n")


def submit_job(spec,config,phase,array=None,previous=None):
    gpu=phase in ("encode_test","train")
    out=Path(spec["output"]).resolve()
    command=["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=32G",
        "--time="+("00:45:00" if phase=="train" else "00:15:00"),f"--job-name=t4c46-{phase}",
        f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous: command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
    if gpu: command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
    if array is not None: command += [f"--array=0-{array-1}%6"]
    command += ["ibex_bash/task4c_final_assessment_4p6.sh",phase,config]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row={"job_id":job,"phase":phase,"version":spec["version"],"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
        "command":command,"dependency":previous,"expected_device":"A100/V100/P100/RTX2080Ti" if gpu else "CPU",**identity(config)}
    append_locked(out/"submissions.jsonl",json.dumps(row)+"\n")
    append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.6 SUBMITTED "+json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
    return job


def submit(spec,config):
    if len(spec["final_seeds"]) != 3 or len(set(spec["final_seeds"])) != 3:
        raise ValueError("Exactly three distinct final seeds are required")
    selected=select_sources(spec)
    development_seeds={seed for row in selected["candidates"] for seed in row["seeds"]}
    if development_seeds.intersection(spec["final_seeds"]):
        raise ValueError("Final seeds must be fresh")
    out=Path(spec["output"])
    out.mkdir(parents=True,exist_ok=False);(out/"logs").mkdir()
    write(out/"config.frozen.json",spec)
    write(out/"selection.json",selected)
    write(out/"selection_lock.json",{"selection_sha256":sha(out/"selection.json"),"config_sha256":sha(config),
                                      "frozen_config_sha256":sha(out/"config.frozen.json")})
    previous=None
    for phase,count in (("prepare",None),("encode_test",None),("train",6),("merge",None)):
        previous=submit_job(spec,config,phase,count,previous)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('submit','prepare','encode_test','train','merge','runtime'))
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text())
    if a.phase=='runtime':runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    elif a.phase=='train':train(spec,a.config,a.index)
    else:{'submit':submit,'prepare':prepare,'encode_test':encode_test,'merge':merge}[a.phase](spec,a.config)


if __name__=='__main__':main()
