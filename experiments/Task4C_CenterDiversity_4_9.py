"""Task4-c 4.9: more training centers at exactly matched head/scale quotas."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

from experiments import Task4C_Sharpness_4_8 as sharpness
from experiments import Task4C_Multiscale_4_1 as physical
from experiments import Task4C_LinePooling_4_3 as encoder

pipeline = sharpness.pipeline
FROZEN_IDENTITY = sharpness.identity
FROZEN_READ_RUN = pipeline.read_run
DEFAULT_CONFIG = "config/Ablation_Task4C_CenterDiversity_4.9.json"


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    if "base_config" not in definition:
        return definition  # Explicit full engineering fixtures.
    if pipeline.sha(definition["base_config"]) != definition["base_config_sha256"]:
        raise ValueError("Frozen base configuration changed")
    if "candidates" in definition and "training" in definition:
        return definition  # Expanded frozen config or engineering fixture.
    spec = json.loads(Path(definition["base_config"]).read_text())
    for key in ("version","output","diversity","arms","target"):
        spec[key] = definition[key]
    spec["training"].update(definition["training_overrides"])
    base = spec["candidates"][0]
    spec["candidates"] = [dict(base,name="candidate_"+str(i).zfill(2),data_arm=arm,sam_rho=0.)
                          for i,arm in enumerate(spec["arms"])]
    spec["regularization"]["rho_candidates"] = [0.]
    spec["base_config"] = definition["base_config"]
    spec["base_config_sha256"] = definition["base_config_sha256"]
    return spec


def identity(config):
    result = FROZEN_IDENTITY(config)
    for p in ("experiments/Task4C_CenterDiversity_4_9.py","ibex_bash/task4c_center_diversity_4p9.sh"):
        result["source_sha256"][p] = pipeline.sha(p)
    definition=json.loads(Path(config).read_text())
    if "base_config" in definition:
        result["source_sha256"][definition["base_config"]] = pipeline.sha(definition["base_config"])
    return result


def retention_plan(metadata, maximum, seed):
    """Select existing views without inspecting labels or model predictions."""
    if maximum not in (1,2):
        raise ValueError("This protocol keeps at most two old views per center")
    keys=np.column_stack((metadata["head_component"],metadata["center_number"]))
    unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    keep=np.zeros(len(keys),bool)
    for i,(head,center) in enumerate(unique):
        ids=np.flatnonzero(inverse==i)
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(head),int(center)]))
        keep[rng.choice(ids,min(len(ids),maximum),replace=False)]=True
    return np.flatnonzero(keep),np.flatnonzero(~keep)


def scale_pair(remaining, rng):
    available=np.flatnonzero(remaining>0)
    tie=rng.random(len(available))
    order=np.lexsort((tie,-remaining[available]))
    return available[order[:2]].tolist()


def quota_table(metadata, nscales):
    return {int(h):np.bincount(metadata["scale_id"][metadata["head_component"]==h],minlength=nscales)
            for h in np.unique(metadata["head_component"])}


def accept_complete_groups(groups, accepted, remaining):
    """Apply quota changes atomically only after every requested view is valid."""
    rows=[];failed=0
    for head,number,chosen in groups:
        keys=[(head,number,scale) for scale in chosen]
        if not all(key in accepted for key in keys):
            failed+=1;continue
        if len(set(chosen))!=len(chosen) or any(remaining[head][scale]<=0 for scale in chosen):
            raise ValueError("Invalid or exhausted scale quota")
        for key in keys:
            rows.append(accepted[key]);remaining[head][key[2]]-=1
    return rows,failed


def audit_rows(rows, original, retained, old_geometry, old_seeds, max_views):
    """Check the actual generated records before the mutating save_split call."""
    keys=np.array([(r["head_component"],r["center_number"],r["scale_id"]) for r in rows])
    if len(keys)!=len(original["labels"]) or len(np.unique(keys,axis=0))!=len(keys):
        raise ValueError("Sample count or center/scale uniqueness changed")
    for head in np.unique(original["head_component"]):
        ids=np.flatnonzero(original["head_component"]==head)
        group=[r for r in rows if r["head_component"]==head]
        if sorted(r["scale_id"] for r in group)!=sorted(original["scale_id"][ids].tolist()):
            raise ValueError("Head/scale quota changed")
        if any(r["label"]!=int(original["labels"][ids[0]]) or r["instance"]!=int(original["instance"][ids[0]]) for r in group):
            raise ValueError("Frozen head label/instance changed")
    _,first,inverse,counts=np.unique(keys[:,:2],axis=0,return_index=True,return_inverse=True,return_counts=True)
    if counts.max()>max_views:
        raise ValueError("Too many views at one center")
    centers=np.array([r["center"] for r in rows])
    if not np.all(centers==centers[first[inverse]]) or len(np.unique(centers[first],axis=0))!=len(first):
        raise ValueError("Inconsistent or duplicated physical center")
    lookup={tuple(k):r for k,r in zip(keys,rows)}
    for i in retained:
        key=tuple(int(original[k][i]) for k in ("head_component","center_number","scale_id"))
        r=lookup[key]
        if not np.array_equal(r["geometry"],old_geometry[i]) or not np.array_equal(r["normalized_seeds"],old_seeds[i]):
            raise ValueError("Retained physical view changed")
    return {"samples":len(rows),"centers":len(first),"paired_centers":int((counts==2).sum()),
            "singleton_centers":int((counts==1).sum()),"retained_views":len(retained),
            "head_scale_label_instance_counts_identical":True,"retained_geometry_exact":True}


def prepare(spec,config,index,input_root=None):
    from scipy import ndimage
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow,native_partition
    from FMT_Utils.Task4C_Multiscale_4_1 import extract_native_partition,center_and_neighbors,trace_primitive_batch
    flow=spec["flows"][index];name=flow["name"]
    parent=Path(spec["parent_output"])/name
    previous=json.loads((parent/"preparation.json").read_text())
    if not previous["spatial_isolation_verified"] or previous["identity"]["config_sha256"]!=spec["parent_config_sha256"]:
        raise ValueError("Unexpected original physical dataset")
    for path,digest in previous["identity"]["source_sha256"].items():
        if pipeline.sha(path)!=digest:raise ValueError("Original generator source changed: "+path)
    for filename in ("geometry.npy","seeds.npy","metadata.npz"):
        if pipeline.sha(parent/"train"/filename)!=previous["splits"]["train"]["files"][filename]:
            raise ValueError("Original training data changed")
    with np.load(parent/"train/metadata.npz") as m:metadata={k:m[k] for k in m.files}
    if previous["flow"]!=flow:
        raise ValueError("Frozen flow/label configuration changed")
    geometry=np.load(parent/"train/geometry.npy",mmap_mode="r")
    seeds=np.load(parent/"train/seeds.npy",mmap_mode="r")
    options=spec["diversity"];scales=spec["scale_sets"]["train"]
    if options["maximum_views_per_new_center"]!=2 or options["new_center_start"]<=int(metadata["center_number"].max()):
        raise ValueError("Expected two-view new centers outside the original number range")
    retained,replaced=retention_plan(metadata,options["keep_views_per_original_center"],options["selection_seed"]+index)
    remaining=quota_table({k:v[replaced] for k,v in metadata.items()},len(scales))
    out=Path(spec["output"])/"physical"/name;out.mkdir(parents=True,exist_ok=False)
    root=Path(input_root or spec["input_root"])
    if pipeline.sha(root/flow["flow"])!=flow["flow_sha256"]:
        raise ValueError("Frozen flow input changed")
    axes,_,heads,oyf,grid,source=load_flow(root/flow["flow"],flow["lambda2_threshold"])
    components,ncomponents=ndimage.label(heads)
    if ncomponents!=previous["catalog"]["head_components"]:
        raise ValueError("Native head component identities changed")
    left,right,support=native_partition(axes,np.array(previous["catalog"]["x_cuts"]),0,
                                      int(source["vorticity_source"].startswith("second")))
    if support!=previous["splits"]["train"]["native_source_x_nodes_including_curl_stencil"]:
        raise ValueError("Training native source support changed")
    mesh=extract_native_partition(grid,axes,left,right);wall_span=float(axes[2][-1]-axes[2][0])
    flat=np.flatnonzero(heads);component=components.ravel()[flat];order=np.argsort(component,kind="stable")
    flat,component=flat[order],component[order]
    descriptors={}
    for head in remaining:
        a,b=np.searchsorted(component,head,"left"),np.searchsorted(component,head,"right")
        ids=np.flatnonzero(metadata["head_component"]==head)
        cells=flat[a:b];xs=np.unravel_index(cells,heads.shape)[2]
        if not np.all(components.ravel()[metadata["source_cell"][ids]]==head):
            raise ValueError("Original seed cell component identity changed")
        if len(cells)<10 or xs.min()<left or xs.max()>=right:
            raise ValueError("A frozen training head lies outside its source partition")
        if len(np.unique(metadata["labels"][ids]))!=1 or len(np.unique(metadata["instance"][ids]))!=1:
            raise ValueError("Original head has inconsistent labels")
        descriptors[head]={"head_component":head,"cell_ids":cells,"label":int(metadata["labels"][ids[0]]),
                           "instance":int(metadata["instance"][ids[0]])}
    rows=[]
    for i in retained:
        r={k:metadata[k][i].copy() for k in metadata if k not in ("labels","counts")}
        r.update(label=int(metadata["labels"][i]),line_count=int(metadata["counts"][i]),
                 geometry=np.array(geometry[i]),normalized_seeds=np.array(seeds[i]))
        rows.append(r)
    rng=np.random.default_rng(options["selection_seed"]+index*1000)
    attempts={h:0 for h in remaining};rejections={};cleaning=[]
    def reject(reason,count=1):rejections[reason]=rejections.get(reason,0)+count
    while any(v.sum() for v in remaining.values()):
        groups=[];batches={s:[] for s in range(len(scales))}
        for head in rng.permutation(list(remaining)):
            head=int(head)
            if not remaining[head].sum():continue
            attempts[head]+=1
            if attempts[head]>options["maximum_attempts_per_head"]:
                raise ValueError("Exhausted bounded center attempts for head "+str(head))
            number=options["new_center_start"]+attempts[head]-1
            chosen=scale_pair(remaining[head],rng);samples=[]
            for scale_id in chosen:
                sample=center_and_neighbors(descriptors[head],components,axes,oyf,number,scales[scale_id],
                                             spec["sampling"]["seed"]+index*100000)
                if sample is None or len(sample["seeds"])<spec["bundles"]["minimum_valid_lines"]:break
                sample.update(head_component=head,label=descriptors[head]["label"],instance=descriptors[head]["instance"],
                              center_number=number,scale_id=scale_id)
                samples.append(sample)
            if len(samples)!=len(chosen):reject("pair_head_seed_rejection");continue
            groups.append((head,number,chosen))
            for sample in samples:batches[sample["scale_id"]].append(sample)
        accepted={}
        for scale_id,samples in batches.items():
            for start in range(0,len(samples),spec["sampling"]["trace_batch_primitives"]):
                valid,failed,stats=trace_primitive_batch(mesh,samples[start:start+spec["sampling"]["trace_batch_primitives"]],
                    scales[scale_id],spec,wall_span,(axes[0][left],axes[0][right]))
                cleaning.append({"scale_id":scale_id,**stats})
                for r in valid:accepted[(r["head_component"],r["center_number"],r["scale_id"])]=r
                for r in failed:reject(r["reason"])
        complete,failed=accept_complete_groups(groups,accepted,remaining)
        rows.extend(complete);reject("paired_trace_not_all_valid",failed)
        pipeline.write(out/"progress.json",{"accepted":len(rows),"target":len(metadata["labels"]),
            "retained":len(retained),"remaining":int(sum(v.sum() for v in remaining.values())),"attempts":sum(attempts.values()),
            "rejections":rejections})
        print(json.dumps({"flow":name,"accepted":len(rows),"target":len(metadata["labels"])}),flush=True)
    audit=audit_rows(rows,metadata,retained,geometry,seeds,options["maximum_views_per_new_center"])
    rng.shuffle(rows);split_report=physical.save_split(rows,out/"train",spec)
    split_report["native_source_x_nodes_including_curl_stencil"]=support
    pipeline.write(out/"preparation.json",{"version":spec["version"],"identity":identity(config),"flow":flow,
        "source":source,"catalog":previous["catalog"],"splits":{"train":split_report},"spatial_isolation_verified":True,
        "isolation_basis":"same frozen head identities and exact original native train support",
        "parent_preparation_sha256":pipeline.sha(parent/"preparation.json"),"audit":audit,"rejections":rejections,
        "line_cleaning":cleaning,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"test_loaded":False})


def preflight(spec,config):
    """Exercise real generation on small subsets of the frozen training heads."""
    root=Path(spec["output"])
    fixture=copy.deepcopy(spec)
    fixture["version"]="Verify_Task4C_CenterDiversityCode_4.9"
    fixture["output"]=str(root/"preflight_run")
    fixture["parent_output"]=str(root/"preflight_parent")
    fixture["sampling"]["minimum_class_count"]=1
    reports=[]
    for index,flow in enumerate(spec["flows"]):
        source=Path(spec["parent_output"])/flow["name"]
        original=json.loads((source/"preparation.json").read_text())
        for filename,digest in original["splits"]["train"]["files"].items():
            if pipeline.sha(source/"train"/filename)!=digest:
                raise ValueError("Preflight original source changed")
        with np.load(source/"train/metadata.npz") as m:metadata={k:m[k] for k in m.files}
        heads=[]
        for label in (0,1):
            available=np.unique(metadata["head_component"][metadata["labels"]==label])
            if len(available)<2:raise ValueError("Preflight needs two heads per class")
            heads.extend(available[:2].tolist())
        ids=np.flatnonzero(np.isin(metadata["head_component"],heads))
        destination=Path(fixture["parent_output"])/flow["name"]
        (destination/"train").mkdir(parents=True,exist_ok=False)
        for filename in ("geometry.npy","seeds.npy"):
            values=np.load(source/"train"/filename,mmap_mode="r")
            np.save(destination/"train"/filename,np.array(values[ids]))
        np.savez_compressed(destination/"train/metadata.npz",**{k:v[ids] for k,v in metadata.items()})
        subset=copy.deepcopy(original)
        subset["fixture_subset"]={"original_preparation_sha256":pipeline.sha(source/"preparation.json"),
            "original_train_row_indices":ids.tolist(),"note":"exact original geometry rows, no test; not a new scientific dataset"}
        subset["splits"]={"train":{"files":{name:pipeline.sha(destination/"train"/name) for name in
            ("geometry.npy","seeds.npy","metadata.npz")},"samples":len(ids),
            "native_source_x_nodes_including_curl_stencil":original["splits"]["train"]["native_source_x_nodes_including_curl_stencil"]}}
        pipeline.write(destination/"preparation.json",subset)
        prepare(fixture,config,index)
        result=json.loads((Path(fixture["output"])/"physical"/flow["name"]/"preparation.json").read_text())
        old_centers=len(np.unique(np.c_[metadata["head_component"][ids],metadata["center_number"][ids]],axis=0))
        if result["audit"]["centers"]<=old_centers:
            raise ValueError("Preflight did not exercise additional physical centers")
        reports.append({"flow":flow["name"],"original_centers":old_centers,"audit":result["audit"]})
    pipeline.write(root/"preflight.json",{"identity":identity(config),"version":fixture["version"],"passed":True,
        "rows":reports,"test_loaded":False,"scope":"real training-only generation and data integrity, not classification performance"})


def arm_spec(spec,arm):
    result=copy.deepcopy(spec);result["output"]=str(Path(spec["output"])/"arms"/arm);return result


def encode(spec,config):
    original=arm_spec(spec,"original")
    sharpness.consistency.encode(original,config)
    old_report=json.loads((Path(original["output"])/"encoding.json").read_text())
    more=arm_spec(spec,"more_centers");more["parent_output"]=str(Path(spec["output"])/"physical")
    more["parent_config_sha256"]=pipeline.sha(config)
    report={"version":spec["version"],"identity":identity(config),"splits":{},"test_encoded":False}
    for flow in spec["flows"]:
        key=flow["name"]+"/validation";source=Path(original["output"])/key;dest=Path(more["output"])/key
        dest.mkdir(parents=True,exist_ok=False)
        for filename in ("fmt.npy","voxels.npy","metadata.npz"):
            shutil.copyfile(source/filename,dest/filename)
            if pipeline.sha(dest/filename)!=old_report["splits"][key]["files"][filename]:raise ValueError("Validation copy changed")
        report["splits"][key]=copy.deepcopy(old_report["splits"][key])
    encoder.encode_partition(more,config,"train",report)
    pipeline.write(Path(more["output"])/"encoding.json",report)


def train(spec,config,index,phase):
    if phase not in ("search","confirm"):raise ValueError("No test phase in this ablation")
    previous_development,previous_confirmation=pipeline.development_plan,pipeline.confirmation_plan
    item=(previous_development if phase=="search" else previous_confirmation)(spec)[index]
    derived=arm_spec(spec,item[1]["data_arm"])
    try:
        pipeline.development_plan=lambda ignored:[item]
        pipeline.confirmation_plan=lambda ignored:[item]
        sharpness.train(derived,config,0,phase)
    finally:
        pipeline.development_plan,pipeline.confirmation_plan=previous_development,previous_confirmation


def read_run(spec,phase,method,candidate,seed):
    return FROZEN_READ_RUN(arm_spec(spec,candidate["data_arm"]),phase,method,candidate,seed)


def rank(spec,config):
    for item in pipeline.development_plan(spec):read_run(spec,"search",*item)
    pipeline.write(Path(spec["output"])/"development_rank.json",{"identity":identity(config),
        "top_two":{method:spec["candidates"] for method in spec["methods"]},"test_loaded":False,
        "rule":"confirm both data arms without search elimination"})


def select(spec,config):
    sharpness.select(spec,config)


def runtime(spec,config,phase,state,code=None):
    import torch
    row={"time":datetime.now(timezone.utc).isoformat(),"phase":phase,"state":state,"exit_code":code,
         "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",**identity(config)}
    pipeline.append_locked(Path(spec["output"])/"runtime_events.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.9 runtime: "+json.dumps(row)+"\n")


def submit_job(spec,config,phase,array=None,previous=None):
    gpu=phase in ("encode","search","confirm");out=Path(spec["output"]).resolve()
    limit="02:00:00" if phase=="prepare" else "00:30:00" if phase=="preflight" else "01:00:00" if gpu else "00:10:00"
    command=["sbatch","--parsable","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=32G",f"--time={limit}",
        f"--job-name=t4c49-{phase}",f"--output={out}/logs/{phase}_%A_%a.out",f"--error={out}/logs/{phase}_%A_%a.err"]
    if previous:command += [f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
    if gpu:command += ["--gres=gpu:1","--constraint=a100|v100|p100|rtx2080ti","--exclude=gpu203-02-r"]
    if array is not None:command += [f"--array=0-{array-1}%4"]
    command += ["ibex_bash/task4c_center_diversity_4p9.sh",phase,config]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row={"job_id":job,"phase":phase,"version":spec["version"],"submitted_at_utc":datetime.now(timezone.utc).isoformat(),
         "command":command,"dependency":previous,"expected_device":"A100/V100/P100/RTX2080Ti" if gpu else "CPU",**identity(config)}
    pipeline.append_locked(out/"submissions.jsonl",json.dumps(row)+"\n")
    pipeline.append_locked("docs/ibex_run_registry.md","\n- Task4-c 4.9 SUBMITTED "+json.dumps(row)+"\n")
    print(json.dumps(row),flush=True);return job


def submit(spec,config):
    out=Path(spec["output"]);out.mkdir(parents=True,exist_ok=False);(out/"logs").mkdir()
    pipeline.write(out/"config.frozen.json",spec);previous=None
    for phase,count in (("preflight",None),("prepare",2),("encode",None),("search",4),("rank",None),("confirm",8),("select",None)):
        previous=submit_job(spec,config,phase,count,previous)


def configure_runtime():
    sharpness.identity=identity
    sharpness.configure_runtime()
    pipeline.read_run=read_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("preflight","prepare","encode","search","rank","confirm","select","submit","runtime"))
    parser.add_argument("--config",default=DEFAULT_CONFIG);parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--input-root");parser.add_argument("--runtime-phase");parser.add_argument("--state");parser.add_argument("--exit-code",type=int)
    a=parser.parse_args();spec=load_spec(a.config);configure_runtime()
    if a.phase=="prepare":prepare(spec,a.config,a.index,a.input_root)
    elif a.phase in ("search","confirm"):train(spec,a.config,a.index,a.phase)
    elif a.phase=="runtime":runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{"preflight":preflight,"encode":encode,"rank":rank,"select":select,"submit":submit}[a.phase](spec,a.config)


if __name__=="__main__":main()
