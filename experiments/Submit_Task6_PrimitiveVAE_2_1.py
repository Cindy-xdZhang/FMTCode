"""Submit an immutable Task6 pipeline; durably register each accepted Slurm job."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

from FMT_Utils.FlowMapData_3D import sha256


def main():
    config="config/mainExp_Task6_PrimitiveVAE_2.1.json"
    spec=json.loads(Path(config).read_text())
    root=Path(spec["output_root"]).resolve()
    root.mkdir(parents=True,exist_ok=True)
    ledger=root/"submissions.jsonl"
    if ledger.exists(): raise FileExistsError("Submission ledger already exists; no duplicate jobs")
    (root/"logs").mkdir(exist_ok=True)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    rows=[]
    for phase,resources,array in [
        ("preflight",["--cpus-per-task=4","--mem=12G","--time=00:20:00"],None),
        ("build",["--cpus-per-task=16","--mem=40G","--time=08:00:00"],f"0-{len(spec['datasets'])-1}%3"),
        ("audit-data",["--cpus-per-task=8","--mem=24G","--time=00:40:00"],None),
        ("train",["--cpus-per-task=4","--mem=20G","--time=05:00:00","--gres=gpu:1","--constraint=a100|v100","--exclude=gpu203-02-r"],f"0-{len(spec['datasets'])*len(spec['arms'])*len(spec['seeds'])-1}%6"),
        ("audit-results",["--cpus-per-task=4","--mem=16G","--time=00:40:00"],None)]:
        args=["sbatch","--parsable",f"--job-name=t6vae-{phase}","--nodes=1","--ntasks=1",f"--chdir={Path.cwd()}",
            f"--output={root}/logs/{phase}_%A_%a.out",f"--error={root}/logs/{phase}_%A_%a.err",*resources]
        if rows: args += [f"--dependency=afterok:{rows[-1]['job_id']}","--kill-on-invalid-dep=yes"]
        if array: args += [f"--array={array}"]
        args += ["ibex_bash/task6_primitive_vae_2p1.sh",phase,config]
        job_id=subprocess.check_output(args,text=True).strip().split(';')[0]
        row=dict(job_id=job_id,experiment=spec["experiment"],task=6,phase=phase,
            submitted=datetime.datetime.now().astimezone().isoformat(),git_commit=commit,config=config,
            config_sha256=sha256(config),device="GPU A100/V100" if phase=="train" else "CPU",array=array,status="SUBMITTED",command=args)
        with ledger.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n"); f.flush(); os.fsync(f.fileno())
        with Path("docs/ibex_run_registry.md").open("a",encoding="utf-8") as f:
            f.write(f"\n- {row['submitted']} | {job_id} | {spec['experiment']} Task6 {phase} | {config} | {commit} | {row['device']} | array={array} | SUBMITTED; events: `{root}/logs`\n")
            f.flush(); os.fsync(f.fileno())
        rows.append(row)
        print(json.dumps(row),flush=True)


if __name__=="__main__": main()
