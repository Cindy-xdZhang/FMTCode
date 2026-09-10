"""Submit data expansion and matched signed-FMT/Raw VAE reconstruction runs."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

from FMT_Utils.FlowMapData_3D import sha256


if __name__=="__main__":
    config="config/mainExp_Task6_Reconstruction_3.1.json"
    spec=json.loads(Path(config).read_text()); root=Path(spec["output_root"]).resolve()
    root.mkdir(parents=True,exist_ok=True); ledger=root/"submissions.jsonl"
    if ledger.exists(): raise FileExistsError(ledger)
    (root/"logs").mkdir(exist_ok=True)
    shutil.copyfile(config,root/"config.frozen.json")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    previous=None
    phases=[("preflight",4,12,"00:20:00",None,False),
        ("build",16,40,"08:00:00",f"0-{len(spec['datasets'])-1}%3",False),
        ("audit-data",8,32,"01:00:00",None,False),
        ("train",4,24,"06:00:00",f"0-{len(spec['datasets'])*len(spec['arms'])*len(spec['seeds'])-1}%6",True),
        ("audit-results",8,24,"01:00:00",None,False)]
    for phase,cpu,mem,limit,array,gpu in phases:
        command=["sbatch","--parsable",f"--job-name=t6-recovery-{phase}","--nodes=1","--ntasks=1",
            f"--cpus-per-task={cpu}",f"--mem={mem}G",f"--time={limit}",
            f"--output={root}/logs/{phase}_%A_%a.out",f"--error={root}/logs/{phase}_%A_%a.err"]
        if array: command+=[f"--array={array}"]
        if previous: command+=[f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes"]
        if gpu: command += ["--gres=gpu:1","--constraint=a100|v100","--exclude=gpu203-02-r"]
        command += ["ibex_bash/task6_reconstruction_3p1.sh",phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,experiment=spec["experiment"],task=6,phase=phase,
            submitted=datetime.datetime.now().astimezone().isoformat(),config=config,config_sha256=sha256(config),
            git_commit=commit,device="GPU A100/V100" if gpu else "CPU",array=array,command=command)
        with ledger.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n"); f.flush(); os.fsync(f.fileno())
        with Path("docs/ibex_run_registry.md").open("a",encoding="utf-8") as f:
            f.write(f"\n- {row['submitted']} | {job} | {spec['experiment']} Task6 {phase} | {config} | {commit} | {row['device']} array={array} | SUBMITTED\n")
            f.flush(); os.fsync(f.fileno())
        print(json.dumps(row),flush=True); previous=job
