"""Submit and immediately register the training-only Task6 audit array."""
import datetime
import json
import os
from pathlib import Path
import subprocess

from FMT_Utils.FlowMapData_3D import sha256


if __name__=="__main__":
    config="config/Verify_Task6_ReconstructionAudit_2.2.json"
    spec=json.loads(Path(config).read_text())
    root=Path(spec["output_root"]).resolve(); root.mkdir(parents=True,exist_ok=True)
    ledger=root/"submission.json"
    if ledger.exists(): raise FileExistsError(ledger)
    (root/"logs").mkdir(exist_ok=True)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    command=["sbatch","--parsable","--job-name=t6-reconstruction-audit","--nodes=1","--ntasks=1",
        "--cpus-per-task=12","--mem=24G","--time=01:00:00",f"--array=0-{len(spec['datasets'])-1}%3",
        f"--output={root}/logs/%A_%a.out",f"--error={root}/logs/%A_%a.err",
        "ibex_bash/task6_reconstruction_audit_2p2.sh",config]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row=dict(job_id=job,experiment=spec["experiment"],task=6,submitted=datetime.datetime.now().astimezone().isoformat(),
        config=config,config_sha256=sha256(config),git_commit=commit,device="CPU",array="0-8%3",command=command)
    with ledger.open("w") as f: json.dump(row,f,indent=2); f.flush(); os.fsync(f.fileno())
    with Path("docs/ibex_run_registry.md").open("a",encoding="utf-8") as f:
        f.write(f"\n- {row['submitted']} | {job} | {spec['experiment']} Task6 audit | {config} | {commit} | CPU array0-8%3 | SUBMITTED\n")
        f.flush(); os.fsync(f.fileno())
    print(json.dumps(row),flush=True)
