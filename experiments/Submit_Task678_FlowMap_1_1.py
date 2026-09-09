"""Submit the Git-pinned Ibex dependency chain; register every submission/event."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess

from FMT_Utils.FlowMapData_3D import sha256
from experiments.Build_Task678_FlowMap_1_1 import DEFAULT_CONFIG


def append_registry(text):
    path = Path("docs/ibex_run_registry.md")
    with path.open("a", encoding="utf-8") as f:
        if os.name != "nt":
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def event(spec, config, phase, state, exit_code):
    job = os.getenv("SLURM_ARRAY_JOB_ID", os.getenv("SLURM_JOB_ID", "local"))
    index = os.getenv("SLURM_ARRAY_TASK_ID", "")
    device = "CPU"
    if phase == "train":
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU unavailable"
    item = dict(time=datetime.datetime.now().astimezone().isoformat(), phase=phase, state=state,
                job_id=job, array_index=index, node=socket.gethostname(), device=device, exit_code=exit_code,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                config=config, config_sha256=sha256(config))
    root = Path(spec["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "job_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")
    append_registry(f"\n- Task678 1.1 `{job}{'_' + index if index else ''}` | {phase} | {state} | {item['time']} | node `{item['node']}` | {device} | exit={exit_code} | commit `{item['git_commit']}` | config `{config}` SHA256 `{item['config_sha256']}`\n")
    print(json.dumps(item), flush=True)


def commands(spec, config, commit):
    root = Path(spec["output_root"]).resolve()
    common = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=24G"]
    phases = [
        ("source", [], ["--time=00:20:00"]),
        ("smoke", [], ["--time=00:20:00"]),
        ("build", ["source", "smoke"], [f"--array=0-{len(spec['datasets'])-1}%4", "--time=03:00:00", "--cpus-per-task=8", "--mem=48G"]),
        ("cache_audit", ["build"], ["--time=00:45:00", "--mem=32G"]),
        ("train", ["cache_audit"], [f"--array=0-{len(spec['datasets'])*len(spec['seeds'])*len(spec['token_dimensions'])-1}%12",
                                   "--time=06:00:00", "--gres=gpu:1", "--constraint=a100|v100", "--exclude=gpu203-02-r"]),
        ("audit", ["train"], ["--time=01:00:00", "--mem=32G"]),
    ]
    return [(phase, deps, common + [f"--job-name=FMT678_{phase}",
            f"--output={root}/logs/{phase}.%A_%a.out", f"--error={root}/logs/{phase}.%A_%a.err"] + extra)
            for phase, deps, extra in phases]


def submit(spec, config, dry_run=False):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root = Path(spec["output_root"])
    registry = root / "submissions.jsonl"
    if registry.exists() and not dry_run:
        raise FileExistsError("Submissions already exist; inspect IDs before any recovery, do not duplicate arrays")
    if not dry_run:
        (root / "logs").mkdir(parents=True, exist_ok=True)
    jobs = {}
    for phase, deps, command in commands(spec, config, commit):
        if deps:
            command += ["--dependency=afterok:" + ":".join(jobs[d] for d in deps)]
        command += ["ibex_bash/mainexp_task678_flowmap_1p1.sh", phase, config, commit]
        if dry_run:
            jobs[phase] = "DRY_" + phase
            print(json.dumps({"phase": phase, "command": command}), flush=True)
            continue
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Unexpected sbatch response {job!r}")
        jobs[phase] = job
        item = dict(phase=phase, job_id=job, submitted=datetime.datetime.now().astimezone().isoformat(),
                    experiment=spec["experiment"], tasks=spec["task_versions"], config=config,
                    config_sha256=sha256(config), git_commit=commit, command=command,
                    expected_device="one A100 or V100 per array element" if phase == "train" else "CPU")
        with registry.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
            f.flush()
            os.fsync(f.fileno())
        append_registry(f"\n- **SUBMITTED Task678 1.1** `{job}` | {phase} | {item['submitted']} | config `{config}` SHA256 `{item['config_sha256']}` | commit `{commit}` | {item['expected_device']} | dependencies `{','.join(jobs[d] for d in deps) or 'none'}`\n")
        print(json.dumps(item), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--event", choices=["RUNNING", "COMPLETED", "FAILED"])
    p.add_argument("--phase")
    p.add_argument("--exit-code", type=int)
    args = p.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if args.event:
        event(spec, args.config, args.phase, args.event, args.exit_code)
    else:
        submit(spec, args.config, args.dry_run)


if __name__ == "__main__":
    main()
