"""Check and summarize all recovery runs, including the unchanged historical test."""
import csv
import json
from pathlib import Path

import numpy as np
from FMT_Utils.FlowMapData_3D import sha256,write_json


def summarize(root):
    config=root/"config.frozen.json"
    spec=json.loads(config.read_text())
    audit=json.loads((root/"final_audit.json").read_text())
    assert audit["passed"] and audit["rows"]==2592 and audit["runs"]==54
    assert audit["provenance"]["config_sha256"]==sha256(config)
    with (root/"metrics.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    index={(r["dataset"],r["arm"],int(r["seed"]),r["role"],int(r["scale_id"])):r for r in rows}
    assert len(index)==2592
    summaries=[]; run_rows=[]
    for dataset in spec["datasets"]:
        data_audit=json.loads((root/"data"/dataset/"audit.json").read_text())
        assert data_audit["passed"] and data_audit["counts"]==spec["sampling"]["retained_samples"]
        summary={"dataset":dataset}
        for arm in spec["arms"]:
            for seed in spec["seeds"]:
                result=json.loads((root/"runs"/dataset/arm/str(seed)/"result.json").read_text())
                assert result["provenance"]["config_sha256"]==sha256(config)
                assert result["provenance"]["git_commit"]==audit["provenance"]["git_commit"]
                assert result["training"]["updates"]==15008
                assert result["training"]["train_examples_exposed"]==7680000
                assert 0<result["training"]["selected_step"]<=15008
                assert result["training"]["selected_validation_rmse_r"]<.8
                for metric in result["metrics"]:
                    flat=index[(dataset,arm,seed,metric["role"],metric["scale_id"])]
                    for k,v in metric.items():
                        if k not in ("role","scale_id") and not isinstance(v,list):
                            np.testing.assert_allclose(float(flat[k]),v,rtol=1e-12,atol=1e-12)
                run_rows.append(dict(dataset=dataset,arm=arm,seed=seed,device=result["device"],
                    actual_job_id=result["provenance"]["job_id"],
                    validation_rmse_r=result["training"]["selected_validation_rmse_r"],
                    selected_step=result["training"]["selected_step"],
                    initialization_validation_rmse_r=result["training"]["curve"][0]["validation_rmse_r"],
                    fit_rmse_r=result["fit_check"]["position_rmse_r"]))
            for role in ("test","unseen_scale","legacy_test"):
                values=np.array([float(index[(dataset,arm,s,role,-1)]["position_rmse_r"]) for s in spec["seeds"]])
                summary[f"{arm}_{role}_mean"]=float(values.mean())
                summary[f"{arm}_{role}_std"]=float(values.std(ddof=1))
                summary[f"{arm}_{role}_max_seed"]=float(values.max())
        summaries.append(summary)
    macro={k:float(np.mean([r[k] for r in summaries])) for k in summaries[0] if k.endswith("_mean")}
    goal={role:all(float(r["position_rmse_r"])<1 for r in rows if r["arm"]=="signed_fmt_vae" and r["role"]==role and int(r["scale_id"])==-1) for role in ("test","unseen_scale","legacy_test")}
    report=dict(experiment=spec["experiment"],audit_passed=True,config_sha256=sha256(config),
        git_commit=audit["provenance"]["git_commit"],metrics_sha256=sha256(root/"metrics.csv"),
        per_flow=summaries,macro=macro,runs=run_rows,all_signed_fmt_runs_below_one=goal,
        metric="Unchanged corresponding 7 x 31 noninitial Euclidean position RMSE / initial radius",
        aggregation="Within-flow three-seed mean, then equal-weight nine flows",
        std="Sample standard deviation across three optimizer seeds, not bootstrap confidence intervals",
        input_contract="signed_fmt10:399 signed real/imaginary coefficients; latent dimension192; decoder only receives latent",
        comparison_scope="New dataset and training; legacy_test fixes the old evaluation trajectories but is an already used benchmark")
    write_json(root/"summary.json",report)
    with (root/"per_flow_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(dict(goals=goal,macro=macro,flows=len(summaries),runs=len(run_rows)),indent=2))


if __name__=="__main__": summarize(Path("outputs/mainExp_Task6_Reconstruction_3.1"))
