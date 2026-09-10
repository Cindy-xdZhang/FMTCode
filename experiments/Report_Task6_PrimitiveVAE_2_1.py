"""Summarize audited Task6 2.1 predictions and training records without model files."""
import csv
import json
from pathlib import Path

import numpy as np

from FMT_Utils.FlowMapData_3D import sha256, write_json


def summarize(root, config):
    spec = json.loads(config.read_text())
    # Retain and hash the exact deployed file. Windows checkout CRLF conversion
    # changes byte hashes but must not be treated as a scientific config change.
    frozen_config = root / "config.frozen.json"
    assert json.loads(frozen_config.read_text()) == spec
    config_hash = sha256(frozen_config)
    audit = json.loads((root / "final_audit.json").read_text())
    assert audit["passed"] and audit["rows"] == 1566
    assert audit["provenance"]["config_sha256"] == config_hash
    commit = audit["provenance"]["git_commit"]
    with (root / "metrics.csv").open(newline="", encoding="utf-8") as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == audit["rows"]
    lookup = {(m["dataset"], m["arm"], int(m["seed"]), m["role"], int(m["scale_id"])):m for m in metrics}
    assert len(lookup) == len(metrics)
    runs = {}
    for dataset in spec["datasets"]:
        data_audit = json.loads((root / "data" / dataset / "audit.json").read_text())
        assert data_audit["passed"]
        assert data_audit["counts"] == spec["sampling"]["retained_samples"]
        for arm in spec["arms"]:
            for seed in spec["seeds"]:
                result = json.loads((root / "runs" / dataset / arm / str(seed) / "result.json").read_text())
                assert result["provenance"]["config_sha256"] == config_hash
                assert result["provenance"]["git_commit"] == commit
                assert result["training"]["updates"] == 14160
                assert result["training"]["train_examples_exposed"] == 7200000
                assert result["vae"] == spec["vae"]
                assert (result["dataset"], result["arm"], result["seed"]) == (dataset, arm, seed)
                for row in result["metrics"]:
                    flat = lookup[(dataset, arm, seed, row["role"], row["scale_id"])]
                    for key, value in row.items():
                        if key not in ("role", "scale_id") and not isinstance(value, list):
                            np.testing.assert_allclose(float(flat[key]), value, rtol=1e-12)
                runs[(dataset, arm, seed)] = result
    tables = []
    for dataset in spec["datasets"]:
        row = {"dataset":dataset}
        for arm in spec["arms"]:
            for role in ("test", "unseen_scale"):
                values = [float(lookup[(dataset,arm,seed,role,-1)]["position_rmse_r"]) for seed in spec["seeds"]]
                row[f"{arm}_{role}_mean"] = float(np.mean(values))
                row[f"{arm}_{role}_std"] = float(np.std(values, ddof=1))
            for key in ("fit", "train", "validation"):
                values = []
                for seed in spec["seeds"]:
                    r = runs[(dataset,arm,seed)]
                    value = r["fit_check"]["position_rmse_r"] if key == "fit" else r["training"]["curve"][-1]["train_probe_rmse_r" if key == "train" else "validation_rmse_r"]
                    values.append(value)
                row[f"{arm}_{key}_mean"] = float(np.mean(values))
        tables.append(row)
    macro = {key:float(np.mean([r[key] for r in tables])) for key in tables[0] if key.endswith("_mean")}
    wins = {role:sum(r[f"fmt_all_vae_{role}_mean"] < r[f"raw_vae_{role}_mean"] for r in tables) for role in ("test", "unseen_scale")}
    summary = dict(experiment=spec["experiment"], git_commit=commit, config_sha256=config_hash,
        audited_rows=len(metrics), completed_training_runs=len(runs), per_flow=tables,
        macro=macro, fmt_lower_error_flow_count=wins,
        metrics_sha256=sha256(root / "metrics.csv"),
        macro_definition="First average three seeds within each flow, then equal-weight nine flows; not pooled point RMSE",
        std_definition="Sample standard deviation across three seeds within a flow",
        fit_definition="512 train primitives, 4000 updates; same primitives evaluated; separate initialization from main training",
        training_probe_definition="Fixed 4096 training primitives; not all 60000 training primitives",
        units="Euclidean position RMSE / initial neighbor radius; noninitial 7 x 31 samples",
        device_counts={name:sum(r["device"]==name for r in runs.values()) for name in sorted(set(r["device"] for r in runs.values()))})
    write_json(root / "summary.json", summary)
    with (root / "per_flow_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0])); writer.writeheader(); writer.writerows(tables)
    lines=["| Flow | FMT-VAE test RMSE/r | Raw-VAE test RMSE/r |", "|---|---:|---:|"]
    for r in tables:
        lines.append(f"| {r['dataset']} | {r['fmt_all_vae_test_mean']:.6f} ± {r['fmt_all_vae_test_std']:.6f} | {r['raw_vae_test_mean']:.6f} ± {r['raw_vae_test_std']:.6f} |")
    (root / "summary_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"macro":macro,"wins":wins,"runs":len(runs)},indent=2))


if __name__ == "__main__":
    summarize(Path("outputs/mainExp_Task6_PrimitiveVAE_2.1"), Path("config/mainExp_Task6_PrimitiveVAE_2.1.json"))
