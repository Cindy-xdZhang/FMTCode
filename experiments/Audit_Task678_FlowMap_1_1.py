"""Independent prediction replay and complete-matrix reporting for Task678."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from FMT_Utils.FlowMapData_3D import trajectory_metrics, sha256, write_json
from experiments.Build_Task678_FlowMap_1_1 import DEFAULT_CONFIG, provenance
from experiments.Run_Task678_FlowMap_1_1 import load_role, stack_data, truth_radius


def audit_cache(spec, config):
    root = Path(spec["output_root"])
    checks = []
    for dataset in spec["datasets"]:
        build = json.loads((root / "build" / f"{dataset}.json").read_text())
        assert build["status"] == "PASS" and build["config_sha256"] == sha256(config)
        frames, counts = set(), []
        for role in ("train", "validation", "test"):
            records = load_role(spec, dataset, role)
            for r in records:
                m = r["metadata"]
                ids = set(range(m["start_index"], m["end_index"] + 1))
                assert not frames.intersection(ids), "Source interpolation frames overlap"
                frames.update(ids)
                data = r["data"]
                n = len(data["origin0"])
                assert all(np.isfinite(a).all() for a in data.values())
                assert data["support0"].shape == (n, 31, 32, 3)
                assert data["context"].shape == (n, 6, 31, 32, 3)
                assert data["target_long"].shape == (n, 8, 125, 3)
                if dataset in spec["cylinder_datasets"]:
                    assert m["time_start"] >= 7.5 - 1e-7
                assert m["task7_minimum_seed_gap"] > m["minimum_required_gap"]
                with np.load(r["feature_file"], allow_pickle=False) as f:
                    for arm in spec["arms"]:
                        for key in ("support0", "support1", "context"):
                            a = f[f"{arm}__{key}"]
                            assert len(a) == n and np.isfinite(a).all()
                counts.append({"role": role, "ordinal": m["ordinal"], "bundles": n, "cache_sha256": m["cache_sha256"]})
        checks.append({"dataset": dataset, "windows": counts})
    write_json(root / "cache_audit.json", {**provenance(config), "status": "PASS", "checks": checks,
                                          "model_metrics_used": False})
    print("CACHE AUDIT PASS", flush=True)


def audit_results(spec, config):
    root = Path(spec["output_root"])
    rows, audits = [], []
    for dataset in spec["datasets"]:
        test = load_role(spec, dataset, "test")
        data = stack_data(test)
        for seed in spec["seeds"]:
            for width in spec["token_dimensions"]:
                base = root / "shards" / dataset / f"seed{seed}_width{width}"
                completed = list(base.glob("attempt_*/completed.json"))
                assert len(completed) == 1, f"Expected one successful attempt: {base}"
                directory = completed[0].parent
                state = json.loads(completed[0].read_text())
                assert state["status"] == "PASS" and state["config_sha256"] == sha256(config)
                assert not any(directory.rglob("*.pt")) and not any(directory.rglob("*.pth")) and not any(directory.rglob("*.ckpt"))
                for task in ("Task6", "Task7", "Task8"):
                    capacity = set()
                    truth, radii = truth_radius(data, task)
                    for arm in spec["arms"] + ["affine_interp7"]:
                        path = directory / f"{task}_{arm}.json"
                        value = json.loads(path.read_text())
                        assert value["config_sha256"] == sha256(config)
                        pred_path = directory / value["prediction_file"]
                        assert value["prediction_sha256"] == sha256(pred_path)
                        with np.load(pred_path, allow_pickle=False) as a:
                            measured = trajectory_metrics(a["prediction"], truth, radii)
                        for k, v in measured.items():
                            if isinstance(v, (float, int, list)):
                                np.testing.assert_allclose(v, value["metrics"][k], rtol=1e-10, atol=1e-10, err_msg=str(path) + "/" + k)
                        # Independently reproduce each source-window metric, not
                        # only the combined count-weighted result.
                        with np.load(pred_path, allow_pickle=False) as a:
                            predictions = a["prediction"]
                        offset, n = 0, len(data["origin0"])
                        for record, saved_window in zip(test, value["per_window"]):
                            count = len(record["data"]["origin0"])
                            ix = np.arange(offset, offset + count)
                            if task == "Task6":
                                ix = np.concatenate([ix, n + ix])
                            window = trajectory_metrics(predictions[ix], truth[ix], radii[ix])
                            for key, val in window.items():
                                if isinstance(val, (float, int, list)):
                                    np.testing.assert_allclose(val, saved_window["metrics"][key], rtol=1e-10, atol=1e-10)
                            rows.append(dict(dataset=dataset, family=spec["families"].get(dataset, dataset), task=task,
                                seed=seed, token_width=width, arm=arm, window=record["metadata"]["ordinal"],
                                **{k: v for k, v in window.items() if isinstance(v, (float, int))},
                                token_payload_bytes=value["token_payload_bytes_per_query_region"],
                                common_metadata_bytes=value["common_metadata_bytes_per_query_region"],
                                decoder_parameters=value["parameter_count"], result_file=str(path),
                                source_cache_sha256=record["metadata"]["cache_sha256"]))
                            offset += count
                        if arm != "affine_interp7":
                            capacity.add(value["parameter_count"])
                            if task == "Task8":
                                assert value["second_query_uses"] == "predicted_first_stage_endpoint"
                        audits.append(str(path))
                    assert len(capacity) == 1, f"Unmatched query network capacity: {dataset}/{task}/{width}"
    expected = len(spec["datasets"]) * len(spec["seeds"]) * len(spec["token_dimensions"]) * 3 * (len(spec["arms"]) + 1)
    assert len(audits) == expected
    with (root / "per_window_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scores = ["position_nrmse", "pair_distance_error", "centered_shape_error", "tetra_volume_error", "endpoint_error", "late_half_error", "between_sample_error"]
    summaries = []
    for task in ("Task6", "Task7", "Task8"):
        for width in spec["token_dimensions"]:
            for arm in spec["arms"] + ["affine_interp7"]:
                per_dataset = {}
                for dataset in spec["datasets"]:
                    selected = [r for r in rows if (r["task"], r["token_width"], r["arm"], r["dataset"]) == (task, width, arm, dataset)]
                    per_dataset[dataset] = {k: float(np.mean([r[k] for r in selected])) for k in scores}
                families = sorted(set(spec["families"].get(d, d) for d in spec["datasets"]))
                per_family = {fam: {k: float(np.mean([v[k] for d, v in per_dataset.items() if spec["families"].get(d, d) == fam])) for k in scores} for fam in families}
                summaries.append(dict(task=task, token_width=width, arm=arm, per_dataset=per_dataset,
                    dataset_macro={k: float(np.mean([v[k] for v in per_dataset.values()])) for k in scores},
                    family_macro={k: float(np.mean([v[k] for v in per_family.values()])) for k in scores}))
    write_json(root / "summary.json", {**provenance(config), "status": "PASS", "summary": summaries,
                                      "scope": spec["scope"], "matrix_metric_records": len(audits)})
    write_json(root / "independent_audit.json", {**provenance(config), "status": "PASS", "matrix_metric_records": len(audits),
        "per_window_rows": len(rows), "checkpoint_files": 0, "metrics_csv_sha256": sha256(root / "per_window_metrics.csv"),
        "summary_sha256": sha256(root / "summary.json")})
    print(f"RESULT AUDIT PASS: {len(audits)} records / {len(rows)} window rows", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache", action="store_true")
    args = p.parse_args()
    spec = json.loads(Path(args.config).read_text())
    spec["_config_sha256"] = sha256(args.config)
    (audit_cache if args.cache else audit_results)(spec, args.config)


if __name__ == "__main__":
    main()
