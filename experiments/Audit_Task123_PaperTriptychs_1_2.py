"""Check exported figure predictions and report differences from frozen runs.

This audit retains all prediction data on Ibex. It prints summary values only.
Replay differences are reported, not used to select a seed or a figure.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path("outputs/Other_Task123_PaperTriptychs_1.2")
tables = {
    "task1": "outputs/mainExp_Task1_3D_4.1/final_runs.csv",
    "task2": "outputs/mainExp_Task2_3D_6.2_uniform_confirmation/confirmation/per_run.csv",
    "task3": "outputs/mainExp_Task3_3D_9.2_uniform_confirmation/confirmation/per_run.csv",
}
records = []
for task, source in tables.items():
    rows = list(csv.DictReader(open(source)))
    for dataset in ("cylinder3d", "halfcylinderRe640", "halfcylinderRe6400", "tangaroa", "boeing747", "deltaWing_LBM"):
        path = root/f"{dataset}_{task}_predictions.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["prediction_sha256"]
        with np.load(path) as data:
            y = data["reference"].astype(bool)
            assert data["seeds"].shape == (len(y), 3)
            for arm in ("raw", "fmt"):
                pred = data[f"{arm}_prediction"].astype(bool)
                assert pred.shape == y.shape
                tp, fp, fn = ((y&pred).sum(), (~y&pred).sum(), (y&~pred).sum())
                f1 = float(2*tp/max(1,2*tp+fp+fn))
                assert abs(f1-meta["seedwise_metrics"][arm]["f1"]) < 1e-12
                def matches(r):
                    if r["dataset"] != dataset:
                        return False
                    if task == "task1":
                        return r["scope"] == "all_confirmation" and r["method"] == arm and int(r["kmeans_seed"]) == 7080
                    if task == "task2":
                        return r["arm"] == arm and int(r["training_seed"]) == 100
                    return r["source"] == ("raw_pca" if arm == "raw" else "fmt") and int(r["seed"]) == 40
                match = [r for r in rows if matches(r)]
                assert len(match) == 1
                key = {"task1":"f1", "task2":"confirmation_f1", "task3":"test_f1"}[task]
                frozen = float(match[0][key])
                replay = meta["replay_all_confirmation_metrics"][arm]["f1"]
                records.append({"task":task,"dataset":dataset,"arm":arm,"slice_f1":f1,
                                "frozen_all_confirmation_f1":frozen,"replay_all_confirmation_f1":replay,
                                "replay_minus_frozen":replay-frozen})
        for medium in ("paper", "slides"):
            stem = root/medium/f"{dataset}_{task}_triptych"
            alignment = json.loads(Path(str(stem)+".alignment.json").read_text())
            assert alignment["verdict"] == "PASS"
            audit = json.loads(stem.with_suffix(".json").read_text())
            assert audit["surface_audit"]["reference_label_mismatch"] == 0
            assert audit['mode'] == ('a' if task=='task1' else 'b')
            scene_path = root/'display_assets'/f"{dataset}_{'task1' if task=='task1' else 'task23'}.npz"
            assert hashlib.sha256(scene_path.read_bytes()).hexdigest() == audit['scene_sha256']
            with np.load(scene_path) as scene:
                assert np.array_equal(scene['reference'],y)
                if task=='task1':
                    assert len(scene['paths']) == (480 if dataset in ('boeing747','deltaWing_LBM') else 240)
                    assert len(scene['geometry']) > 0 if dataset in ('boeing747','deltaWing_LBM') else len(scene['geometry']) == 0
            for ext in (".png", ".pdf", ".svg"):
                assert stem.with_suffix(ext).stat().st_size > 0
            if medium == "paper":
                assert stem.with_suffix(".tiff").stat().st_size > 0
assert not list((root/"temporary_training").rglob("*.pt"))
report = {"status":"PASS", "prediction_metric_rows":len(records), "figures":36,
          "seedwise_metrics_recomputed":True,"reference_labels_match":True,
          "alignment_all_pass":True,"temporary_checkpoint_count":0,
          "max_absolute_replay_f1_difference":max(abs(r["replay_minus_frozen"]) for r in records),
          "replay_differences_are_not_selection_criteria":True,"records":records}
(root/"independent_figure_data_audit.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
