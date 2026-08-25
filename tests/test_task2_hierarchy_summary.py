import csv
import json

from Summarize_Task2_3D_Hierarchy import summarize


def test_hierarchy_summary_requires_strict_order(tmp_path):
    task2 = tmp_path / "task2.json"
    task2.write_text(json.dumps({
        "experiment": "test",
        "paper_table": [
            {"dataset": "a", "raw_f1_mean": 0.6, "fmt_f1_mean": 0.7},
            {"dataset": "b", "raw_f1_mean": 0.8, "fmt_f1_mean": 0.9},
        ],
    }), encoding="utf-8")
    task1 = tmp_path / "task1.csv"
    with task1.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "fmt_f1_mean"])
        writer.writeheader()
        writer.writerows([
            {"dataset": "a", "fmt_f1_mean": 0.5},
            {"dataset": "b", "fmt_f1_mean": 0.6},
        ])
    output = summarize(task2, [str(task1)], tmp_path / "hierarchy.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregate"]["hierarchy_satisfied"] is True
    assert payload["aggregate"]["raw_vae_minus_task1"] > 0
    assert payload["aggregate"]["fmt_vae_minus_raw_vae"] > 0
