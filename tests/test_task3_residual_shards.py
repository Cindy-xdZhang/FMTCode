import csv
import json
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Merge_Task3_ResidualShards import MODES, run


def _write_shard(root, group, dataset, mode, seeds):
    shard = root / f"development_{group}" / f"{mode}_shards" / dataset
    (shard / "checkpoints").mkdir(parents=True)
    (shard / "histories").mkdir(parents=True)
    rows = []
    variant = MODES[mode]
    for seed in seeds:
        rows.append({
            "dataset": dataset, "variant": variant, "seed": seed,
            "validation_average_precision": 0.5 + seed / 1000.0,
        })
        (shard / "checkpoints" / f"{dataset}_{variant}_seed{seed}.pt").write_bytes(
            f"checkpoint-{group}-{mode}-{dataset}-{seed}".encode()
        )
        (shard / "histories" / f"{dataset}_{variant}_seed{seed}.csv").write_text(
            "epoch,score\n1,0.5\n", encoding="utf-8"
        )
    with (shard / "per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_residual_shards_is_complete_and_idempotent():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "results"
        seeds = [30, 31]
        groups = [
            {"name": "old8", "datasets": ["flow_a"]},
            {"name": "new2", "datasets": ["flow_b"]},
        ]
        for group in groups:
            for mode in MODES:
                for dataset in group["datasets"]:
                    _write_shard(root, group["name"], dataset, mode, seeds)
        config = Path(directory) / "evaluate.yaml"
        config.write_text(
            yaml.safe_dump({"seeds": seeds, "groups": groups}), encoding="utf-8"
        )
        manifest_path = run(config, root)
        first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(first_manifest) == 4
        assert sum(item["row_count"] for item in first_manifest) == 8
        for group in groups:
            for mode in MODES:
                target = root / f"development_{group['name']}" / mode
                assert (target / "per_run.csv").exists()
                assert len(list((target / "checkpoints").glob("*.pt"))) == 2
                assert len(list((target / "histories").glob("*.csv"))) == 2
        second_manifest_path = run(config, root)
        second_manifest = json.loads(
            second_manifest_path.read_text(encoding="utf-8")
        )
        assert second_manifest == first_manifest


if __name__ == "__main__":
    test_merge_residual_shards_is_complete_and_idempotent()
    print("TASK3 RESIDUAL SHARD TEST PASSED")
