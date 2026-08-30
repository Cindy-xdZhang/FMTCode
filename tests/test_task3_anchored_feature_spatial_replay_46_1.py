import csv
import tempfile
import unittest
from pathlib import Path

from Replay_Task3_AnchoredFeatureSpatial_46_1 import (
    _checkpoint_from_result,
    _decode_job,
    _load_spec,
    _selected_candidate,
    _validate_shard_rows,
)


CONFIG = "config/Verify_Task3_AnchoredFeatureSpatialReplay_46.1.yaml"


class Task3AnchoredFeatureSpatialReplayTests(unittest.TestCase):
    def test_config_marks_replay_as_exposed_development(self):
        spec = _load_spec(CONFIG)
        self.assertEqual(spec["status"], "exposed_development_replay")
        self.assertEqual(len(spec["datasets"]), 10)
        self.assertEqual(spec["paired_seeds"], [40, 41])
        self.assertEqual(spec["confirmation_count"], 4)
        self.assertEqual(spec["expected_ivd_percentile"], 95.0)
        self.assertEqual(spec["target_dataset_macro_f1_gain"], 0.15)
        self.assertEqual(
            spec["replay_population"]["seed_grid_phase"],
            [0.318359375, 0.4561042524005485, -0.3352],
        )

    def test_job_mapping_has_exactly_ten_datasets(self):
        spec = _load_spec(CONFIG)
        self.assertEqual(_decode_job(spec, 0), "channel")
        self.assertEqual(_decode_job(spec, 9), "smokeBuoyancy")
        with self.assertRaises(IndexError):
            _decode_job(spec, 10)

    def test_selected_candidate_is_bound_to_feature(self):
        source = {
            "groups": {"family": {"datasets": ["flow"]}},
            "candidates": [{"id": "d01", "fmt_feature": "aivd1w3"}],
        }
        selection = {"primary_by_group": {
            "family": {"candidate_id": "d01", "fmt_feature": "aivd1w3"}
        }}
        family, candidate = _selected_candidate(source, selection, "flow")
        self.assertEqual(family, "family")
        self.assertEqual(candidate["id"], "d01")
        selection["primary_by_group"]["family"]["fmt_feature"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "feature changed"):
            _selected_candidate(source, selection, "flow")

    def test_checkpoint_must_remain_inside_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "outputs" / "candidate" / "fmt" / "per_run.csv"
            checkpoint = result.parent / "checkpoints" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"frozen")
            result.parent.mkdir(parents=True, exist_ok=True)
            with result.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "dataset", "seed", "variant", "checkpoint",
                ])
                writer.writeheader()
                writer.writerow({
                    "dataset": "flow", "seed": 40,
                    "variant": "raw_fmt_residual",
                    "checkpoint": str(checkpoint.relative_to(root)),
                })
            observed, _ = _checkpoint_from_result(root, result, {
                "dataset": "flow", "seed": 40,
                "variant": "raw_fmt_residual",
            })
            self.assertEqual(observed, checkpoint)
            outside = root / "outside.pt"
            outside.write_bytes(b"outside")
            with result.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["checkpoint"] = str(outside)
            with result.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                _checkpoint_from_result(root, result, {
                    "dataset": "flow", "seed": 40,
                    "variant": "raw_fmt_residual",
                })

    def test_shard_requires_both_frozen_arms(self):
        spec = {
            "experiment": "replay", "status": "exposed_development_replay",
            "config_sha256": "config", "paired_seeds": [40],
        }
        manifest = {"models": []}
        rows = []
        for source, method in (
            ("fmt", "fmt_residual"),
            ("raw_pca", "raw_pca_residual"),
        ):
            model = {
                "dataset": "flow", "physical_family": "family",
                "seed": 40, "source": source, "candidate_id": "d01",
                "fmt_feature": "aivd1w3", "checkpoint_sha256": source,
            }
            manifest["models"].append(model)
            rows.append({
                "experiment": "replay", "status": "exposed_development_replay",
                "config_sha256": "config",
                "preflight_manifest_sha256": "manifest",
                "dataset": "flow", "physical_family": "family",
                "seed": 40, "source": source, "method": method,
                "candidate_id": "d01", "fmt_feature": "aivd1w3",
                "checkpoint_sha256": source, "f1": 0.8,
                "average_precision": 0.9, "sample_count": 100,
                "positive_fraction": 0.05,
            })
        _validate_shard_rows(
            rows, spec, manifest, "manifest", "flow", require_complete=True
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            _validate_shard_rows(
                rows[:1], spec, manifest, "manifest", "flow",
                require_complete=True,
            )


if __name__ == "__main__":
    unittest.main()
