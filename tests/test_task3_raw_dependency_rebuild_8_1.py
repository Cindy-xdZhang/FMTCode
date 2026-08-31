import tempfile
import unittest
from pathlib import Path

import torch
import yaml

import Repair_Task3_RawDependencies_8_1 as repair


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "Verify_Task3_RawDependencyRebuild_8.1.yaml"


class TestTask3RawDependencyRebuild81(unittest.TestCase):
    def test_registered_job_map_has_two_replicas_and_two_groups(self):
        spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            repair._job_map(spec),
            [("a", "old8"), ("a", "new2"),
             ("b", "old8"), ("b", "new2")],
        )
        datasets = {
            dataset
            for group in spec["groups"].values()
            for dataset in group["datasets"]
        }
        self.assertEqual(len(datasets), 10)
        self.assertEqual(spec["paired_seeds"], [40, 41])
        self.assertEqual(spec["variants"], ["raw"])
        self.assertFalse(spec["confirmation_metrics_opened"])
        self.assertFalse(spec["confirmation_hyperparameters_changed"])

    def test_state_hash_is_key_order_independent_and_value_sensitive(self):
        first = {
            "b": torch.tensor([2.0]),
            "a": torch.tensor([[1.0, 3.0]]),
        }
        second = {"a": first["a"].clone(), "b": first["b"].clone()}
        self.assertEqual(repair._state_hash(first), repair._state_hash(second))
        second["b"][0] += 1.0
        self.assertNotEqual(repair._state_hash(first), repair._state_hash(second))

    def test_metric_comparison_ignores_elapsed_and_checkpoint_only(self):
        base = {
            "parameter_count": "10",
            "best_epoch": "4",
            **{name: "0.25" for name in repair.FLOAT_FIELDS},
            "elapsed_seconds": "1.0",
            "checkpoint": "a.pt",
        }
        changed = dict(base, elapsed_seconds="999", checkpoint="b.pt")
        self.assertEqual(repair._metric_difference(base, changed), 0.0)
        changed["validation_f1"] = "0.2500000001"
        self.assertGreater(repair._metric_difference(base, changed), 0.0)

    def test_confirmation_result_detector_blocks_partial_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(repair._confirmation_result_paths(root), [])
            shard = root / "shards" / "channel.csv"
            shard.parent.mkdir(parents=True)
            shard.write_text("x\n", encoding="utf-8")
            self.assertEqual(repair._confirmation_result_paths(root), [shard])


if __name__ == "__main__":
    unittest.main()
