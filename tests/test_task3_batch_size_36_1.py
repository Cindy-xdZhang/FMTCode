"""Contracts for the paired Task3 batch-size search 36.1."""

import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import _batch_size


CONFIG = "config/Verify_Task3_BatchSize_36.1.yaml"


class BatchSizeTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 7)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 69), ("smokeBuoyancy", 6))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_batch_size(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for row in spec["optimization_candidates"]:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"batch_size"})
            self.assertEqual(row["sources"], ["feature"])
            observed.append(_batch_size(row["training"]))
        self.assertEqual(observed, [32, 64, 128, 256, 512, 1024, 2048])

    def test_batch_size_rejects_silent_truncation_and_invalid_values(self):
        for value in (None, True, False, 0, -1, 2.5, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _batch_size({"batch_size": value})
        self.assertEqual(_batch_size({"batch_size": 32.0}), 32)

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "b04_control_batch512",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 7)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(set(recipe["training"]), {"batch_size"})


if __name__ == "__main__":
    unittest.main()
