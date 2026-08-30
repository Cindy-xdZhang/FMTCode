"""Contracts for the paired Task3 training-horizon search 47.1."""

import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_TrainingHorizon_47.1.yaml"


class TrainingHorizonTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_control_is_exact_and_epoch_grid_is_complete(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        self.assertEqual(set(candidates[0]), {"id", "sources"})
        self.assertEqual(
            [row["training"]["max_epochs"] for row in candidates[1:]],
            [20, 40, 60, 80, 125, 150, 200, 300],
        )
        for row in candidates[1:]:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"max_epochs"})
            self.assertGreater(row["training"]["max_epochs"], 0)

    def test_zero_tolerance_absolute_fmt_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "h00_control_epochs100",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 9)
            self.assertNotIn("training", recipes[0])
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)


if __name__ == "__main__":
    unittest.main()
