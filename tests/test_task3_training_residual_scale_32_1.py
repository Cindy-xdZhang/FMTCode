"""Contracts for Task3 training-time residual-scale search 32.1."""

import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_TrainingResidualScale_32.1.yaml"


class TrainingResidualScaleTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_training_alpha_grid_is_complete_and_one_factor(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        observed = [row["training"]["training_alpha"] for row in candidates]
        self.assertEqual(
            observed,
            [1.0, 0.125, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0],
        )
        for row in candidates:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"training_alpha"})
            self.assertEqual(row["sources"], ["feature"])

    def test_control_and_zero_tolerance_absolute_guard(self):
        spec = _load_optimization_spec(CONFIG)
        selection = spec["optimization_selection"]
        self.assertEqual(
            selection["absolute_fmt_guard"],
            {
                "control_optimization_id": "t00_control_a1",
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
            self.assertEqual(len(recipes), 9)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(set(recipe["training"]), {"training_alpha"})


if __name__ == "__main__":
    unittest.main()
