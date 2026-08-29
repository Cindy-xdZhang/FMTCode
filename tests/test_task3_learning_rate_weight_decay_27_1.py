"""Contracts for Task3 learning-rate x weight-decay search 27.1."""

import unittest

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_LearningRateWeightDecay_27.1.yaml"


class LearningRateWeightDecayTests(unittest.TestCase):
    def test_preregistered_factorial_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 15)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 149), ("smokeBuoyancy", 14))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.190)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.892)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_is_complete_five_by_three(self):
        spec = _load_optimization_spec(CONFIG)
        observed = {
            (
                float(row["training"]["learning_rate"]),
                float(row["training"]["weight_decay"]),
            )
            for row in spec["optimization_candidates"]
        }
        expected = {
            (learning_rate, weight_decay)
            for learning_rate in (0.0002, 0.0005, 0.001, 0.002, 0.005)
            for weight_decay in (0.0, 0.0001, 0.001)
        }
        self.assertEqual(observed, expected)

    def test_exact_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        control = next(
            row for row in spec["optimization_candidates"]
            if row["id"] == "l07_control_lr1e3_wd1e4"
        )
        self.assertEqual(control["training"], {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        })
        guard = spec["optimization_selection"]["absolute_fmt_guard"]
        self.assertEqual(
            guard["control_optimization_id"], control["id"]
        )
        self.assertEqual(guard["f1_tolerance"], 0.0)
        self.assertEqual(guard["average_precision_tolerance"], 0.0)

    def test_only_optimizer_changes_and_both_arms_share_it(self):
        spec = _load_optimization_spec(CONFIG)
        for row in spec["optimization_candidates"]:
            self.assertEqual(row["sources"], ["feature"])
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(
                set(row["training"]), {"learning_rate", "weight_decay"}
            )

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 15)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(
                    set(recipe["training"]),
                    {"learning_rate", "weight_decay"},
                )


if __name__ == "__main__":
    unittest.main()
