"""Contracts for Task3 Adam-family epsilon search 42.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import (
    _build_optimizer,
    _optimizer_epsilon,
)


CONFIG = "config/Verify_Task3_AdamEpsilon_42.1.yaml"


class AdamEpsilonTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 8)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_epsilon_grid_is_complete_and_one_factor(self):
        spec = _load_optimization_spec(CONFIG)
        observed = [
            row["training"]["optimizer_epsilon"]
            for row in spec["optimization_candidates"]
        ]
        self.assertEqual(
            observed,
            [None, 1e-12, 1e-10, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4],
        )
        for row in spec["optimization_candidates"]:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"optimizer_epsilon"})
            self.assertEqual(row["sources"], ["feature"])

    def test_control_and_zero_tolerance_absolute_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "q00_control_default",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_epsilon_validation(self):
        self.assertIsNone(_optimizer_epsilon({}))
        self.assertIsNone(_optimizer_epsilon({"optimizer_epsilon": None}))
        self.assertEqual(_optimizer_epsilon({"optimizer_epsilon": 1e-6}), 1e-6)
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _optimizer_epsilon({"optimizer_epsilon": value})

    def test_control_preserves_pytorch_default_and_override_is_exact(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        base = {
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        }
        control, _, _ = _build_optimizer(base, [parameter])
        self.assertEqual(control.param_groups[0]["eps"], 1e-8)
        override, _, _ = _build_optimizer(
            {**base, "optimizer_epsilon": 1e-5}, [parameter]
        )
        self.assertEqual(override.param_groups[0]["eps"], 1e-5)

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 8)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(set(recipe["training"]), {"optimizer_epsilon"})


if __name__ == "__main__":
    unittest.main()
