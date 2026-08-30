"""Contracts for Task3 global gradient-norm clipping search 33.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import (
    _clip_trainable_gradients,
    _gradient_clip_norm,
)


CONFIG = "config/Verify_Task3_GradientClipping_33.1.yaml"


class GradientClippingTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 8)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_clipping_grid_is_complete_and_one_factor(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        observed = [
            row["training"]["gradient_clip_norm"] for row in candidates
        ]
        self.assertEqual(observed, [None, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        for row in candidates:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"gradient_clip_norm"})
            self.assertEqual(row["sources"], ["feature"])

    def test_control_and_zero_tolerance_absolute_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "g00_control_none",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_clip_norm_validation(self):
        self.assertIsNone(_gradient_clip_norm({}))
        self.assertIsNone(_gradient_clip_norm({"gradient_clip_norm": None}))
        self.assertEqual(_gradient_clip_norm({"gradient_clip_norm": 0.5}), 0.5)
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _gradient_clip_norm({"gradient_clip_norm": value})

    def test_disabled_clipping_is_an_exact_no_op(self):
        model = torch.nn.Linear(3, 1, bias=False)
        model.weight.grad = torch.tensor([[3.0, 4.0, 0.0]])
        before = model.weight.grad.clone()
        self.assertIsNone(_clip_trainable_gradients(model, None))
        torch.testing.assert_close(model.weight.grad, before, rtol=0.0, atol=0.0)

    def test_global_norm_is_clipped_and_preclip_norm_is_reported(self):
        model = torch.nn.Linear(3, 1, bias=False)
        model.weight.grad = torch.tensor([[3.0, 4.0, 0.0]])
        observed = _clip_trainable_gradients(model, 1.0)
        self.assertAlmostEqual(observed, 5.0, places=6)
        self.assertLessEqual(float(model.weight.grad.norm()), 1.0 + 1e-6)

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 8)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(set(recipe["training"]), {"gradient_clip_norm"})


if __name__ == "__main__":
    unittest.main()
