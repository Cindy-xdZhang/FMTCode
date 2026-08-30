"""Contracts for Task3 cosine minimum-learning-rate search 43.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import _build_scheduler


CONFIG = "config/Verify_Task3_CosineMinLR_43.1.yaml"


class CosineMinLRTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_is_complete_and_control_has_no_override(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        self.assertEqual(set(candidates[0]), {"id", "sources"})
        observed = [
            row["training"]["minimum_learning_rate_ratio"]
            for row in candidates[1:]
        ]
        self.assertEqual(observed, [0.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
        for row in candidates[1:]:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(
                set(row["training"]),
                {"scheduler", "minimum_learning_rate_ratio"},
            )
            self.assertEqual(row["training"]["scheduler"], "cosine")

    def test_control_and_zero_tolerance_absolute_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "c00_control_constant",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_control_is_none_and_cosine_eta_min_is_exact(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        base = {
            "learning_rate": 1e-3,
            "max_epochs": 100,
            "scheduler": "none",
        }
        scheduler, name, warmup, ratio = _build_scheduler(base, optimizer)
        self.assertIsNone(scheduler)
        self.assertEqual(name, "none")
        self.assertEqual(warmup, 0)
        self.assertEqual(ratio, 0.1)
        scheduler, name, _, _ = _build_scheduler(
            {**base, "scheduler": "cosine", "minimum_learning_rate_ratio": 0.25},
            optimizer,
        )
        self.assertEqual(name, "cosine")
        self.assertEqual(scheduler.eta_min, 2.5e-4)

    def test_ratio_validation(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        base = {
            "learning_rate": 1e-3,
            "max_epochs": 100,
            "scheduler": "cosine",
        }
        for value in (-0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _build_scheduler(
                        {**base, "minimum_learning_rate_ratio": value}, optimizer
                    )

    def test_completed_anchored_feature_resolves_for_every_family(self):
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
