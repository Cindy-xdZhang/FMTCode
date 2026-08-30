"""Contracts for the paired Task3 linear warmup search 35.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import _build_scheduler, _warmup_parameters


CONFIG = "config/Verify_Task3_LinearWarmup_35.1.yaml"


class LinearWarmupTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 7)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 69), ("smokeBuoyancy", 6))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_warmup_length(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for row in spec["optimization_candidates"]:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(
                set(row["training"]), {"warmup_epochs", "warmup_start_ratio"}
            )
            self.assertEqual(row["sources"], ["feature"])
            observed.append(_warmup_parameters({
                **row["training"], "max_epochs": 100,
            })[:2])
        self.assertEqual(
            observed,
            [(0, 0.1), (1, 0.1), (2, 0.1), (5, 0.1),
             (10, 0.1), (20, 0.1), (40, 0.1)],
        )

    def test_null_control_creates_no_scheduler(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.001)
        scheduler, name, epochs, ratio = _build_scheduler({
            "scheduler": "none", "max_epochs": 100,
            "learning_rate": 0.001,
            "warmup_epochs": None, "warmup_start_ratio": None,
        }, optimizer)
        self.assertIsNone(scheduler)
        self.assertEqual(name, "none")
        self.assertEqual((epochs, ratio), (0, 0.1))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_linear_schedule_reaches_base_learning_rate(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.001)
        scheduler, name, epochs, ratio = _build_scheduler({
            "scheduler": "none", "max_epochs": 100,
            "learning_rate": 0.001,
            "warmup_epochs": 2, "warmup_start_ratio": 0.1,
        }, optimizer)
        self.assertEqual(name, "linear_warmup_constant")
        self.assertEqual((epochs, ratio), (2, 0.1))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0001)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00055)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_invalid_warmup_fails_before_training(self):
        invalid = [
            {"warmup_epochs": -1, "max_epochs": 100},
            {"warmup_epochs": 100, "max_epochs": 100},
            {"warmup_epochs": 1, "warmup_start_ratio": 0.0,
             "max_epochs": 100},
            {"warmup_epochs": 1, "warmup_start_ratio": 1.1,
             "max_epochs": 100},
            {"warmup_epochs": 1, "scheduler": "cosine", "max_epochs": 100},
        ]
        for row in invalid:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    _warmup_parameters(row)

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "w00_control_none",
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
                self.assertEqual(
                    set(recipe["training"]),
                    {"warmup_epochs", "warmup_start_ratio"},
                )


if __name__ == "__main__":
    unittest.main()
