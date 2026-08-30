"""Contracts for Task3 binary label-smoothing search 41.1."""

import unittest

import torch
from torch.nn import functional as F

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import (
    _WeightedFocalBCEWithLogitsLoss,
    _build_training_loss,
)


CONFIG = "config/Verify_Task3_LabelSmoothing_41.1.yaml"


class LabelSmoothingTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_smoothing_grid_is_complete_and_one_factor(self):
        spec = _load_optimization_spec(CONFIG)
        observed = [
            row["training"]["label_smoothing"]
            for row in spec["optimization_candidates"]
        ]
        self.assertEqual(
            observed, [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        )
        for row in spec["optimization_candidates"]:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"label_smoothing"})
            self.assertEqual(row["sources"], ["feature"])

    def test_control_and_zero_tolerance_absolute_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "s00_control_zero",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_zero_smoothing_is_exact_weighted_bce(self):
        logits = torch.tensor([-2.0, -0.25, 0.5, 3.0])
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        criterion = _WeightedFocalBCEWithLogitsLoss(
            pos_weight=2.5, gamma=0.0, label_smoothing=0.0
        )
        expected = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor(2.5)
        )
        torch.testing.assert_close(criterion(logits, targets), expected)

    def test_positive_smoothing_matches_declared_soft_targets(self):
        logits = torch.tensor([-1.0, 2.0])
        targets = torch.tensor([0.0, 1.0])
        criterion = _WeightedFocalBCEWithLogitsLoss(
            pos_weight=1.0, gamma=0.0, label_smoothing=0.2
        )
        expected_targets = torch.tensor([0.1, 0.9])
        expected = F.binary_cross_entropy_with_logits(logits, expected_targets)
        torch.testing.assert_close(criterion(logits, targets), expected)

    def test_smoothing_validation_and_metadata(self):
        _, metadata = _build_training_loss(
            {"loss": "weighted_bce", "label_smoothing": 0.02},
            positive=20.0,
            negative=80.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(metadata["label_smoothing"], 0.02)
        for value in (-0.1, 1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _build_training_loss(
                        {"label_smoothing": value}, 20.0, 80.0,
                        torch.device("cpu"),
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
                self.assertEqual(set(recipe["training"]), {"label_smoothing"})


if __name__ == "__main__":
    unittest.main()
