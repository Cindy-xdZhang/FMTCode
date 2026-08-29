from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
)
from Verify_Task3_FMTResidual import (
    _build_training_loss,
    _soft_tversky_loss,
)


CONFIG = Path("config/Verify_Task3_OverlapLoss_15.1.yaml")


class Task3OverlapLossTests(unittest.TestCase):
    def test_zero_weight_is_exact_noop_with_zero_finite_gradient(self):
        logits = torch.tensor([0.3, -0.7], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        loss = _soft_tversky_loss(logits, targets, {})
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_correct_logits_have_lower_soft_dice_loss(self):
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        recipe = {"overlap_loss_weight": 1.0}
        correct = _soft_tversky_loss(
            torch.tensor([4.0, -4.0, 4.0, -4.0]), targets, recipe
        )
        reversed_loss = _soft_tversky_loss(
            torch.tensor([-4.0, 4.0, -4.0, 4.0]), targets, recipe
        )
        self.assertLess(float(correct), float(reversed_loss))

    def test_false_negative_weight_changes_tversky_penalty_as_declared(self):
        recipe = {
            "overlap_loss_weight": 1.0,
            "overlap_false_positive_weight": 0.2,
            "overlap_false_negative_weight": 0.8,
        }
        false_negative = _soft_tversky_loss(
            torch.tensor([8.0, -8.0]), torch.tensor([1.0, 1.0]), recipe
        )
        false_positive = _soft_tversky_loss(
            torch.tensor([8.0, 8.0]), torch.tensor([1.0, 0.0]), recipe
        )
        self.assertGreater(float(false_negative), float(false_positive))

    def test_sample_weight_normalization_is_scale_invariant(self):
        logits = torch.tensor([0.1, -0.2, 0.8, -1.2])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        recipe = {"overlap_loss_weight": 0.3}
        first = _soft_tversky_loss(
            logits, targets, recipe, torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
        second = _soft_tversky_loss(
            logits, targets, recipe, torch.tensor([10.0, 20.0, 30.0, 40.0])
        )
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_invalid_overlap_configs_are_rejected(self):
        bad = [
            {"overlap_loss_weight": -0.1},
            {"overlap_false_positive_weight": -0.1},
            {"overlap_false_negative_weight": -0.1},
            {
                "overlap_false_positive_weight": 0.0,
                "overlap_false_negative_weight": 0.0,
            },
            {"overlap_smoothing": 0.0},
            {"overlap_loss_weight": np.nan},
        ]
        for recipe in bad:
            with self.subTest(recipe=recipe):
                with self.assertRaises(ValueError):
                    _build_training_loss(
                        recipe, 10.0, 90.0, torch.device("cpu")
                    )

    def test_15_1_grid_and_array_bounds_are_frozen(self):
        overlay = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(overlay["experiment"], "Verify_Task3_OverlapLoss_15.1")
        self.assertEqual(len(spec["datasets"]), 10)
        self.assertEqual(len(spec["optimization_candidates"]), 12)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(
            len(spec["datasets"])
            * len(spec["optimization_candidates"])
            * len(spec["paired_seeds"])
            * 2,
            720,
        )
        self.assertEqual(_decode_job(spec, 0), (spec["datasets"][0], 0))
        self.assertEqual(_decode_job(spec, 119), (spec["datasets"][-1], 11))
        with self.assertRaises(IndexError):
            _decode_job(spec, 120)


if __name__ == "__main__":
    unittest.main()
