"""Contracts for the paired Task3 focal-loss gamma search 39.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import _build_training_loss


CONFIG = "config/Verify_Task3_FocalGamma_39.1.yaml"
EXPECTED = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00,
            1.50, 2.00, 2.50, 3.00, 4.00, 5.00]


class FocalGammaTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 12)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 119), ("smokeBuoyancy", 11))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_focal_loss_and_gamma(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["feature"])
            if index == 0:
                self.assertEqual(
                    set(row), {"id", "sources"},
                    "weighted-BCE control must be an exact no-override cell",
                )
                observed.append(0.0)
                continue
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"loss", "focal_gamma"})
            self.assertEqual(row["training"]["loss"], "focal")
            observed.append(float(row["training"]["focal_gamma"]))
        self.assertEqual(observed, EXPECTED)

    def test_loss_metadata_and_forward_are_finite(self):
        logits = torch.tensor([-80.0, -2.0, 0.0, 2.0, 80.0])
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        spec = _load_optimization_spec(CONFIG)
        for index, row in enumerate(spec["optimization_candidates"]):
            training = dict(row.get("training", {}))
            criterion, metadata = _build_training_loss(
                training, positive=2.0, negative=6.0,
                device=torch.device("cpu"),
            )
            expected_loss = "weighted_bce" if index == 0 else "focal"
            self.assertEqual(metadata["loss"], expected_loss)
            self.assertEqual(metadata["focal_gamma"], EXPECTED[index])
            self.assertTrue(torch.isfinite(criterion(logits, targets)))

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "f00_control_weighted_bce",
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
            self.assertEqual(len(recipes), 12)
            for index, recipe in enumerate(recipes):
                self.assertIn("fmt_feature", recipe)
                if index == 0:
                    self.assertNotIn("training", recipe)
                else:
                    self.assertEqual(
                        set(recipe["training"]), {"loss", "focal_gamma"}
                    )


if __name__ == "__main__":
    unittest.main()
