"""Contracts for the paired Task3 positive-class-weight search 37.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import _build_training_loss


CONFIG = "config/Verify_Task3_PositiveWeightScale_37.1.yaml"


class PositiveWeightScaleTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 15)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 149), ("smokeBuoyancy", 14))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_positive_weight_scale(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["feature"])
            if index == 8:
                self.assertEqual(
                    set(row), {"id", "sources"},
                    "the scale-1 control must be an exact no-override cell",
                )
                observed.append(1.0)
                continue
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"positive_weight_scale"})
            observed.append(float(row["training"]["positive_weight_scale"]))
        self.assertEqual(
            observed,
            [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75,
             1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00],
        )

    def test_loss_metadata_applies_each_declared_scale_exactly(self):
        spec = _load_optimization_spec(CONFIG)
        for row in spec["optimization_candidates"]:
            training = dict(row.get("training", {}))
            expected = float(training.get("positive_weight_scale", 1.0))
            _, metadata = _build_training_loss(
                training, positive=2.0, negative=6.0,
                device=torch.device("cpu"),
            )
            self.assertEqual(metadata["positive_weight_scale"], expected)
            self.assertEqual(metadata["positive_weight"], 3.0 * expected)

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "p08_control_scale100",
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
            self.assertEqual(len(recipes), 15)
            for index, recipe in enumerate(recipes):
                self.assertIn("fmt_feature", recipe)
                if index == 8:
                    self.assertNotIn("training", recipe)
                else:
                    self.assertEqual(
                        set(recipe["training"]), {"positive_weight_scale"}
                    )


if __name__ == "__main__":
    unittest.main()
