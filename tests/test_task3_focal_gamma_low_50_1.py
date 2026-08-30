"""Contracts for the low-gamma paired Task3 follow-up 50.1."""

import hashlib
import json
import unittest
from pathlib import Path

import torch
import yaml

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import _build_training_loss


CONFIG = "config/Verify_Task3_FocalGammaLow_50.1.yaml"
EXPECTED = [0.0, 0.010, 0.025, 0.050, 0.075, 0.100, 0.150, 0.200]


class LowFocalGammaTests(unittest.TestCase):
    def test_parent_selection_is_completed_and_frozen(self):
        overlay = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
        parent = Path(overlay["motivation_selection"])
        if not parent.is_file():
            parent = Path(overlay["motivation_selection_local_mirror"])
        self.assertTrue(parent.is_file())
        self.assertEqual(
            hashlib.sha256(parent.read_bytes()).hexdigest(),
            overlay["motivation_selection_sha256"],
        )
        payload = json.loads(parent.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["experiment"], overlay["motivation_expected_experiment"]
        )
        self.assertFalse(payload["confirmation_opened"])
        self.assertEqual(
            payload["primary_by_group"]["channel"]["optimization_id"],
            "f01_gamma010",
        )
        self.assertEqual(
            payload["primary_by_group"]["halfcylinder"]["optimization_id"],
            "f01_gamma010",
        )

    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 8)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
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
                self.assertEqual(set(row), {"id", "sources"})
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
            criterion, metadata = _build_training_loss(
                dict(row.get("training", {})), positive=2.0, negative=6.0,
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
                "control_optimization_id": "l00_control_weighted_bce",
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
            self.assertEqual(len(recipes), 8)
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
