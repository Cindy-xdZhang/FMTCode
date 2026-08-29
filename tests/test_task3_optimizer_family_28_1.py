"""Contracts for the paired Task3 optimizer-family search 28.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
)
from Verify_Task3_FMTResidual import _build_optimizer


CONFIG = "config/Verify_Task3_OptimizerFamily_28.1.yaml"


class OptimizerFamilyTests(unittest.TestCase):
    def test_registered_candidates_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 6)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 59), ("smokeBuoyancy", 5))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_only_optimizer_family_changes_after_frozen_27_1_cell(self):
        spec = _load_optimization_spec(CONFIG)
        expected = {
            "p00_control_adamw": {"optimizer": "adamw"},
            "p01_adamw_amsgrad": {
                "optimizer": "adamw", "optimizer_amsgrad": True,
            },
            "p02_adam": {"optimizer": "adam"},
            "p03_adam_amsgrad": {
                "optimizer": "adam", "optimizer_amsgrad": True,
            },
            "p04_radam": {"optimizer": "radam"},
            "p05_nadam": {"optimizer": "nadam"},
        }
        observed = {}
        for row in spec["optimization_candidates"]:
            self.assertEqual(row["sources"], ["optimizer_cell"])
            self.assertEqual(set(row), {"id", "sources", "training"})
            observed[row["id"]] = row["training"]
        self.assertEqual(observed, expected)

    def test_exact_adamw_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        guard = spec["optimization_selection"]["absolute_fmt_guard"]
        self.assertEqual(
            guard["control_optimization_id"], "p00_control_adamw"
        )
        self.assertEqual(guard["f1_tolerance"], 0.0)
        self.assertEqual(guard["average_precision_tolerance"], 0.0)

    def test_optimizer_builder_supports_registered_families(self):
        expected_classes = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "radam": torch.optim.RAdam,
            "nadam": torch.optim.NAdam,
        }
        for name, expected_class in expected_classes.items():
            parameter = torch.nn.Parameter(torch.zeros(2))
            optimizer, observed_name, amsgrad = _build_optimizer({
                "optimizer": name,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }, [parameter])
            self.assertIsInstance(optimizer, expected_class)
            self.assertEqual(observed_name, name)
            self.assertFalse(amsgrad)
            loss = (parameter - 1.0).square().sum()
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertGreater(float(parameter.detach().abs().sum()), 0.0)

    def test_amsgrad_is_explicit_and_invalid_combinations_fail(self):
        for name in ("adamw", "adam"):
            parameter = torch.nn.Parameter(torch.zeros(2))
            optimizer, _, amsgrad = _build_optimizer({
                "optimizer": name,
                "optimizer_amsgrad": True,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }, [parameter])
            self.assertTrue(amsgrad)
            self.assertTrue(optimizer.defaults["amsgrad"])
        with self.assertRaisesRegex(ValueError, "does not support"):
            _build_optimizer({
                "optimizer": "radam",
                "optimizer_amsgrad": True,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }, [torch.nn.Parameter(torch.zeros(2))])


if __name__ == "__main__":
    unittest.main()
