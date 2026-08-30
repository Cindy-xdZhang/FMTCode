"""Contracts for Task3 trainable-parameter EMA search 40.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import (
    _TrainableParameterEMA,
    _parameter_ema_decay,
)


CONFIG = "config/Verify_Task3_ParameterEMA_40.1.yaml"


class ParameterEMATests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_decay_grid_is_complete_and_one_factor(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        observed = [
            row["training"]["parameter_ema_decay"] for row in candidates
        ]
        self.assertEqual(
            observed, [None, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
        )
        for row in candidates:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(set(row["training"]), {"parameter_ema_decay"})
            self.assertEqual(row["sources"], ["feature"])

    def test_control_and_zero_tolerance_absolute_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "e00_control_none",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_decay_validation(self):
        self.assertIsNone(_parameter_ema_decay({}))
        self.assertIsNone(_parameter_ema_decay({"parameter_ema_decay": None}))
        self.assertEqual(_parameter_ema_decay({"parameter_ema_decay": 0.99}), 0.99)
        for value in (0.0, 1.0, -0.1, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _parameter_ema_decay({"parameter_ema_decay": value})

    def test_ema_updates_only_trainable_parameters(self):
        model = torch.nn.Linear(2, 1, bias=True)
        model.bias.requires_grad_(False)
        with torch.no_grad():
            model.weight.fill_(0.0)
            model.bias.fill_(7.0)
        ema = _TrainableParameterEMA(model, 0.5)
        self.assertEqual(set(ema.shadow), {"weight"})
        with torch.no_grad():
            model.weight.fill_(2.0)
            model.bias.fill_(9.0)
        ema.update(model)
        torch.testing.assert_close(ema.shadow["weight"], torch.ones_like(model.weight))
        self.assertEqual(float(model.bias), 9.0)

    def test_average_parameters_restores_live_training_state(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema = _TrainableParameterEMA(model, 0.5)
        with torch.no_grad():
            model.weight.fill_(4.0)
        ema.update(model)
        with ema.average_parameters(model):
            self.assertEqual(float(model.weight), 2.0)
        self.assertEqual(float(model.weight), 4.0)

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 9)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(set(recipe["training"]), {"parameter_ema_decay"})


if __name__ == "__main__":
    unittest.main()
