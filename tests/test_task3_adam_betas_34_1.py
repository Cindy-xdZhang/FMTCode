"""Contracts for the paired Task3 AdamW beta search 34.1."""

import unittest

import torch

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import _build_optimizer, _optimizer_betas


CONFIG = "config/Verify_Task3_AdamBetas_34.1.yaml"


class AdamBetaTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_is_complete_and_changes_only_two_betas(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for row in spec["optimization_candidates"]:
            self.assertEqual(set(row), {"id", "sources", "training"})
            self.assertEqual(
                set(row["training"]), {"optimizer_beta1", "optimizer_beta2"}
            )
            self.assertEqual(row["sources"], ["feature"])
            observed.append(_optimizer_betas(row["training"])[0])
        self.assertEqual(
            observed,
            [
                (0.9, 0.999),
                (0.5, 0.9), (0.5, 0.99), (0.5, 0.999),
                (0.9, 0.9), (0.9, 0.99),
                (0.95, 0.9), (0.95, 0.99), (0.95, 0.999),
            ],
        )

    def test_null_control_is_exact_default_and_guard_reference(self):
        spec = _load_optimization_spec(CONFIG)
        control = spec["optimization_candidates"][0]
        self.assertEqual(
            control["training"],
            {"optimizer_beta1": None, "optimizer_beta2": None},
        )
        betas, overridden = _optimizer_betas(control["training"])
        self.assertEqual(betas, (0.9, 0.999))
        self.assertFalse(overridden)
        guard = spec["optimization_selection"]["absolute_fmt_guard"]
        self.assertEqual(
            guard,
            {
                "control_optimization_id": "b00_control_b1_090_b2_0999",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_builder_applies_explicit_betas_without_changing_defaults(self):
        base = {"learning_rate": 0.001, "weight_decay": 0.0001}
        default_parameter = torch.nn.Parameter(torch.zeros(2))
        default_optimizer, _, _ = _build_optimizer(base, [default_parameter])
        null_parameter = torch.nn.Parameter(torch.zeros(2))
        null_optimizer, _, _ = _build_optimizer({
            **base, "optimizer_beta1": None, "optimizer_beta2": None,
        }, [null_parameter])
        self.assertEqual(default_optimizer.defaults, null_optimizer.defaults)

        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer, _, _ = _build_optimizer({
            **base, "optimizer_beta1": 0.5, "optimizer_beta2": 0.99,
        }, [parameter])
        self.assertEqual(optimizer.defaults["betas"], (0.5, 0.99))

    def test_invalid_betas_fail_before_training(self):
        for key in ("optimizer_beta1", "optimizer_beta2"):
            for value in (-0.1, 1.0, float("inf"), float("nan")):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        _optimizer_betas({key: value})

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 9)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(
                    set(recipe["training"]),
                    {"optimizer_beta1", "optimizer_beta2"},
                )


if __name__ == "__main__":
    unittest.main()
