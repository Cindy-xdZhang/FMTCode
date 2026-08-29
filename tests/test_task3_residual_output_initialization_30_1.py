"""Contracts for Task3 residual-output initialization search 30.1."""

import unittest

import torch
from torch import nn

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualOutputInitialization_30.1.yaml"


def _model(**overrides):
    raw = PathlineBinaryClassifier3D(
        variant="raw", temporal_width=8, embedding_dim=16
    )
    options = {
        "embedding_dim": 16,
        "auxiliary_dim": 8,
        "residual_input": "geometry_fmt",
        "head_architecture": "deep_mlp",
        "head_hidden_dim": 12,
        "head_depth": 2,
        "head_dropout": 0.0,
    }
    options.update(overrides)
    return PathlineFMTResidualClassifier3D(raw, fmt_dim=10, **options)


def _terminal_linear(module):
    return [child for child in module.modules()
            if isinstance(child, nn.Linear)][-1]


class ResidualOutputInitializationTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_exact_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        control = spec["optimization_candidates"][0]
        self.assertEqual(control["id"], "i00_control_default")
        self.assertEqual(control["model"], {
            "residual_output_initialization": "default",
            "residual_output_initialization_scale": 1.0,
        })
        guard = spec["optimization_selection"]["absolute_fmt_guard"]
        self.assertEqual(guard["control_optimization_id"], control["id"])
        self.assertEqual(guard["f1_tolerance"], 0.0)
        self.assertEqual(guard["average_precision_tolerance"], 0.0)

    def test_candidates_only_change_registered_initialization(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for row in spec["optimization_candidates"]:
            self.assertEqual(row["sources"], ["feature"])
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(set(row["model"]), {
                "residual_output_initialization",
                "residual_output_initialization_scale",
            })
            observed.append((
                row["model"]["residual_output_initialization"],
                float(row["model"]["residual_output_initialization_scale"]),
            ))
        self.assertEqual(observed, [
            ("default", 1.0), ("zero", 0.0),
            ("normal", 0.0001), ("normal", 0.001),
            ("normal", 0.005), ("normal", 0.01),
            ("normal", 0.025), ("normal", 0.05),
            ("xavier_uniform", 0.1),
        ])

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 9)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(set(recipe["model"]), {
                    "residual_output_initialization",
                    "residual_output_initialization_scale",
                })

    def test_default_is_byte_compatible_with_legacy_constructor(self):
        torch.manual_seed(7068)
        legacy = _model()
        torch.manual_seed(7068)
        explicit = _model(
            residual_output_initialization="default",
            residual_output_initialization_scale=1.0,
        )
        self.assertEqual(list(legacy.state_dict()), list(explicit.state_dict()))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]))

    def test_zero_initialization_preserves_raw_logit_for_dual_route(self):
        torch.manual_seed(11)
        model = _model(
            residual_input="dual",
            residual_output_initialization="zero",
            residual_output_initialization_scale=0.0,
        )
        pathlines = torch.randn(5, 7, 6, 3)
        auxiliary = torch.randn(5, 10)
        raw_logit, residual_logit = model.forward_components(
            pathlines, auxiliary
        )
        self.assertTrue(torch.equal(residual_logit, torch.zeros_like(residual_logit)))
        self.assertTrue(torch.equal(model(pathlines, auxiliary), raw_logit))
        self.assertTrue(torch.equal(
            _terminal_linear(model.residual_head).weight,
            torch.zeros_like(_terminal_linear(model.residual_head).weight),
        ))
        self.assertTrue(torch.equal(
            _terminal_linear(model.fmt_only_head).weight,
            torch.zeros_like(_terminal_linear(model.fmt_only_head).weight),
        ))

    def test_small_normal_is_deterministic_and_zero_bias(self):
        torch.manual_seed(23)
        first = _model(
            residual_output_initialization="normal",
            residual_output_initialization_scale=0.001,
        )
        torch.manual_seed(23)
        second = _model(
            residual_output_initialization="normal",
            residual_output_initialization_scale=0.001,
        )
        first_output = _terminal_linear(first.residual_head)
        second_output = _terminal_linear(second.residual_head)
        self.assertTrue(torch.equal(first_output.weight, second_output.weight))
        self.assertGreater(float(first_output.weight.abs().max()), 0.0)
        self.assertLess(float(first_output.weight.std()), 0.002)
        self.assertTrue(torch.equal(
            first_output.bias, torch.zeros_like(first_output.bias)
        ))

    def test_initialized_model_backpropagates_after_one_update(self):
        torch.manual_seed(31)
        model = _model(
            residual_output_initialization="zero",
            residual_output_initialization_scale=0.0,
        )
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad], lr=0.1
        )
        pathlines = torch.randn(6, 7, 6, 3)
        auxiliary = torch.randn(6, 10)
        target = torch.randn(6)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(pathlines, auxiliary) - target).square().mean()
            loss.backward()
            optimizer.step()
        gradients = [
            parameter.grad for parameter in model.fmt_encoder.parameters()
        ]
        self.assertTrue(any(
            gradient is not None and float(gradient.abs().sum()) > 0.0
            for gradient in gradients
        ))

    def test_invalid_initialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "residual_output_initialization"):
            _model(residual_output_initialization="unknown")
        with self.assertRaisesRegex(ValueError, "positive scale"):
            _model(
                residual_output_initialization="normal",
                residual_output_initialization_scale=0.0,
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            _model(
                residual_output_initialization="zero",
                residual_output_initialization_scale=-1.0,
            )

    def test_model_kwargs_preserve_defaults_and_new_fields(self):
        defaults = residual_model_kwargs({})
        self.assertEqual(defaults["residual_output_initialization"], "default")
        self.assertEqual(defaults["residual_output_initialization_scale"], 1.0)
        changed = residual_model_kwargs({
            "residual_output_initialization": "xavier_uniform",
            "residual_output_initialization_scale": 0.1,
        })
        self.assertEqual(
            changed["residual_output_initialization"], "xavier_uniform"
        )
        self.assertEqual(changed["residual_output_initialization_scale"], 0.1)


if __name__ == "__main__":
    unittest.main()
