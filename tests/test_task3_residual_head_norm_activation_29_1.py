"""Contracts for Task3 residual-head normalization x activation search 29.1."""

import unittest

import torch
from torch import nn

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    _RMSNorm,
    residual_model_kwargs,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualHeadNormActivation_29.1.yaml"


def _model(**overrides):
    raw = PathlineBinaryClassifier3D(
        variant="raw", temporal_width=8, embedding_dim=16
    )
    options = {
        "embedding_dim": 16,
        "auxiliary_dim": 8,
        "head_architecture": "deep_mlp",
        "head_hidden_dim": 12,
        "head_depth": 2,
        "head_dropout": 0.0,
    }
    options.update(overrides)
    return PathlineFMTResidualClassifier3D(raw, fmt_dim=10, **options)


class ResidualHeadNormActivationTests(unittest.TestCase):
    def test_registered_factorial_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_is_complete_three_by_three(self):
        spec = _load_optimization_spec(CONFIG)
        observed = {
            (
                row["model"]["head_normalization"],
                row["model"]["head_activation"],
            )
            for row in spec["optimization_candidates"]
        }
        expected = {
            (normalization, activation)
            for normalization in ("layernorm", "rmsnorm", "none")
            for activation in ("gelu", "silu", "relu")
        }
        self.assertEqual(observed, expected)

    def test_exact_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        control = spec["optimization_candidates"][0]
        self.assertEqual(control["id"], "n00_control_layernorm_gelu")
        self.assertEqual(control["model"], {
            "head_normalization": "layernorm",
            "head_activation": "gelu",
        })
        guard = spec["optimization_selection"]["absolute_fmt_guard"]
        self.assertEqual(guard["control_optimization_id"], control["id"])
        self.assertEqual(guard["f1_tolerance"], 0.0)
        self.assertEqual(guard["average_precision_tolerance"], 0.0)

    def test_candidates_only_change_registered_head_factors(self):
        spec = _load_optimization_spec(CONFIG)
        for row in spec["optimization_candidates"]:
            self.assertEqual(row["sources"], ["feature"])
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(
                set(row["model"]),
                {"head_normalization", "head_activation"},
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
                self.assertEqual(
                    set(recipe["model"]),
                    {"head_normalization", "head_activation"},
                )

    def test_explicit_control_is_byte_compatible_with_legacy_default(self):
        torch.manual_seed(7068)
        legacy = _model()
        torch.manual_seed(7068)
        explicit = _model(
            head_normalization="layernorm", head_activation="gelu"
        )
        self.assertEqual(list(legacy.state_dict()), list(explicit.state_dict()))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]))

    def test_all_registered_layers_build_and_backpropagate(self):
        normalization_classes = {
            "layernorm": nn.LayerNorm,
            "rmsnorm": _RMSNorm,
            "none": nn.Identity,
        }
        activation_classes = {
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
        }
        for normalization, normalization_class in normalization_classes.items():
            for activation, activation_class in activation_classes.items():
                model = _model(
                    head_normalization=normalization,
                    head_activation=activation,
                )
                modules = list(model.residual_head)
                self.assertIsInstance(modules[1], normalization_class)
                self.assertIsInstance(modules[2], activation_class)
                self.assertIsInstance(modules[5], normalization_class)
                self.assertIsInstance(modules[6], activation_class)
                pathlines = torch.randn(4, 7, 6, 3)
                auxiliary = torch.randn(4, 10)
                loss = model(pathlines, auxiliary).square().mean()
                loss.backward()
                gradients = [
                    parameter.grad for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(
                    gradient is not None and torch.isfinite(gradient).all()
                    for gradient in gradients
                ))

    def test_model_kwargs_preserve_legacy_defaults_and_new_fields(self):
        defaults = residual_model_kwargs({})
        self.assertEqual(defaults["head_normalization"], "layernorm")
        self.assertEqual(defaults["head_activation"], "gelu")
        changed = residual_model_kwargs({
            "head_normalization": "rmsnorm",
            "head_activation": "silu",
        })
        self.assertEqual(changed["head_normalization"], "rmsnorm")
        self.assertEqual(changed["head_activation"], "silu")

    def test_non_deep_head_rejects_silent_unused_override(self):
        with self.assertRaisesRegex(ValueError, "require.*deep_mlp"):
            _model(head_architecture="mlp", head_activation="silu")


if __name__ == "__main__":
    unittest.main()
