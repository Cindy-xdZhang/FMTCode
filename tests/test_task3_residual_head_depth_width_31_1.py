"""Contracts for Task3 residual-head depth x width search 31.1."""

import unittest

import torch
from torch import nn

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualHeadDepthWidth_31.1.yaml"


def _model(width, depth):
    raw = PathlineBinaryClassifier3D(
        variant="raw", temporal_width=8, embedding_dim=16
    )
    return PathlineFMTResidualClassifier3D(
        raw,
        fmt_dim=10,
        embedding_dim=16,
        auxiliary_dim=8,
        residual_input="geometry_fmt",
        head_architecture="deep_mlp",
        head_hidden_dim=int(width),
        head_depth=int(depth),
        head_dropout=0.0,
        head_normalization="layernorm",
        head_activation="gelu",
    )


class ResidualHeadDepthWidthTests(unittest.TestCase):
    def test_registered_factorial_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 15)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 149), ("smokeBuoyancy", 14))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_is_complete_five_by_three(self):
        spec = _load_optimization_spec(CONFIG)
        observed = {
            (row["model"]["head_hidden_dim"], row["model"]["head_depth"])
            for row in spec["optimization_candidates"]
        }
        expected = {
            (width, depth)
            for width in (32, 48, 64, 80, 96)
            for depth in (1, 2, 3)
        }
        self.assertEqual(observed, expected)

    def test_exact_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        control = spec["optimization_candidates"][0]
        self.assertEqual(control["id"], "c00_control_w64_d2")
        self.assertEqual(control["model"], {
            "head_hidden_dim": 64,
            "head_depth": 2,
        })
        guard = spec["optimization_selection"]["absolute_fmt_guard"]
        self.assertEqual(guard["control_optimization_id"], control["id"])
        self.assertEqual(guard["f1_tolerance"], 0.0)
        self.assertEqual(guard["average_precision_tolerance"], 0.0)

    def test_candidates_only_change_registered_capacity_factors(self):
        spec = _load_optimization_spec(CONFIG)
        for row in spec["optimization_candidates"]:
            self.assertEqual(row["sources"], ["feature"])
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(
                set(row["model"]), {"head_hidden_dim", "head_depth"}
            )

    def test_completed_anchored_feature_resolves_for_every_family(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(resolved), set(spec["groups"]))
        self.assertEqual(set(hashes), {"feature"})
        for recipes in resolved.values():
            self.assertEqual(len(recipes), 15)
            for recipe in recipes:
                self.assertIn("fmt_feature", recipe)
                self.assertEqual(
                    set(recipe["model"]),
                    {"head_hidden_dim", "head_depth"},
                )

    def test_dense_head_depth_and_width_are_real_registered_layers(self):
        for width in (32, 64, 96):
            for depth in (1, 2, 3):
                model = _model(width, depth)
                linears = [
                    layer for layer in model.residual_head
                    if isinstance(layer, nn.Linear)
                ]
                self.assertEqual(len(linears), depth + 1)
                self.assertEqual(linears[0].out_features, width)
                self.assertTrue(all(
                    layer.out_features == width
                    for layer in linears[:-1]
                ))
                self.assertEqual(linears[-1].out_features, 1)

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

    def test_parameter_count_increases_with_width_and_depth(self):
        count = lambda model: sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertLess(count(_model(32, 2)), count(_model(64, 2)))
        self.assertLess(count(_model(64, 1)), count(_model(64, 3)))


if __name__ == "__main__":
    unittest.main()
