"""Contracts for the paired Task3 residual-head dropout search 38.1."""

import unittest

import torch.nn as nn

from FMT_Utils.PathlineClassifier_3D import (
    _dense_head,
    residual_model_kwargs,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _optimization_candidate,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualDropout_38.1.yaml"
EXPECTED = [0.0, 0.025, 0.050, 0.075, 0.100,
            0.150, 0.200, 0.300, 0.400, 0.500]


class ResidualDropoutTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 10)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 99), ("smokeBuoyancy", 9))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.195)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_head_dropout(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["feature"])
            if index == 0:
                self.assertEqual(
                    set(row), {"id", "sources"},
                    "zero-dropout control must be an exact no-override cell",
                )
                observed.append(0.0)
                continue
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(set(row["model"]), {"head_dropout"})
            observed.append(float(row["model"]["head_dropout"]))
        self.assertEqual(observed, EXPECTED)

    def test_resolved_candidate_applies_each_dropout_exactly(self):
        spec = _load_optimization_spec(CONFIG)
        resolved, _ = _resolve_combination_candidates(spec)
        base = {
            group: {**spec["model_override"], "training": {}}
            for group in spec["groups"]
        }
        manifest = {
            "base_candidate_by_group": base,
            "optimization_candidates_by_group": resolved,
        }
        dataset = spec["datasets"][0]
        for index, expected in enumerate(EXPECTED):
            candidate = _optimization_candidate(
                spec, manifest, dataset, index
            )
            self.assertEqual(
                residual_model_kwargs(candidate)["head_dropout"], expected
            )

    def test_dense_head_contains_declared_dropout_probability(self):
        for expected in EXPECTED:
            head = _dense_head(
                input_dim=12,
                hidden_dim=16,
                depth=2,
                dropout=expected,
                normalization="layernorm",
                activation="gelu",
            )
            rates = [
                module.p for module in head.modules()
                if isinstance(module, nn.Dropout)
            ]
            self.assertEqual(rates, [expected, expected])

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "d00_control_dropout0000",
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
            self.assertEqual(len(recipes), 10)
            for index, recipe in enumerate(recipes):
                self.assertIn("fmt_feature", recipe)
                if index == 0:
                    self.assertNotIn("model", recipe)
                else:
                    self.assertEqual(set(recipe["model"]), {"head_dropout"})


if __name__ == "__main__":
    unittest.main()
