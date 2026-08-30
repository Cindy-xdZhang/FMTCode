"""Contracts for the high-dropout Task3 follow-up 51.1."""

import hashlib
import json
import unittest
from pathlib import Path

import torch.nn as nn
import yaml

from FMT_Utils.PathlineClassifier_3D import _dense_head, residual_model_kwargs
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _optimization_candidate,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ResidualDropoutHigh_51.1.yaml"
EXPECTED = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


class HighResidualDropoutTests(unittest.TestCase):
    def test_parent_selection_is_completed_frozen_and_boundary_motivated(self):
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
            payload["primary_by_group"]["f22raptor"]["optimization_id"],
            "d09_dropout0500",
        )
        self.assertEqual(
            payload["primary_by_group"]["boeing747"]["optimization_id"],
            "d09_dropout0500",
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

    def test_grid_changes_only_head_dropout(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["feature"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(0.0)
                continue
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(set(row["model"]), {"head_dropout"})
            observed.append(float(row["model"]["head_dropout"]))
        self.assertEqual(observed, EXPECTED)

    def test_resolved_candidates_and_dense_heads_use_exact_rates(self):
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
            candidate = _optimization_candidate(spec, manifest, dataset, index)
            self.assertEqual(
                residual_model_kwargs(candidate)["head_dropout"], expected
            )
            head = _dense_head(
                input_dim=12,
                hidden_dim=16,
                depth=2,
                dropout=expected,
                normalization="layernorm",
                activation="gelu",
            )
            rates = [module.p for module in head.modules()
                     if isinstance(module, nn.Dropout)]
            self.assertEqual(rates, [expected, expected])

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "h00_control_dropout0000",
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
                    self.assertNotIn("model", recipe)
                else:
                    self.assertEqual(set(recipe["model"]), {"head_dropout"})


if __name__ == "__main__":
    unittest.main()
