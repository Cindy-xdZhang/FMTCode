"""Contracts for the paired Task3 auxiliary-dropout search 53.1."""

import unittest

import torch.nn as nn

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Search_Task3_FMTResidual_3D import _candidate_spec
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _optimization_candidate,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_AuxiliaryDropout_53.1.yaml"
EXPECTED = [0.0, 0.025, 0.050, 0.100, 0.150, 0.200,
            0.300, 0.400, 0.500, 0.600, 0.700]


class AuxiliaryDropoutTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.200)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_auxiliary_dropout(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["feature"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(0.0)
                continue
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(set(row["model"]), {"auxiliary_dropout"})
            observed.append(float(row["model"]["auxiliary_dropout"]))
        self.assertEqual(observed, EXPECTED)

    def test_resolved_candidates_and_run_specs_preserve_exact_rates(self):
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
        matching_groups = [
            group for group in spec["groups"].values()
            if dataset in group["datasets"]
        ]
        self.assertEqual(len(matching_groups), 1)
        group = matching_groups[0]
        for index, expected in enumerate(EXPECTED):
            candidate = _optimization_candidate(spec, manifest, dataset, index)
            self.assertEqual(
                residual_model_kwargs(candidate)["auxiliary_dropout"], expected
            )
            run_spec = _candidate_spec(
                spec, group, candidate, dataset, 40, "fmt", spec["output_root"], 8
            )
            self.assertEqual(run_spec["model"]["auxiliary_dropout"], expected)

    def test_model_places_one_dropout_after_auxiliary_projection(self):
        for expected in EXPECTED:
            raw = PathlineBinaryClassifier3D(variant="raw")
            model = PathlineFMTResidualClassifier3D(
                raw, fmt_dim=8, head_architecture="deep_mlp",
                head_hidden_dim=16, auxiliary_dropout=expected,
            )
            self.assertIsInstance(model.auxiliary_dropout, nn.Dropout)
            self.assertEqual(model.auxiliary_dropout.p, expected)
        with self.assertRaisesRegex(ValueError, "auxiliary_dropout"):
            PathlineFMTResidualClassifier3D(
                PathlineBinaryClassifier3D(variant="raw"),
                fmt_dim=8, auxiliary_dropout=1.0,
            )

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "a00_control_auxdrop0000",
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
            self.assertEqual(len(recipes), 11)
            for index, recipe in enumerate(recipes):
                self.assertIn("fmt_feature", recipe)
                if index == 0:
                    self.assertNotIn("model", recipe)
                else:
                    self.assertEqual(
                        set(recipe["model"]), {"auxiliary_dropout"}
                    )


if __name__ == "__main__":
    unittest.main()
