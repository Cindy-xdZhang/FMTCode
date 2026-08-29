import unittest
from pathlib import Path

import numpy as np
import torch

from Search_Task3_FMTResidual_3D import _candidate_spec
from Search_Task3_LossOptimization_7_1 import (
    _apply_absolute_fmt_guard,
    _decode_job,
    _load_optimization_spec,
)
from Verify_Task3_FMTResidual import (
    _probabilities,
    _residual_gate_numpy,
    _residual_gate_parameters,
    _residual_gate_torch,
)


CONFIG = "config/Verify_Task3_ConfidenceGatedResidual_23.1.yaml"


class Task3ConfidenceGatedResidualTests(unittest.TestCase):
    def test_gate_numpy_and_torch_are_symmetric_and_identical(self):
        raw = np.asarray([-10.0, -2.0, 0.0, 2.0, 10.0], dtype=np.float64)
        model = {
            "residual_gate": "raw_uncertainty",
            "residual_gate_temperature": 1.0,
            "residual_gate_floor": 0.25,
        }
        observed = _residual_gate_numpy(raw, model)
        torch_observed = _residual_gate_torch(
            torch.from_numpy(raw), model
        ).numpy()
        np.testing.assert_allclose(observed, torch_observed, atol=1e-12)
        np.testing.assert_allclose(observed, observed[::-1], atol=1e-12)
        self.assertAlmostEqual(float(observed[2]), 1.0)
        self.assertGreater(float(observed[0]), 0.25)
        self.assertLess(float(observed[0]), 0.251)

    def test_none_gate_is_exact_historical_probability(self):
        raw = np.asarray([-1.0, 0.5], dtype=np.float64)
        residual = np.asarray([0.4, -0.2], dtype=np.float64)
        model = {
            "residual_gate": "none",
            "residual_gate_temperature": 1.0,
            "residual_gate_floor": 0.0,
        }
        expected = 1.0 / (1.0 + np.exp(-(raw + 1.5 * residual)))
        np.testing.assert_array_equal(
            _residual_gate_numpy(raw, model), np.ones_like(raw)
        )
        np.testing.assert_array_equal(
            _probabilities(raw, residual, 1.5, model), expected
        )

    def test_invalid_gate_hyperparameters_fail_explicitly(self):
        invalid = (
            ({"residual_gate": "unknown"}, "residual_gate"),
            ({"residual_gate_temperature": 0.0}, "temperature"),
            ({"residual_gate_floor": -0.1}, "floor"),
            ({"residual_gate_floor": 1.1}, "floor"),
            ({"residual_gate": "none", "residual_gate_floor": 0.5}, "none"),
        )
        for model, message in invalid:
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, message):
                    _residual_gate_parameters(model)

    def test_candidate_spec_preserves_gate_into_checkpoint_model_config(self):
        spec = {
            "experiment": "x", "expected_slices": 10,
            "sampled_steps": 32, "raw_pca_random_state": 7068,
            "raw_wide_parameter_count": 999999,
            "screen_split": {"train": [0], "validation": [1]},
            "training": {"batch_size": 8},
        }
        candidate = {
            "id": "g", "fmt_feature": "fmt_all",
            "residual_gate": "raw_uncertainty",
            "residual_gate_temperature": 2.0,
            "residual_gate_floor": 0.5,
        }
        group = {
            "source_cache_root": "source", "label_cache_root": "labels",
            "raw_checkpoint_dir": "raw",
        }
        run = _candidate_spec(
            spec, group, candidate, "channel", 40, "fmt",
            Path("out"), 16,
        )
        self.assertEqual(run["model"]["residual_gate"], "raw_uncertainty")
        self.assertEqual(run["model"]["residual_gate_temperature"], 2.0)
        self.assertEqual(run["model"]["residual_gate_floor"], 0.5)

    def test_absolute_fmt_guard_rejects_gap_from_fmt_degradation(self):
        rows = [
            {
                "optimization_id": "g00_control", "eligible": True,
                "status": "", "ineligible_reasons_json": "[]",
                "dataset_macro_fmt_f1": 0.90,
                "dataset_macro_fmt_average_precision": 0.95,
            },
            {
                "optimization_id": "pass", "eligible": True,
                "status": "", "ineligible_reasons_json": "[]",
                "dataset_macro_fmt_f1": 0.899,
                "dataset_macro_fmt_average_precision": 0.949,
            },
            {
                "optimization_id": "f1_drop", "eligible": True,
                "status": "", "ineligible_reasons_json": "[]",
                "dataset_macro_fmt_f1": 0.897,
                "dataset_macro_fmt_average_precision": 0.96,
            },
            {
                "optimization_id": "ap_drop", "eligible": True,
                "status": "", "ineligible_reasons_json": "[]",
                "dataset_macro_fmt_f1": 0.91,
                "dataset_macro_fmt_average_precision": 0.947,
            },
        ]
        selection = {"absolute_fmt_guard": {
            "control_optimization_id": "g00_control",
            "f1_tolerance": 0.002,
            "average_precision_tolerance": 0.002,
        }}
        guarded = _apply_absolute_fmt_guard(rows, selection)
        self.assertTrue(guarded[0]["eligible"])
        self.assertTrue(guarded[1]["eligible"])
        self.assertFalse(guarded[2]["eligible"])
        self.assertFalse(guarded[3]["eligible"])
        self.assertEqual(guarded[2]["status"], "absolute_fmt_guard_failed")
        self.assertEqual(guarded[3]["status"], "absolute_fmt_guard_failed")

    def test_config_declares_complete_factorial_and_closed_confirmation(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(set(spec["combination_sources"]), {"core"})
        self.assertEqual(len(spec["optimization_candidates"]), 13)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 129), ("smokeBuoyancy", 12))
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"]
            ["control_optimization_id"],
            "g00_control",
        )


if __name__ == "__main__":
    unittest.main()
