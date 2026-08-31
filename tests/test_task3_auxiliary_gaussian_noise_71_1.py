"""Contracts for Task3 auxiliary Gaussian-noise search 71.1."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
    trainable_parameter_count,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_AuxiliaryGaussianNoise_71.1.yaml"
EXPECTED = [None, 0.005, 0.010, 0.025, 0.050, 0.100,
            0.200, 0.300, 0.500, 0.750, 1.000]


class AuxiliaryGaussianNoiseTests(unittest.TestCase):
    def test_grid_job_count_source_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1",
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.214)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_auxiliary_noise_standard_deviation(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(None)
            else:
                self.assertEqual(set(row), {"id", "sources", "model"})
                self.assertEqual(set(row["model"]), {"auxiliary_noise_std"})
                observed.append(float(row["model"]["auxiliary_noise_std"]))
        self.assertEqual(observed, EXPECTED)

    def test_portfolio_normalization_values_are_preserved(self):
        spec = _load_optimization_spec(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "portfolio_selection.json"
            primary = {}
            for group in spec["groups"]:
                primary[group] = {
                    "optimization_id": f"source_{group}",
                    "optimization_recipe_json": json.dumps({
                        "id": f"source_{group}",
                        "fmt_feature": "fmt_all",
                        "model": {
                            "auxiliary_projection": "linear_layernorm_gelu",
                            "auxiliary_normalization_initial_scale": 0.25,
                            "auxiliary_normalization_initial_bias": -0.10,
                            "auxiliary_normalization_epsilon": 1e-4,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": "Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1",
                "confirmation_opened": False,
                "primary_by_group": primary,
            }), encoding="utf-8")
            spec["combination_sources"]["portfolio"]["selection"] = str(
                source_path
            )
            resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(hashes), {"portfolio"})
        for rows in resolved.values():
            control = rows[0]["model"]
            self.assertEqual(control["auxiliary_normalization_initial_scale"], 0.25)
            self.assertEqual(control["auxiliary_normalization_initial_bias"], -0.10)
            self.assertEqual(control["auxiliary_normalization_epsilon"], 1e-4)
            self.assertNotIn("auxiliary_noise_std", control)
            candidate = rows[1]["model"]
            self.assertEqual(candidate["auxiliary_normalization_epsilon"], 1e-4)
            self.assertEqual(candidate["auxiliary_noise_std"], 0.005)

    @staticmethod
    def _model(noise_std=0.0):
        return PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8,
            auxiliary_dim=4,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
            auxiliary_projection="linear_layernorm_gelu",
            auxiliary_noise_std=noise_std,
        )

    def test_zero_noise_preserves_state_parameters_output_and_rng(self):
        torch.manual_seed(7068)
        control = self._model()
        torch.manual_seed(7068)
        explicit_zero = self._model(0.0)
        self.assertEqual(
            trainable_parameter_count(control),
            trainable_parameter_count(explicit_zero),
        )
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, explicit_zero.state_dict()[name]), name)
        values = torch.randn(6, 4)
        explicit_zero.train()
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        result = explicit_zero.auxiliary_noise(values)
        after = torch.get_rng_state().clone()
        self.assertIs(result, values)
        self.assertTrue(torch.equal(before, after))

    def test_positive_noise_is_train_only_and_seed_reproducible(self):
        model = self._model(0.2)
        values = torch.ones(8, 4)
        model.train()
        torch.manual_seed(123)
        first = model.auxiliary_noise(values)
        torch.manual_seed(123)
        second = model.auxiliary_noise(values)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, values))
        model.eval()
        self.assertIs(model.auxiliary_noise(values), values)

    def test_invalid_noise_and_model_spec_round_trip(self):
        for invalid in (-0.1, float("inf"), float("nan")):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "auxiliary_noise_std"):
                    self._model(invalid)
        self.assertEqual(
            residual_model_kwargs({"auxiliary_noise_std": 0.25})[
                "auxiliary_noise_std"
            ],
            0.25,
        )
        self.assertEqual(residual_model_kwargs({})["auxiliary_noise_std"], 0.0)

    def test_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "c00_control_noise000",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_slurm_scripts_match_grid_and_retain_models(self):
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_gaussian_noise_71.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_gaussian_noise_71.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-109%24", gpu)
        self.assertIn("expected 660 per-run CSV files", evidence)
        self.assertIn("expected 660 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
