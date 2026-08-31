"""Contracts for Task3 auxiliary normalization-epsilon search 69.1."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

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


CONFIG = "config/Verify_Task3_AuxiliaryNormEpsilon_69.1.yaml"
EXPECTED = [None, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6,
            1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


class AuxiliaryNormEpsilonTests(unittest.TestCase):
    def test_grid_job_count_source_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_AuxiliaryNormBiasPortfolio_68.1",
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.213)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_auxiliary_normalization_epsilon(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["portfolio"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(None)
            else:
                self.assertEqual(set(row), {"id", "sources", "model"})
                self.assertEqual(
                    set(row["model"]), {"auxiliary_normalization_epsilon"}
                )
                observed.append(float(
                    row["model"]["auxiliary_normalization_epsilon"]
                ))
        self.assertEqual(observed, EXPECTED)

    def test_portfolio_affine_values_are_preserved_before_epsilon_override(self):
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
                            "head_dropout": 0.4,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": "Verify_Task3_AuxiliaryNormBiasPortfolio_68.1",
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
            self.assertNotIn("auxiliary_normalization_epsilon", control)
            candidate = rows[1]["model"]
            self.assertEqual(candidate["auxiliary_normalization_initial_scale"], 0.25)
            self.assertEqual(candidate["auxiliary_normalization_initial_bias"], -0.10)
            self.assertEqual(candidate["auxiliary_normalization_epsilon"], 1e-12)
            self.assertEqual(candidate["head_dropout"], 0.4)

    @staticmethod
    def _model(epsilon=None, projection="linear_layernorm_gelu"):
        kwargs = {
            "auxiliary_normalization_initial_scale": 0.25,
            "auxiliary_normalization_initial_bias": -0.10,
        }
        if epsilon is not None:
            kwargs["auxiliary_normalization_epsilon"] = epsilon
        if projection == "linear_rmsnorm_gelu":
            kwargs.pop("auxiliary_normalization_initial_bias")
        return PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8,
            auxiliary_dim=4,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
            auxiliary_projection=projection,
            **kwargs,
        )

    def test_epsilon_changes_no_checkpoint_tensor_or_random_initialization(self):
        torch.manual_seed(7068)
        control = self._model()
        torch.manual_seed(7068)
        changed = self._model(1e-2)
        self.assertEqual(control.fmt_encoder[1].eps, 1e-5)
        self.assertEqual(changed.fmt_encoder[1].eps, 1e-2)
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, changed.state_dict()[name]), name)
        self.assertEqual(changed.auxiliary_normalization_epsilon_layer_count, 1)

    def test_layernorm_and_rmsnorm_are_supported_and_round_trip(self):
        rms = self._model(1e-4, projection="linear_rmsnorm_gelu")
        self.assertEqual(rms.fmt_encoder[1].eps, 1e-4)
        self.assertEqual(
            residual_model_kwargs({
                "auxiliary_normalization_epsilon": 1e-4,
            })["auxiliary_normalization_epsilon"],
            1e-4,
        )
        self.assertIsNone(
            residual_model_kwargs({})["auxiliary_normalization_epsilon"]
        )

    def test_invalid_epsilon_and_projection_without_norm_are_rejected(self):
        for invalid in (0.0, -1e-5, float("inf"), float("nan")):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(
                    ValueError, "auxiliary_normalization_epsilon"
                ):
                    self._model(invalid)
        with self.assertRaisesRegex(ValueError, "LayerNorm or RMSNorm"):
            self._model(1e-5, projection="linear_gelu")

    def test_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "c00_control_normeps",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_slurm_scripts_match_grid_and_retain_models(self):
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_norm_epsilon_69.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_norm_epsilon_69.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-109%24", gpu)
        self.assertIn("expected 660 per-run CSV files", evidence)
        self.assertIn("expected 660 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
