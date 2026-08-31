"""Contracts for Task3 fixed auxiliary post-normalization search 75.1."""

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


CONFIG = "config/Verify_Task3_AuxiliaryPostNorm_75.1.yaml"
EXPECTED = [None, "center", "rms", "layer"]


class AuxiliaryPostNormTests(unittest.TestCase):
    def test_grid_job_count_source_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 4)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 39), ("smokeBuoyancy", 3))
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1",
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.216)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_post_normalization_mode(self):
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
                    set(row["model"]), {"auxiliary_post_normalization"}
                )
                observed.append(row["model"]["auxiliary_post_normalization"])
        self.assertEqual(observed, EXPECTED)

    def test_portfolio_noise_scale_and_normalization_are_preserved(self):
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
                            "auxiliary_noise_std": 0.2,
                            "auxiliary_feature_scale": 1.5,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": "Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1",
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
            self.assertEqual(control["auxiliary_noise_std"], 0.2)
            self.assertEqual(control["auxiliary_feature_scale"], 1.5)
            self.assertNotIn("auxiliary_post_normalization", control)
            candidate = rows[1]["model"]
            self.assertEqual(candidate["auxiliary_normalization_epsilon"], 1e-4)
            self.assertEqual(candidate["auxiliary_noise_std"], 0.2)
            self.assertEqual(candidate["auxiliary_feature_scale"], 1.5)
            self.assertEqual(candidate["auxiliary_post_normalization"], "center")

    @staticmethod
    def _model(mode="none"):
        return PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8,
            auxiliary_dim=4,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
            auxiliary_projection="linear_layernorm_gelu",
            auxiliary_post_normalization=mode,
        )

    def test_explicit_none_preserves_state_parameters_outputs_and_rng(self):
        torch.manual_seed(7068)
        control = self._model()
        torch.manual_seed(7068)
        explicit_none = self._model("none")
        self.assertEqual(
            trainable_parameter_count(control),
            trainable_parameter_count(explicit_none),
        )
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, explicit_none.state_dict()[name]), name)
        pathlines = torch.randn(5, 7, 6, 3)
        features = torch.randn(5, 8)
        control.eval()
        explicit_none.eval()
        before = torch.get_rng_state().clone()
        expected = control.forward_components(pathlines, features)
        middle = torch.get_rng_state().clone()
        observed = explicit_none.forward_components(pathlines, features)
        after = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(before, middle))
        self.assertTrue(torch.equal(middle, after))
        for left, right in zip(expected, observed):
            self.assertTrue(torch.equal(left, right))

    def test_fixed_modes_have_registered_statistics(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, -1.0, 2.0, 5.0]])
        none = self._model("none").auxiliary_post_normalizer(values)
        self.assertIs(none, values)
        centered = self._model("center").auxiliary_post_normalizer(values)
        self.assertTrue(torch.allclose(centered.mean(dim=-1), torch.zeros(2)))
        rms = self._model("rms").auxiliary_post_normalizer(values)
        self.assertTrue(torch.allclose(
            rms.square().mean(dim=-1), torch.ones(2), atol=2e-5
        ))
        layer = self._model("layer").auxiliary_post_normalizer(values)
        self.assertTrue(torch.allclose(layer.mean(dim=-1), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(
            layer.square().mean(dim=-1), torch.ones(2), atol=2e-5
        ))

    def test_invalid_mode_and_model_spec_round_trip(self):
        with self.assertRaisesRegex(ValueError, "auxiliary_post_normalization"):
            self._model("batchnorm")
        self.assertEqual(
            residual_model_kwargs({"auxiliary_post_normalization": "rms"})[
                "auxiliary_post_normalization"
            ],
            "rms",
        )
        self.assertEqual(
            residual_model_kwargs({})["auxiliary_post_normalization"], "none"
        )

    def test_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "n00_control_none",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_slurm_scripts_match_grid_and_retain_models(self):
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_post_norm_75.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_post_norm_75.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-39%24", gpu)
        self.assertIn("expected 240 per-run CSV files", evidence)
        self.assertIn("expected 240 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
