"""Contracts for Task3 auxiliary normalization-scale search 65.1."""

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


CONFIG = "config/Verify_Task3_AuxiliaryNormScale_65.1.yaml"
EXPECTED = [None, 0.0, 0.01, 0.03, 0.10, 0.25, 0.50, 0.75, 1.50, 2.0, 4.0]


class AuxiliaryNormScaleTests(unittest.TestCase):
    def test_grid_job_count_source_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        source = spec["combination_sources"]["portfolio"]
        self.assertEqual(
            source["expected_experiment"],
            "Verify_Task3_AuxiliaryEpsilonPortfolio_64.1",
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.211)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_auxiliary_normalization_scale(self):
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
                    set(row["model"]),
                    {"auxiliary_normalization_initial_scale"},
                )
                observed.append(float(
                    row["model"]["auxiliary_normalization_initial_scale"]
                ))
        self.assertEqual(observed, EXPECTED)

    def test_portfolio_recipe_is_preserved_before_local_override(self):
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
                            "head_dropout": 0.25,
                        },
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": "Verify_Task3_AuxiliaryEpsilonPortfolio_64.1",
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
            self.assertEqual(control["head_dropout"], 0.25)
            self.assertNotIn("auxiliary_normalization_initial_scale", control)
            candidate = rows[1]["model"]
            self.assertEqual(candidate["head_dropout"], 0.25)
            self.assertEqual(candidate["auxiliary_normalization_initial_scale"], 0.0)

    @staticmethod
    def _model(scale=None, projection="linear_layernorm_gelu"):
        kwargs = {}
        if scale is not None:
            kwargs["auxiliary_normalization_initial_scale"] = scale
        return PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8,
            auxiliary_dim=4,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
            auxiliary_projection=projection,
            **kwargs,
        )

    def test_scale_changes_only_auxiliary_norm_weight_at_initialization(self):
        torch.manual_seed(7068)
        control = self._model()
        torch.manual_seed(7068)
        scaled = self._model(0.25)
        changed = []
        for name, value in control.state_dict().items():
            if not torch.equal(value, scaled.state_dict()[name]):
                changed.append(name)
        self.assertEqual(changed, ["fmt_encoder.1.weight"])
        self.assertTrue(torch.equal(
            control.fmt_encoder[1].weight,
            torch.ones_like(control.fmt_encoder[1].weight),
        ))
        self.assertTrue(torch.equal(
            scaled.fmt_encoder[1].weight,
            torch.full_like(scaled.fmt_encoder[1].weight, 0.25),
        ))
        self.assertEqual(control.auxiliary_normalization_layer_count, 0)
        self.assertEqual(scaled.auxiliary_normalization_layer_count, 1)

    def test_zero_scale_is_trainable_and_model_spec_round_trips(self):
        model = self._model(0.0)
        self.assertTrue(model.fmt_encoder[1].weight.requires_grad)
        self.assertEqual(
            residual_model_kwargs({
                "auxiliary_normalization_initial_scale": 0.0,
            })["auxiliary_normalization_initial_scale"],
            0.0,
        )
        self.assertIsNone(
            residual_model_kwargs({})[
                "auxiliary_normalization_initial_scale"
            ]
        )

    def test_invalid_scale_and_norm_free_projection_are_rejected(self):
        for invalid in (-0.01, float("inf"), float("nan")):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(
                    ValueError, "auxiliary_normalization_initial_scale"
                ):
                    self._model(invalid)
        with self.assertRaisesRegex(ValueError, "requires an auxiliary projection"):
            self._model(0.5, projection="linear")

    def test_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "c00_control_normscale",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )

    def test_slurm_scripts_match_scale_and_retain_models(self):
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_norm_scale_65.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_norm_scale_65.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-109%24", gpu)
        self.assertIn("expected 660 per-run CSV files", evidence)
        self.assertIn("expected 660 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
