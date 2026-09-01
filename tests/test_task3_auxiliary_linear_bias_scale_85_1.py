"""Contracts for paired auxiliary Linear-bias scale search 85.1."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

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


CONFIG = "config/Verify_Task3_AuxiliaryLinearBiasScale_85.1.yaml"
SCALES = [None, 0.0, 0.1, 0.25, 0.5, 2.0, 4.0, 8.0]


def _model(**overrides):
    raw = PathlineBinaryClassifier3D(
        variant="raw", temporal_width=8, embedding_dim=16
    )
    options = {
        "embedding_dim": 16,
        "auxiliary_dim": 8,
        "auxiliary_projection": "mlp_layernorm_gelu",
        "auxiliary_hidden_dim": 12,
        "head_architecture": "deep_mlp",
        "head_hidden_dim": 12,
        "head_depth": 2,
    }
    options.update(overrides)
    return PathlineFMTResidualClassifier3D(raw, fmt_dim=10, **options)


def _linear_bias_names(model):
    return {
        f"fmt_encoder.{name}.bias"
        for name, child in model.fmt_encoder.named_modules()
        if isinstance(child, nn.Linear) and child.bias is not None
    }


class AuxiliaryLinearBiasScaleTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 8)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["model.auxiliary_linear_bias_initial_scale"],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.224)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.895)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_registered_bias_scale(self):
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
                    {"auxiliary_linear_bias_initial_scale"},
                )
                observed.append(float(
                    row["model"]["auxiliary_linear_bias_initial_scale"]
                ))
        self.assertEqual(observed, SCALES)

    def test_source_recipe_is_preserved_except_bias_scale(self):
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
                            "auxiliary_linear_weight_initialization": (
                                "orthogonal"
                            ),
                            "auxiliary_linear_weight_initialization_gain": 0.5,
                        },
                        "training": {"max_epochs": 160},
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_AuxiliaryLinearWeightInitializationPortfolio_84.1"
                ),
                "confirmation_opened": False,
                "primary_by_group": primary,
            }), encoding="utf-8")
            spec["combination_sources"]["portfolio"]["selection"] = str(
                source_path
            )
            resolved, hashes = _resolve_combination_candidates(spec)
        self.assertEqual(set(hashes), {"portfolio"})
        for rows in resolved.values():
            self.assertNotIn(
                "auxiliary_linear_bias_initial_scale", rows[0]["model"]
            )
            self.assertEqual(
                rows[1]["model"]["auxiliary_linear_weight_initialization"],
                "orthogonal",
            )
            self.assertEqual(
                rows[1]["model"]["auxiliary_linear_bias_initial_scale"], 0.0
            )
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)

    def test_default_is_exact_historical_no_op(self):
        torch.manual_seed(7068)
        legacy = _model()
        legacy_next = torch.rand(5)
        torch.manual_seed(7068)
        explicit = _model(auxiliary_linear_bias_initial_scale=1.0)
        explicit_next = torch.rand(5)
        self.assertEqual(explicit.auxiliary_linear_bias_layer_count, 0)
        self.assertTrue(torch.equal(legacy_next, explicit_next))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)

    def test_candidate_changes_only_linear_biases_without_rng_shift(self):
        torch.manual_seed(17)
        control = _model()
        control_next = torch.rand(5)
        torch.manual_seed(17)
        candidate = _model(auxiliary_linear_bias_initial_scale=0.5)
        candidate_next = torch.rand(5)
        changed = _linear_bias_names(candidate)
        self.assertEqual(candidate.auxiliary_linear_bias_layer_count, 2)
        self.assertTrue(torch.equal(control_next, candidate_next))
        for name, value in control.state_dict().items():
            other = candidate.state_dict()[name]
            if name in changed:
                self.assertTrue(torch.equal(other, value * 0.5), name)
            else:
                self.assertTrue(torch.equal(value, other), name)

    def test_zero_bias_and_blockwise_branches(self):
        torch.manual_seed(29)
        model = _model(
            auxiliary_projection="blockwise_mlp_gelu",
            auxiliary_block_dims=[4, 6],
            auxiliary_linear_bias_initial_scale=0.0,
        )
        biases = [
            child.bias for child in model.fmt_encoder.modules()
            if isinstance(child, nn.Linear)
        ]
        self.assertEqual(len(biases), 4)
        self.assertEqual(model.auxiliary_linear_bias_layer_count, 4)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in biases))

    def test_invalid_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            _model(auxiliary_linear_bias_initial_scale=-0.1)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            _model(auxiliary_linear_bias_initial_scale=float("nan"))

    def test_model_kwargs_and_evidence_contract(self):
        self.assertEqual(
            residual_model_kwargs({})["auxiliary_linear_bias_initial_scale"],
            1.0,
        )
        self.assertEqual(
            residual_model_kwargs({
                "auxiliary_linear_bias_initial_scale": 0.25
            })["auxiliary_linear_bias_initial_scale"],
            0.25,
        )
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "b00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_linear_bias_scale_85.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_linear_bias_scale_85.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-79%24", gpu)
        self.assertIn("expected 480 per-run CSV files", evidence)
        self.assertIn("expected 480 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
