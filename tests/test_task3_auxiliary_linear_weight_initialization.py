"""Contracts for paired auxiliary linear-weight initialization search 83.1."""

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


CONFIG = "config/Verify_Task3_AuxiliaryLinearWeightInitialization_83.1.yaml"


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


def _linear_parameter_names(model):
    return {
        f"fmt_encoder.{name}.weight"
        for name, child in model.fmt_encoder.named_modules()
        if isinstance(child, nn.Linear)
    }


class AuxiliaryLinearWeightInitializationTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        self.assertEqual(
            spec["allowed_source_overrides"],
            [
                "model.auxiliary_linear_weight_initialization",
                "model.auxiliary_linear_weight_initialization_gain",
            ],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.222)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.894)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_registered_weight_initialization(self):
        spec = _load_optimization_spec(CONFIG)
        observed = [("default", 1.0)]
        control = spec["optimization_candidates"][0]
        self.assertEqual(control, {
            "id": "i00_control_source", "sources": ["portfolio"]
        })
        for row in spec["optimization_candidates"][1:]:
            self.assertEqual(row["sources"], ["portfolio"])
            self.assertEqual(set(row), {"id", "sources", "model"})
            self.assertEqual(set(row["model"]), {
                "auxiliary_linear_weight_initialization",
                "auxiliary_linear_weight_initialization_gain",
            })
            observed.append((
                row["model"]["auxiliary_linear_weight_initialization"],
                float(row["model"][
                    "auxiliary_linear_weight_initialization_gain"
                ]),
            ))
        self.assertEqual(observed, [
            ("default", 1.0),
            ("xavier_uniform", 0.25),
            ("xavier_uniform", 0.5),
            ("xavier_uniform", 1.0),
            ("xavier_uniform", 2.0 ** 0.5),
            ("orthogonal", 0.25),
            ("orthogonal", 0.5),
            ("orthogonal", 1.0),
            ("orthogonal", 2.0 ** 0.5),
        ])

    def test_source_recipe_is_preserved_except_registered_fields(self):
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
                        "model": {"auxiliary_post_normalization": "rms"},
                        "training": {"max_epochs": 160},
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1"
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
                "auxiliary_linear_weight_initialization", rows[0]["model"]
            )
            self.assertEqual(rows[0]["training"]["max_epochs"], 160)
            self.assertEqual(
                rows[1]["model"]["auxiliary_post_normalization"], "rms"
            )
            self.assertEqual(
                rows[1]["model"]["auxiliary_linear_weight_initialization"],
                "xavier_uniform",
            )

    def test_default_is_exact_historical_no_op(self):
        torch.manual_seed(7068)
        legacy = _model()
        legacy_next = torch.rand(5)
        torch.manual_seed(7068)
        explicit = _model(
            auxiliary_linear_weight_initialization="default",
            auxiliary_linear_weight_initialization_gain=1.0,
        )
        explicit_next = torch.rand(5)
        self.assertEqual(explicit.auxiliary_linear_weight_layer_count, 0)
        self.assertTrue(torch.equal(legacy_next, explicit_next))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)

    def test_candidate_changes_only_projection_linear_weights(self):
        torch.manual_seed(17)
        control = _model()
        control_next = torch.rand(5)
        torch.manual_seed(17)
        candidate = _model(
            auxiliary_linear_weight_initialization="orthogonal",
            auxiliary_linear_weight_initialization_gain=0.5,
        )
        candidate_next = torch.rand(5)
        changed = _linear_parameter_names(candidate)
        self.assertEqual(candidate.auxiliary_linear_weight_layer_count, 2)
        self.assertTrue(torch.equal(control_next, candidate_next))
        self.assertTrue(changed)
        for name, value in control.state_dict().items():
            other = candidate.state_dict()[name]
            if name in changed:
                self.assertFalse(torch.equal(value, other), name)
            else:
                self.assertTrue(torch.equal(value, other), name)

    def test_candidate_is_deterministic_and_preserves_biases(self):
        torch.manual_seed(23)
        control = _model()
        torch.manual_seed(23)
        first = _model(
            auxiliary_linear_weight_initialization="xavier_uniform",
            auxiliary_linear_weight_initialization_gain=1.0,
        )
        torch.manual_seed(23)
        second = _model(
            auxiliary_linear_weight_initialization="xavier_uniform",
            auxiliary_linear_weight_initialization_gain=1.0,
        )
        for name, value in first.fmt_encoder.state_dict().items():
            self.assertTrue(
                torch.equal(value, second.fmt_encoder.state_dict()[name]), name
            )
            if name.endswith("bias"):
                self.assertTrue(
                    torch.equal(value, control.fmt_encoder.state_dict()[name]),
                    name,
                )

    def test_blockwise_projection_initializes_every_branch(self):
        torch.manual_seed(29)
        model = _model(
            auxiliary_projection="blockwise_mlp_gelu",
            auxiliary_block_dims=[4, 6],
            auxiliary_linear_weight_initialization="orthogonal",
            auxiliary_linear_weight_initialization_gain=1.0,
        )
        layers = [
            child for child in model.fmt_encoder.modules()
            if isinstance(child, nn.Linear)
        ]
        self.assertEqual(len(layers), 4)
        self.assertEqual(model.auxiliary_linear_weight_layer_count, 4)

    def test_invalid_scheme_and_gain_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "auxiliary_linear_weight_initialization"
        ):
            _model(auxiliary_linear_weight_initialization="unknown")
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            _model(auxiliary_linear_weight_initialization_gain=0.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            _model(auxiliary_linear_weight_initialization_gain=float("nan"))

    def test_model_kwargs_preserve_new_fields(self):
        defaults = residual_model_kwargs({})
        self.assertEqual(
            defaults["auxiliary_linear_weight_initialization"], "default"
        )
        self.assertEqual(
            defaults["auxiliary_linear_weight_initialization_gain"], 1.0
        )
        changed = residual_model_kwargs({
            "auxiliary_linear_weight_initialization": "orthogonal",
            "auxiliary_linear_weight_initialization_gain": 0.5,
        })
        self.assertEqual(
            changed["auxiliary_linear_weight_initialization"], "orthogonal"
        )
        self.assertEqual(
            changed["auxiliary_linear_weight_initialization_gain"], 0.5
        )

    def test_zero_tolerance_guard_and_checkpoint_contract(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "i00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_linear_weight_initialization_83.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_linear_weight_initialization_83.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-89%24", gpu)
        self.assertIn("expected 540 per-run CSV files", evidence)
        self.assertIn("expected 540 temporary checkpoints", evidence)
        self.assertIn("Audit_Task3_ParameterSearch.py", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
