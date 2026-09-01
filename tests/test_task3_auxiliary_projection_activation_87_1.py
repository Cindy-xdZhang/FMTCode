"""Contracts for paired auxiliary projection-activation search 87.1."""

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


CONFIG = "config/Verify_Task3_AuxiliaryProjectionActivation_87.1.yaml"
ACTIVATIONS = [
    None, "identity", "silu", "relu", "leaky_relu_01", "elu", "mish",
    "tanh",
]


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


def _activation_types(model):
    supported = (
        nn.Identity, nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.ELU,
        nn.Mish, nn.Tanh,
    )
    return [
        type(child) for child in model.fmt_encoder.modules()
        if isinstance(child, supported)
    ]


class AuxiliaryProjectionActivationTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 8)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 79), ("smokeBuoyancy", 7))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["model.auxiliary_projection_activation_override"],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.226)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.896)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_registered_activation(self):
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
                    {"auxiliary_projection_activation_override"},
                )
                observed.append(
                    row["model"]["auxiliary_projection_activation_override"]
                )
        self.assertEqual(observed, ACTIVATIONS)

    def test_source_recipe_is_preserved_except_activation(self):
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
                            "auxiliary_linear_bias_initial_scale": 0.25,
                        },
                        "training": {"max_epochs": 160},
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_AuxiliaryLinearBiasScalePortfolio_86.1"
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
                "auxiliary_projection_activation_override", rows[0]["model"]
            )
            self.assertEqual(
                rows[1]["model"]["auxiliary_linear_weight_initialization"],
                "orthogonal",
            )
            self.assertEqual(
                rows[1]["model"]["auxiliary_linear_bias_initial_scale"], 0.25
            )
            self.assertEqual(
                rows[1]["model"]["auxiliary_projection_activation_override"],
                "identity",
            )
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)

    def test_default_is_exact_historical_no_op(self):
        torch.manual_seed(7068)
        legacy = _model()
        legacy_next = torch.rand(5)
        torch.manual_seed(7068)
        explicit = _model(auxiliary_projection_activation_override="source")
        explicit_next = torch.rand(5)
        self.assertEqual(
            explicit.auxiliary_projection_activation_layer_count, 0
        )
        self.assertEqual(_activation_types(explicit), [nn.GELU, nn.GELU])
        self.assertTrue(torch.equal(legacy_next, explicit_next))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)

    def test_candidates_change_only_activation_without_rng_or_capacity_shift(self):
        expected = {
            "identity": nn.Identity,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "leaky_relu_01": nn.LeakyReLU,
            "elu": nn.ELU,
            "mish": nn.Mish,
            "tanh": nn.Tanh,
        }
        for name, activation_type in expected.items():
            torch.manual_seed(17)
            control = _model()
            control_next = torch.rand(5)
            torch.manual_seed(17)
            candidate = _model(
                auxiliary_projection_activation_override=name
            )
            candidate_next = torch.rand(5)
            self.assertEqual(
                candidate.auxiliary_projection_activation_layer_count, 2
            )
            self.assertEqual(
                _activation_types(candidate), [activation_type, activation_type]
            )
            self.assertEqual(
                sum(value.numel() for value in control.parameters()),
                sum(value.numel() for value in candidate.parameters()),
            )
            self.assertTrue(torch.equal(control_next, candidate_next))
            for key, value in control.state_dict().items():
                self.assertTrue(
                    torch.equal(value, candidate.state_dict()[key]), key
                )

    def test_blockwise_replaces_every_branch_activation(self):
        model = _model(
            auxiliary_projection="blockwise_mlp_gelu",
            auxiliary_block_dims=[4, 6],
            auxiliary_projection_activation_override="silu",
        )
        self.assertEqual(model.auxiliary_projection_activation_layer_count, 4)
        self.assertEqual(_activation_types(model), [nn.SiLU] * 4)

    def test_invalid_and_activation_free_overrides_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be"):
            _model(auxiliary_projection_activation_override="softmax")
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            _model(
                auxiliary_projection="linear",
                auxiliary_projection_activation_override="relu",
            )

    def test_model_kwargs_and_evidence_contract(self):
        self.assertEqual(
            residual_model_kwargs({})[
                "auxiliary_projection_activation_override"
            ],
            "source",
        )
        self.assertEqual(
            residual_model_kwargs({
                "auxiliary_projection_activation_override": "mish"
            })["auxiliary_projection_activation_override"],
            "mish",
        )
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "a00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_projection_activation_87.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_projection_activation_87.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-79%24", gpu)
        self.assertIn("expected 480 per-run CSV files", evidence)
        self.assertIn("expected 480 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
