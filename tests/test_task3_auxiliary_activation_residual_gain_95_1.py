"""Contracts for paired auxiliary activation-residual-gain search 95.1."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from FMT_Utils.PathlineClassifier_3D import (
    _ActivationInputScale,
    _ActivationInputShift,
    _ActivationResidualGain,
    _ActivationResidualMix,
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_AuxiliaryActivationResidualGain_95.1.yaml"
GAINS = [None, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


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


class AuxiliaryActivationResidualGainTests(unittest.TestCase):
    def test_registered_grid_targets_and_job_count(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        self.assertEqual(
            spec["allowed_source_overrides"],
            ["model.auxiliary_projection_activation_residual_gain"],
        )
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.234)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.900)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_registered_residual_gain(self):
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
                    {"auxiliary_projection_activation_residual_gain"},
                )
                observed.append(float(
                    row["model"][
                        "auxiliary_projection_activation_residual_gain"
                    ]
                ))
        self.assertEqual(observed, GAINS)

    def test_source_recipe_is_preserved_except_residual_gain(self):
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
                            "auxiliary_projection_activation_override": "silu",
                            "auxiliary_projection_activation_input_scale": 0.5,
                            "auxiliary_projection_activation_input_shift": -0.25,
                            "auxiliary_projection_activation_residual_mix": 0.25,
                        },
                        "training": {"max_epochs": 160},
                    }, sort_keys=True),
                }
            source_path.write_text(json.dumps({
                "experiment": (
                    "Verify_Task3_AuxiliaryActivationResidualMixPortfolio_94.1"
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
                "auxiliary_projection_activation_residual_gain", rows[0]["model"]
            )
            self.assertEqual(
                rows[1]["model"][
                    "auxiliary_projection_activation_residual_mix"
                ],
                0.25,
            )
            self.assertEqual(
                rows[1]["model"][
                    "auxiliary_projection_activation_residual_gain"
                ],
                0.01,
            )
            self.assertEqual(rows[1]["training"]["max_epochs"], 160)

    def test_default_is_exact_no_op(self):
        torch.manual_seed(7068)
        legacy = _model()
        legacy_next = torch.rand(5)
        torch.manual_seed(7068)
        explicit = _model(auxiliary_projection_activation_residual_gain=0.0)
        explicit_next = torch.rand(5)
        self.assertEqual(
            explicit.auxiliary_projection_activation_residual_gain_layer_count,
            0,
        )
        self.assertFalse(any(
            isinstance(child, _ActivationResidualGain)
            for child in explicit.fmt_encoder.modules()
        ))
        self.assertTrue(torch.equal(legacy_next, explicit_next))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)

    def test_candidate_wraps_only_activations_without_rng_or_capacity_shift(self):
        torch.manual_seed(17)
        control = _model()
        control_next = torch.rand(5)
        torch.manual_seed(17)
        candidate = _model(auxiliary_projection_activation_residual_gain=0.5)
        candidate_next = torch.rand(5)
        wrappers = [
            child for child in candidate.fmt_encoder.modules()
            if isinstance(child, _ActivationResidualGain)
        ]
        self.assertEqual(
            candidate.auxiliary_projection_activation_residual_gain_layer_count,
            2,
        )
        self.assertEqual(len(wrappers), 2)
        self.assertTrue(all(child.gain == 0.5 for child in wrappers))
        self.assertTrue(all(isinstance(child.activation, nn.GELU) for child in wrappers))
        self.assertEqual(
            sum(value.numel() for value in control.parameters()),
            sum(value.numel() for value in candidate.parameters()),
        )
        self.assertTrue(torch.equal(control_next, candidate_next))
        for key, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)

    def test_gain_is_composed_after_scale_shift_and_mix(self):
        model = _model(
            auxiliary_projection_activation_override="identity",
            auxiliary_projection_activation_input_scale=2.0,
            auxiliary_projection_activation_input_shift=0.5,
            auxiliary_projection_activation_residual_mix=0.25,
            auxiliary_projection_activation_residual_gain=0.5,
        )
        scale_wrappers = [
            child for child in model.fmt_encoder.modules()
            if isinstance(child, _ActivationInputScale)
        ]
        shift_wrappers = [
            child for child in model.fmt_encoder.modules()
            if isinstance(child, _ActivationInputShift)
        ]
        gain_wrappers = [
            child for child in model.fmt_encoder.modules()
            if isinstance(child, _ActivationResidualGain)
        ]
        mix_wrappers = [
            child for child in model.fmt_encoder.modules()
            if isinstance(child, _ActivationResidualMix)
        ]
        self.assertEqual(len(scale_wrappers), 2)
        self.assertEqual(len(shift_wrappers), 2)
        self.assertEqual(len(gain_wrappers), 2)
        self.assertEqual(len(mix_wrappers), 2)
        self.assertTrue(all(
            isinstance(child.activation, _ActivationInputShift)
            for child in scale_wrappers
        ))
        self.assertTrue(all(
            isinstance(child.activation, _ActivationResidualGain)
            for child in shift_wrappers
        ))
        self.assertTrue(all(
            isinstance(child.activation, _ActivationResidualMix)
            for child in gain_wrappers
        ))
        self.assertTrue(all(
            isinstance(child.activation, nn.Identity)
            for child in mix_wrappers
        ))

    def test_residual_gain_numeric_definition(self):
        wrapper = _ActivationResidualGain(nn.ReLU(), 0.25)
        values = torch.tensor([-2.0, 3.0])
        self.assertTrue(torch.equal(wrapper(values), torch.tensor([-0.5, 3.75])))
        composed = _ActivationResidualGain(
            _ActivationResidualMix(nn.ReLU(), 0.5), 0.25
        )
        self.assertTrue(torch.equal(
            composed(values), torch.tensor([-1.5, 3.75])
        ))

    def test_blockwise_wraps_every_branch_activation(self):
        model = _model(
            auxiliary_projection="blockwise_mlp_gelu",
            auxiliary_block_dims=[4, 6],
            auxiliary_projection_activation_residual_gain=0.25,
        )
        self.assertEqual(
            model.auxiliary_projection_activation_residual_gain_layer_count, 4
        )

    def test_invalid_and_activation_free_gains_are_rejected(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "non-negative"):
                _model(auxiliary_projection_activation_residual_gain=value)
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            _model(
                auxiliary_projection="linear",
                auxiliary_projection_activation_residual_gain=0.5,
            )

    def test_model_kwargs_and_evidence_contract(self):
        self.assertEqual(
            residual_model_kwargs({})[
                "auxiliary_projection_activation_residual_gain"
            ],
            0.0,
        )
        self.assertEqual(
            residual_model_kwargs({
                "auxiliary_projection_activation_residual_gain": 0.25
            })["auxiliary_projection_activation_residual_gain"],
            0.25,
        )
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "g00_control_source",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )
        gpu = Path(
            "ibex_bash/verify_task3_auxiliary_activation_residual_gain_95.1_gpu.sh"
        ).read_text(encoding="utf-8")
        evidence = Path(
            "ibex_bash/verify_task3_auxiliary_activation_residual_gain_95.1_evidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-89%24", gpu)
        self.assertIn("expected 540 per-run CSV files", evidence)
        self.assertIn("expected 540 temporary checkpoints", evidence)
        self.assertNotIn("-delete", evidence)


if __name__ == "__main__":
    unittest.main()
