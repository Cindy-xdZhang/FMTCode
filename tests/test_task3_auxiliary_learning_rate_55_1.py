"""Contracts for Task3 auxiliary-projection learning-rate search 55.1."""

import unittest

import torch

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
)
from Search_Task3_FMTResidual_3D import _candidate_spec
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _optimization_candidate,
    _resolve_combination_candidates,
)
from Verify_Task3_FMTResidual import (
    _auxiliary_learning_rate_multiplier,
    _build_optimizer,
    _optimizer_parameter_spec,
)


CONFIG = "config/Verify_Task3_AuxiliaryLearningRate_55.1.yaml"
EXPECTED = [1.0, 0.05, 0.10, 0.25, 0.50, 2.0, 4.0, 8.0, 16.0]


class AuxiliaryLearningRateTests(unittest.TestCase):
    def test_grid_job_count_and_registered_targets(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["optimization_candidates"]), 9)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 89), ("smokeBuoyancy", 8))
        selection = spec["optimization_selection"]
        self.assertEqual(selection["target_dataset_macro_f1_gain"], 0.200)
        self.assertEqual(selection["target_absolute_fmt_f1"], 0.893)
        self.assertFalse(selection["confirmation_opened"])

    def test_grid_changes_only_projection_learning_rate(self):
        spec = _load_optimization_spec(CONFIG)
        observed = []
        for index, row in enumerate(spec["optimization_candidates"]):
            self.assertEqual(row["sources"], ["feature"])
            if index == 0:
                self.assertEqual(set(row), {"id", "sources"})
                observed.append(1.0)
            else:
                self.assertEqual(set(row), {"id", "sources", "training"})
                self.assertEqual(
                    set(row["training"]),
                    {"auxiliary_learning_rate_multiplier"},
                )
                observed.append(float(
                    row["training"]["auxiliary_learning_rate_multiplier"]
                ))
        self.assertEqual(observed, EXPECTED)

    def test_resolved_candidates_and_both_arm_specs_preserve_multiplier(self):
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
        group = next(
            value for value in spec["groups"].values()
            if dataset in value["datasets"]
        )
        for index, expected in enumerate(EXPECTED):
            candidate = _optimization_candidate(spec, manifest, dataset, index)
            for arm in ("fmt", "raw_pca"):
                run_spec = _candidate_spec(
                    spec, group, candidate, dataset, 40, arm,
                    spec["output_root"], 8,
                )
                self.assertEqual(
                    _auxiliary_learning_rate_multiplier(run_spec["training"]),
                    expected,
                )

    def test_control_keeps_one_historical_optimizer_group(self):
        model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8, head_architecture="deep_mlp", head_hidden_dim=16,
        )
        training = {"learning_rate": 0.001, "weight_decay": 0.0001}
        parameters, multiplier, auxiliary_group = _optimizer_parameter_spec(
            model, training
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(auxiliary_group, 0)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_noncontrol_changes_only_projection_group_rate(self):
        model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=8, head_architecture="deep_mlp", head_hidden_dim=16,
        )
        training = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "auxiliary_learning_rate_multiplier": 4.0,
        }
        parameters, multiplier, auxiliary_group = _optimizer_parameter_spec(
            model, training
        )
        optimizer, _, _ = _build_optimizer(training, parameters)
        self.assertEqual(multiplier, 4.0)
        self.assertEqual(auxiliary_group, 1)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.004)
        auxiliary_ids = {
            id(parameter) for parameter in model.fmt_encoder.parameters()
            if parameter.requires_grad
        }
        observed_auxiliary_ids = {
            id(parameter)
            for parameter in optimizer.param_groups[1]["params"]
        }
        self.assertEqual(observed_auxiliary_ids, auxiliary_ids)
        downstream_ids = {
            id(parameter)
            for parameter in optimizer.param_groups[0]["params"]
        }
        self.assertFalse(downstream_ids.intersection(auxiliary_ids))

    def test_multiplier_validation_and_optimizer_step(self):
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(
                ValueError, "auxiliary_learning_rate_multiplier"
            ):
                _auxiliary_learning_rate_multiplier({
                    "auxiliary_learning_rate_multiplier": invalid
                })
        model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"), fmt_dim=8,
        )
        training = {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "auxiliary_learning_rate_multiplier": 2.0,
        }
        parameters, _, _ = _optimizer_parameter_spec(model, training)
        optimizer, _, _ = _build_optimizer(training, parameters)
        pathlines = torch.randn(4, 7, 8, 3)
        auxiliary = torch.randn(4, 8)
        before = next(model.fmt_encoder.parameters()).detach().clone()
        loss = model(pathlines, auxiliary).square().mean()
        loss.backward()
        optimizer.step()
        after = next(model.fmt_encoder.parameters()).detach()
        self.assertFalse(torch.equal(before, after))

    def test_control_and_zero_tolerance_guard(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            spec["optimization_selection"]["absolute_fmt_guard"],
            {
                "control_optimization_id": "c00_control_auxlr1",
                "f1_tolerance": 0.0,
                "average_precision_tolerance": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
