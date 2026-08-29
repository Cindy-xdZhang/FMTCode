"""Contracts for Task3 auxiliary deep supervision 26.1."""

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
)
from Verify_Task3_FMTResidual import (
    _auxiliary_supervision_loss,
    _build_training_loss,
)


CONFIG = "config/Verify_Task3_AuxiliaryDeepSupervision_26.1.yaml"


class AuxiliaryDeepSupervisionTests(unittest.TestCase):
    def test_preregistered_grid_and_targets(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        self.assertEqual(len(candidates), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        self.assertEqual(
            spec["optimization_selection"]["target_dataset_macro_f1_gain"],
            0.185,
        )
        self.assertEqual(
            spec["optimization_selection"]["target_absolute_fmt_f1"],
            0.892,
        )
        guard = spec["optimization_selection"]["absolute_fmt_guard"]
        self.assertEqual(guard["f1_tolerance"], 0.0)
        self.assertEqual(guard["average_precision_tolerance"], 0.0)
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])

    def test_grid_is_exact_two_by_five_plus_control(self):
        spec = _load_optimization_spec(CONFIG)
        candidates = spec["optimization_candidates"]
        self.assertEqual(candidates[0]["id"], "a00_control")
        observed = {
            (
                row["model"]["auxiliary_classifier_architecture"],
                float(row["training"]["auxiliary_supervision_loss_weight"]),
            )
            for row in candidates[1:]
        }
        expected = {
            (architecture, weight)
            for architecture in ("linear", "mlp")
            for weight in (0.01, 0.03, 0.10, 0.30, 1.00)
        }
        self.assertEqual(observed, expected)

    def test_auxiliary_head_does_not_change_inference_initialization(self):
        torch.manual_seed(17)
        control = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=12,
            auxiliary_dim=8,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
        )
        torch.manual_seed(17)
        supervised = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=12,
            auxiliary_dim=8,
            head_architecture="deep_mlp",
            head_hidden_dim=16,
            auxiliary_classifier_architecture="linear",
        )
        supervised_state = supervised.state_dict()
        for key, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, supervised_state[key]), key)

    def test_direct_loss_reaches_projection_but_not_frozen_raw(self):
        torch.manual_seed(3)
        model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"),
            fmt_dim=10,
            auxiliary_dim=6,
            auxiliary_classifier_architecture="mlp",
            auxiliary_classifier_hidden_dim=7,
        )
        pathlines = torch.randn(12, 7, 32, 3)
        features = torch.randn(12, 10)
        labels = torch.tensor([0.0, 1.0] * 6)
        _, _, auxiliary = model.forward_components(
            pathlines, features, return_auxiliary=True
        )
        training = {"auxiliary_supervision_loss_weight": 0.3}
        criterion, _ = _build_training_loss(
            training, positive=6.0, negative=6.0, device=torch.device("cpu")
        )
        loss = _auxiliary_supervision_loss(
            model, auxiliary, labels, training, criterion
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        projection_gradients = [
            parameter.grad for parameter in model.fmt_encoder.parameters()
        ]
        classifier_gradients = [
            parameter.grad for parameter in model.auxiliary_classifier.parameters()
        ]
        self.assertTrue(any(
            gradient is not None and bool(torch.any(gradient != 0.0))
            for gradient in projection_gradients
        ))
        self.assertTrue(any(
            gradient is not None and bool(torch.any(gradient != 0.0))
            for gradient in classifier_gradients
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in model.raw_model.parameters()
        ))

    def test_supervision_weight_and_classifier_must_be_consistent(self):
        model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"), fmt_dim=4
        )
        criterion, _ = _build_training_loss(
            {"auxiliary_supervision_loss_weight": 0.1},
            positive=2.0,
            negative=2.0,
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(RuntimeError, "configured auxiliary"):
            _auxiliary_supervision_loss(
                model,
                torch.randn(4, 64),
                torch.tensor([0.0, 1.0, 0.0, 1.0]),
                {"auxiliary_supervision_loss_weight": 0.1},
                criterion,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _build_training_loss(
                {"auxiliary_supervision_loss_weight": -0.1},
                positive=2.0,
                negative=2.0,
                device=torch.device("cpu"),
            )

    def test_paired_arms_have_identical_trainable_capacity(self):
        kwargs = residual_model_kwargs({
            "embedding_dim": 128,
            "auxiliary_dim": 8,
            "head_architecture": "deep_mlp",
            "head_hidden_dim": 64,
            "head_depth": 2,
            "auxiliary_classifier_architecture": "mlp",
            "auxiliary_classifier_hidden_dim": 32,
        })
        fmt_model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"), fmt_dim=10, **kwargs
        )
        raw_pca_model = PathlineFMTResidualClassifier3D(
            PathlineBinaryClassifier3D(variant="raw"), fmt_dim=10, **kwargs
        )
        self.assertEqual(
            trainable_parameter_count(fmt_model),
            trainable_parameter_count(raw_pca_model),
        )


if __name__ == "__main__":
    unittest.main()
