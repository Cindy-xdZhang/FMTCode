from pathlib import Path
import unittest

import numpy as np
import torch

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
)
from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
)
from Verify_Task3_FMTResidual import (
    _build_training_loss,
    _supervised_contrastive_loss,
)


CONFIG = Path("config/Verify_Task3_SupervisedContrastive_18.1.yaml")


class Task3SupervisedContrastiveTests(unittest.TestCase):
    def test_default_forward_contract_is_unchanged(self):
        torch.manual_seed(7)
        raw = PathlineBinaryClassifier3D(variant="raw")
        model = PathlineFMTResidualClassifier3D(raw, fmt_dim=6)
        pathlines = torch.randn(3, 5, 8, 3)
        features = torch.randn(3, 6)
        legacy = model.forward_components(pathlines, features)
        extended = model.forward_components(
            pathlines, features, return_auxiliary=True
        )
        self.assertEqual(len(legacy), 2)
        self.assertEqual(len(extended), 3)
        self.assertTrue(torch.equal(legacy[0], extended[0]))
        self.assertTrue(torch.equal(legacy[1], extended[1]))
        self.assertEqual(tuple(extended[2].shape), (3, 64))

    def test_zero_weight_is_exact_noop_with_zero_gradient(self):
        embeddings = torch.randn(4, 5, requires_grad=True)
        loss = _supervised_contrastive_loss(
            embeddings, torch.tensor([0.0, 0.0, 1.0, 1.0]), {}
        )
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(
            embeddings.grad, torch.zeros_like(embeddings)
        ))

    def test_class_separated_embeddings_have_lower_loss(self):
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        recipe = {
            "supervised_contrastive_loss_weight": 1.0,
            "supervised_contrastive_temperature": 0.1,
        }
        separated = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]
        ], requires_grad=True)
        mixed = torch.tensor([
            [1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]
        ], requires_grad=True)
        separated_loss = _supervised_contrastive_loss(
            separated, labels, recipe
        )
        mixed_loss = _supervised_contrastive_loss(mixed, labels, recipe)
        mixed_loss.backward()
        self.assertLess(float(separated_loss), float(mixed_loss))
        self.assertTrue(torch.isfinite(mixed.grad).all())

    def test_one_class_or_unpaired_batches_are_exact_noops(self):
        recipe = {"supervised_contrastive_loss_weight": 1.0}
        one_class = torch.randn(3, 4, requires_grad=True)
        unpaired = torch.randn(2, 4, requires_grad=True)
        loss = _supervised_contrastive_loss(
            one_class, torch.zeros(3), recipe
        ) + _supervised_contrastive_loss(
            unpaired, torch.tensor([0.0, 1.0]), recipe
        )
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(
            one_class.grad, torch.zeros_like(one_class)
        ))
        self.assertTrue(torch.equal(
            unpaired.grad, torch.zeros_like(unpaired)
        ))

    def test_sample_weight_normalization_is_scale_invariant(self):
        embeddings = torch.tensor([
            [1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2]
        ])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        recipe = {"supervised_contrastive_loss_weight": 0.1}
        first = _supervised_contrastive_loss(
            embeddings, labels, recipe, torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
        second = _supervised_contrastive_loss(
            embeddings, labels, recipe,
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
        )
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_invalid_hyperparameters_are_rejected(self):
        for recipe in (
            {"supervised_contrastive_loss_weight": -0.1},
            {"supervised_contrastive_loss_weight": np.nan},
            {"supervised_contrastive_temperature": 0.0},
            {"supervised_contrastive_temperature": np.inf},
        ):
            with self.subTest(recipe=recipe):
                with self.assertRaises(ValueError):
                    _build_training_loss(
                        recipe, 10.0, 90.0, torch.device("cpu")
                    )

    def test_18_1_grid_and_array_bounds_are_frozen(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(len(spec["datasets"]), 10)
        self.assertEqual(len(spec["optimization_candidates"]), 11)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])
        self.assertEqual(_decode_job(spec, 109), ("smokeBuoyancy", 10))
        with self.assertRaises(IndexError):
            _decode_job(spec, 110)


if __name__ == "__main__":
    unittest.main()
