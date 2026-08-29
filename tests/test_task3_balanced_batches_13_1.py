from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from Verify_Task3_FMTClassifier import (
    _ExactClassBalancedBatchSampler,
    _loader,
)
from Verify_Task3_FMTResidual import _build_training_loss
from Search_Task3_LossOptimization_7_1 import _load_optimization_spec


CONFIG = Path("config/Verify_Task3_ClassBalancedBatches_13.1.yaml")


def _split(labels, fmt_shift=0.0):
    labels = np.asarray(labels, dtype=np.float32)
    index = np.arange(len(labels), dtype=np.float32)
    raw = index[:, None, None, None]
    fmt = (index + float(fmt_shift))[:, None]
    return raw, fmt, labels


def _epoch_batches(sampler):
    return [list(batch) for batch in sampler]


class Task3BalancedBatchesTests(unittest.TestCase):
    def test_exact_class_composition_and_reproducible_epoch_sequence(self):
        labels = np.asarray([0] * 17 + [1] * 3)
        first = _ExactClassBalancedBatchSampler(labels, 8, 0.25, 7068)
        second = _ExactClassBalancedBatchSampler(labels, 8, 0.25, 7068)
        first_epoch = _epoch_batches(first)
        second_epoch = _epoch_batches(second)
        self.assertEqual(first_epoch, second_epoch)
        self.assertEqual(len(first_epoch), 3)
        self.assertTrue(all(len(batch) == 8 for batch in first_epoch))
        self.assertTrue(
            all(int(labels[batch].sum()) == 2 for batch in first_epoch)
        )
        self.assertEqual(_epoch_batches(first), _epoch_batches(second))
        self.assertNotEqual(
            _epoch_batches(
                _ExactClassBalancedBatchSampler(labels, 8, 0.25, 7069)
            ),
            first_epoch,
        )

    def test_paired_arms_receive_identical_training_indices(self):
        labels = np.asarray([0] * 13 + [1] * 5)
        fmt_loader = _loader(
            _split(labels, 0.0), 6, True, 40, False,
            positive_fraction=1.0 / 3.0,
        )
        pca_loader = _loader(
            _split(labels, 1000.0), 6, True, 40, False,
            positive_fraction=1.0 / 3.0,
        )
        fmt_indices = [raw[:, 0, 0, 0].tolist() for raw, _, _ in fmt_loader]
        pca_indices = [raw[:, 0, 0, 0].tolist() for raw, _, _ in pca_loader]
        self.assertEqual(fmt_indices, pca_indices)

    def test_balanced_sampler_is_training_only_and_legacy_loader_is_unchanged(self):
        labels = np.asarray([0, 0, 0, 1], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "only for training"):
            _loader(
                _split(labels), 4, False, 7, False,
                positive_fraction=0.5,
            )
        first = _loader(_split(labels), 2, True, 9, False)
        second = _loader(_split(labels), 2, True, 9, False)
        self.assertEqual(
            [batch[0].tolist() for batch in first],
            [batch[0].tolist() for batch in second],
        )

    def test_sampled_prevalence_adjusts_positive_weight_without_double_weighting(self):
        _, metadata = _build_training_loss(
            {"positive_weight_scale": 1.0},
            positive=10.0,
            negative=90.0,
            device=torch.device("cpu"),
            sampled_positive_fraction=0.25,
        )
        self.assertAlmostEqual(metadata["positive_weight"], 3.0)
        self.assertAlmostEqual(0.25 * metadata["positive_weight"], 0.75)
        self.assertAlmostEqual(
            1.0 - metadata["sampled_positive_fraction"], 0.75
        )
        _, legacy = _build_training_loss(
            {}, 10.0, 90.0, torch.device("cpu")
        )
        self.assertAlmostEqual(legacy["positive_weight"], 9.0)
        self.assertIsNone(legacy["sampled_positive_fraction"])

    def test_sampler_rejects_invalid_positive_fraction(self):
        for fraction in (0.0, 1.0, -0.1, np.nan):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "positive_fraction"):
                    _ExactClassBalancedBatchSampler(
                        [0, 1], 2, fraction, 1
                    )

    def test_13_1_grid_and_confirmation_boundary_are_frozen(self):
        overlay = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            overlay["experiment"],
            "Verify_Task3_ClassBalancedBatches_13.1",
        )
        self.assertEqual(len(spec["datasets"]), 10)
        self.assertEqual(len(spec["optimization_candidates"]), 10)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertFalse(
            spec["optimization_selection"]["confirmation_opened"]
        )
        self.assertEqual(
            len(spec["datasets"])
            * len(spec["optimization_candidates"])
            * len(spec["paired_seeds"])
            * 2,
            600,
        )


if __name__ == "__main__":
    unittest.main()
