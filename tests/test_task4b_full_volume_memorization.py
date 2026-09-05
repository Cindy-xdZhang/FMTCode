import numpy as np
import torch

from experiments.Verify_Task4B_FullVolumeMemorization_1_1 import (
    _fit_metrics,
    epoch_permutation,
    permutation_certificate,
)


def test_epoch_permutation_is_complete_and_reproducible():
    first = epoch_permutation(101, 7068, 3)
    second = epoch_permutation(101, 7068, 3)
    other_epoch = epoch_permutation(101, 7068, 4)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_epoch)
    certificate = permutation_certificate(first, 101)
    assert certificate["draw_count"] == 101
    assert certificate["unique_count"] == 101
    assert certificate["duplicate_count"] == 0
    assert certificate["missing_count"] == 0


def test_exact_fit_gate_uses_integer_errors_and_positive_margin():
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    logits = np.full((4, 4), -2.0, dtype=np.float64)
    logits[np.arange(4), labels] = 2.0
    metrics = _fit_metrics(labels, logits)
    assert metrics["accuracy"] == 1.0
    assert metrics["error_count"] == 0
    assert metrics["macro_f1"] == 1.0
    assert metrics["minimum_true_logit_margin"] == 4.0


def test_model_family_accepts_frozen_high_capacity_widths():
    from FMT_Utils.Task4B_Classifier_3D import PathlineMulticlassClassifier3D

    model = PathlineMulticlassClassifier3D(
        variant="raw_fmt",
        fmt_dim=161,
        temporal_width=128,
        embedding_dim=1024,
        auxiliary_dim=1024,
        dropout=0.0,
    )
    logits = model(torch.randn(2, 7, 33, 3), torch.randn(2, 161))
    assert logits.shape == (2, 4)

