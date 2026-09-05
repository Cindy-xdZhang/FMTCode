import torch
import numpy as np

from FMT_Utils.Task4B_Classifier_3D import (
    PathlineMulticlassClassifier3D,
    trainable_parameter_count,
)
from Train_Task4B_FourClassClassifier_1_1 import (
    ExactBalancedMulticlassBatchSampler,
    _validate_cache_sha256,
)


def test_all_task4b_variants_return_four_logits():
    pathlines = torch.randn(5, 7, 33, 3)
    fmt = torch.randn(5, 161)
    counts = {}
    for variant in ("raw", "raw_wide", "fmt_only", "raw_fmt"):
        model = PathlineMulticlassClassifier3D(
            variant=variant,
            fmt_dim=161,
            temporal_width=32,
            embedding_dim=64,
            auxiliary_dim=32,
        )
        logits = model(
            pathlines,
            fmt if variant in {"fmt_only", "raw_fmt"} else None,
        )
        assert logits.shape == (5, 4)
        counts[variant] = trainable_parameter_count(model)
    assert counts["raw_wide"] > counts["raw_fmt"] > counts["raw"]


def test_balanced_sampler_emits_equal_four_class_quota():
    labels = np.repeat(np.arange(4), [3, 5, 7, 9])
    sampler = ExactBalancedMulticlassBatchSampler(
        labels, batch_size=8, num_classes=4, seed=7068
    )
    first = np.asarray(next(iter(sampler)), dtype=np.int64)
    assert np.bincount(labels[first], minlength=4).tolist() == [2, 2, 2, 2]


def test_cache_sha256_is_checked_before_training(tmp_path):
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"frozen-task4b-cache")
    expected = "4609909716bac105a1a155136ae764e16af786686bed43c313ef3758e9da885e"
    assert _validate_cache_sha256(cache, expected) == expected
    try:
        _validate_cache_sha256(cache, "0" * 64)
    except RuntimeError as error:
        assert "differs from the frozen config" in str(error)
    else:
        raise AssertionError("a mismatched Task4-b cache digest was accepted")
