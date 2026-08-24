import csv

import numpy as np
import torch

from Evaluate_Task3_MainTable import _select_raw_by_validation
from FMT_Utils.PathlineClassifier_3D import PathlineBinaryClassifier3D
from Verify_Task3_FMTResidual import (
    _apply_raw_pca_transform,
    _fit_raw_pca_auxiliary,
    _select_alpha,
    _train_one,
)


def _split(rng, count):
    raw = rng.normal(size=(count, 7, 4, 3)).astype(np.float32)
    unused_fmt = rng.normal(size=(count, 5)).astype(np.float32)
    labels = (rng.random(count) > 0.5).astype(np.float32)
    return raw, unused_fmt, labels


def test_raw_pca_is_train_only_and_reproducible():
    rng = np.random.default_rng(4)
    train = _split(rng, 30)
    validation = _split(rng, 9)
    transformed, state = _fit_raw_pca_auxiliary(
        (train, validation, None), n_components=8, random_state=12
    )
    train_out, validation_out, test_out = transformed
    assert test_out is None
    assert train_out[1].shape == (30, 8)
    assert validation_out[1].shape == (9, 8)
    np.testing.assert_allclose(
        validation_out[1],
        _apply_raw_pca_transform(validation[0], state),
        rtol=1e-5, atol=1e-5,
    )
    np.testing.assert_allclose(train_out[1].mean(axis=0), 0.0, atol=2e-5)
    np.testing.assert_allclose(train_out[1].std(axis=0), 1.0, atol=2e-5)


def test_fixed_nonzero_alpha_is_allowed_for_ap_selection():
    targets = np.asarray([0, 0, 1, 1], dtype=bool)
    raw = np.asarray([-1.0, -0.5, 0.2, 0.4])
    residual = np.asarray([0.0, 0.1, 0.3, 0.5])
    alpha, score = _select_alpha(
        targets, raw, residual, np.asarray([1.0]), "average_precision"
    )
    assert alpha == 1.0
    assert 0.0 <= score <= 1.0


def test_raw_method_selection_uses_family_validation_ap(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    pca_path = tmp_path / "pca.csv"
    baseline_rows = []
    pca_rows = []
    for dataset in ("re160", "re640"):
        for seed in (30, 31):
            baseline_rows.extend([
                {"dataset": dataset, "variant": "raw", "seed": seed,
                 "validation_average_precision": 0.60},
                {"dataset": dataset, "variant": "raw_wide", "seed": seed,
                 "validation_average_precision": 0.62},
            ])
            pca_rows.append(
                {"dataset": dataset, "variant": "raw_pca_residual", "seed": seed,
                 "validation_average_precision": 0.64}
            )
    for path, rows in ((baseline_path, baseline_rows), (pca_path, pca_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    spec = {
        "seeds": [30, 31],
        "families": {"half-cylinder": ["re160", "re640"]},
        "groups": [{
            "datasets": ["re160", "re640"],
            "development_result_csvs": [str(baseline_path), str(pca_path)],
        }],
    }
    selected, rows = _select_raw_by_validation(
        spec, {"re160": "half-cylinder", "re640": "half-cylinder"}
    )
    assert selected == {"half-cylinder": "raw_pca_residual"}
    assert rows[0]["selection_used_confirmation"] == 0


def test_raw_pca_residual_training_smoke(tmp_path):
    rng = np.random.default_rng(9)
    train = _split(rng, 40)
    validation = _split(rng, 16)
    train = (train[0], train[1], np.tile([0.0, 1.0], 20).astype(np.float32))
    validation = (
        validation[0], validation[1],
        np.tile([0.0, 1.0], 8).astype(np.float32),
    )
    stats = {
        "raw_mean": np.zeros((1, 1, 1, 1), dtype=np.float32),
        "raw_std": np.ones((1, 1, 1, 1), dtype=np.float32),
        "fmt_mean": np.zeros((1, 5), dtype=np.float32),
        "fmt_std": np.ones((1, 5), dtype=np.float32),
    }
    checkpoint_dir = tmp_path / "raw" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    raw_model = PathlineBinaryClassifier3D(
        variant="raw", fmt_dim=8, temporal_width=16,
        embedding_dim=32, auxiliary_dim=8,
    )
    torch.save({
        "state_dict": raw_model.state_dict(), "variant": "raw",
        "dataset": "toy", "seed": 30, "threshold": 0.5,
        "best_epoch": 1, "normalization": stats,
        "config": {"model": {
            "temporal_width": 16, "embedding_dim": 32, "auxiliary_dim": 8,
        }},
    }, checkpoint_dir / "toy_raw_seed30.pt")
    spec = {
        "raw_checkpoint_dir": str(checkpoint_dir),
        "auxiliary_source": "raw_pca", "raw_pca_components": 8,
        "raw_pca_random_state": 7, "raw_wide_parameter_count": 10_000_000,
        "model": {"embedding_dim": 32, "auxiliary_dim": 8,
                  "residual_input": "geometry_fmt"},
        "fusion": {"fixed_alpha": 1.0, "selection_metric": "average_precision"},
        "training": {"batch_size": 16, "learning_rate": 1e-3,
                     "weight_decay": 1e-4, "max_epochs": 1,
                     "patience": 1, "min_delta": 0.0},
    }
    row = _train_one(
        spec, "toy", 30, (train, validation, None), stats,
        torch.device("cpu"), tmp_path / "result",
    )
    assert row["variant"] == "raw_pca_residual"
    assert row["validation_alpha"] == 1.0
    saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    assert saved["auxiliary_transform"]["components"].shape == (8, 84)
