import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D, PathlineFMTResidualClassifier3D,
    trainable_parameter_count,
)
from Verify_Task3_FMTResidual import _select_alpha


ROOT = Path(__file__).resolve().parents[1]


def test_residual_preserves_raw_and_stays_below_capacity_control():
    torch.manual_seed(3)
    raw_model = PathlineBinaryClassifier3D("raw", fmt_dim=161).eval()
    residual = PathlineFMTResidualClassifier3D(raw_model, fmt_dim=161).eval()
    pathlines = torch.randn(5, 7, 32, 3)
    fmt = torch.randn(5, 161)
    raw_logit = raw_model(pathlines)
    assert torch.equal(residual(pathlines, fmt, alpha=0.0), raw_logit)
    total = sum(parameter.numel() for parameter in residual.parameters())
    trainable = trainable_parameter_count(residual)
    raw_wide = trainable_parameter_count(
        PathlineBinaryClassifier3D("raw_wide", fmt_dim=161)
    )
    assert total == 125506
    assert trainable == 35585
    assert total < raw_wide == 148225


def test_all_flow_configs_share_frozen_protocol():
    baseline = yaml.safe_load((
        ROOT / "config" / "Confirm_Task3UniversalityBaselines_1.1.yaml"
    ).read_text(encoding="utf-8"))
    residual = yaml.safe_load((
        ROOT / "config" / "Confirm_Task3UniversalityResidual_1.1.yaml"
    ).read_text(encoding="utf-8"))
    expected = [
        "cylinder3d", "halfcylinderRe640", "halfcylinderRe6400",
        "tangaroa", "deltaWing_resampled", "deltaWing_LBM",
        "f22raptor", "channel",
    ]
    assert baseline["datasets"] == residual["datasets"] == expected
    assert baseline["split"] == residual["split"] == {
        "train_ordinals": [0, 1, 2, 3, 4, 5],
        "validation_ordinals": [6, 7],
        "test_ordinals": [8, 9],
    }
    assert baseline["training"]["seeds"] == residual["training"]["seeds"] == [21, 22]
    assert residual["fusion"]["alpha_min"] == 0.0
    assert residual["fmt_subset"] == "all"


def test_time_local_gram_concat_stays_below_raw_wide_capacity():
    raw_model = PathlineBinaryClassifier3D("raw", fmt_dim=392).eval()
    residual = PathlineFMTResidualClassifier3D(raw_model, fmt_dim=392).eval()
    total = sum(parameter.numel() for parameter in residual.parameters())
    raw_wide = trainable_parameter_count(
        PathlineBinaryClassifier3D("raw_wide", fmt_dim=392)
    )
    assert total == 140290
    assert total < raw_wide == 148225

    fmt_only = PathlineFMTResidualClassifier3D(
        raw_model, fmt_dim=224, residual_input="fmt_only"
    ).eval()
    pathlines = torch.randn(4, 7, 32, 3)
    fmt = torch.randn(4, 224)
    raw_logit = raw_model(pathlines)
    assert torch.equal(fmt_only(pathlines, fmt, alpha=0.0), raw_logit)
    assert trainable_parameter_count(fmt_only) < trainable_parameter_count(residual)

    dual = PathlineFMTResidualClassifier3D(
        raw_model, fmt_dim=308, residual_input="dual"
    ).eval()
    assert trainable_parameter_count(dual) < raw_wide
    assert torch.equal(
        dual(pathlines, torch.randn(4, 308), alpha=0.0), raw_logit
    )
    kinematic_dual = PathlineFMTResidualClassifier3D(
        raw_model, fmt_dim=268, residual_input="dual"
    ).eval()
    assert trainable_parameter_count(kinematic_dual) < raw_wide


def test_minimum_gain_alpha_balances_both_metrics():
    targets = np.asarray([0, 0, 1, 1], dtype=bool)
    raw = np.asarray([-1.0, -0.5, 0.2, 0.4])
    residual = np.asarray([-0.2, -0.1, 0.3, 0.5])
    alpha, score = _select_alpha(
        targets, raw, residual, [0.0, 1.0], objective="minimum_gain",
        baseline_metrics={"f1": 0.5, "average_precision": 0.5},
    )
    assert alpha in {0.0, 1.0}
    assert np.isfinite(score)
    alpha, score = _select_alpha(
        targets, raw, residual, [0.0, 1.0],
        objective="constrained_average_precision",
        baseline_metrics={"f1": 0.5, "average_precision": 0.5},
        minimum_f1_gain=0.0,
    )
    assert alpha in {0.0, 1.0}
    assert score > 1.0


if __name__ == "__main__":
    test_residual_preserves_raw_and_stays_below_capacity_control()
    test_all_flow_configs_share_frozen_protocol()
    test_time_local_gram_concat_stays_below_raw_wide_capacity()
    test_minimum_gain_alpha_balances_both_metrics()
    print("TASK3 UNIVERSALITY RESIDUAL TEST PASSED")
