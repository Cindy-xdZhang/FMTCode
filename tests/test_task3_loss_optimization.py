import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from Search_Task3_LossOptimization_7_1 import (
    _canonical_text_sha256,
    _is_numerical_instability,
    _load_numerical_instability,
    _load_optimization_spec,
    _optimization_candidate,
    _write_numerical_instability,
)
from Verify_Task3_FMTResidual import (
    _build_training_loss,
    _raw_hardness_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "Verify_Task3_LossOptimization_7.1.yaml"


def test_weighted_bce_default_matches_torch_reference():
    logits = torch.tensor([-2.0, -0.5, 0.3, 1.7], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    criterion, metadata = _build_training_loss(
        {}, positive=2.0, negative=6.0, device=torch.device("cpu")
    )
    observed = criterion(logits, targets)
    expected = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(3.0)
    )
    assert torch.equal(observed, expected)
    assert metadata == {
        "loss": "weighted_bce",
        "positive_weight_scale": 1.0,
        "positive_weight": 3.0,
        "focal_gamma": 0.0,
        "raw_hardness_scale": 0.0,
        "raw_hardness_power": 1.0,
        "raw_hardness_temperature": 1.0,
        "raw_error_boost": 0.0,
    }


def test_focal_loss_is_finite_and_backpropagates():
    logits = torch.tensor([-3.0, -0.2, 0.4, 2.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    criterion, metadata = _build_training_loss(
        {"loss": "focal", "focal_gamma": 2.0,
         "positive_weight_scale": 0.75},
        positive=2.0, negative=6.0, device=torch.device("cpu"),
    )
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert metadata["positive_weight"] == pytest.approx(2.25)


def test_fractional_focal_gamma_is_stable_for_saturated_correct_logits():
    logits = torch.tensor([-100.0, 100.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0])
    criterion, _ = _build_training_loss(
        {"loss": "focal", "focal_gamma": 0.5},
        positive=1.0, negative=1.0, device=torch.device("cpu"),
    )
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_log_domain_focal_matches_direct_formula_away_from_saturation():
    logits = torch.tensor([-2.0, -0.3, 0.4, 1.7])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    criterion, metadata = _build_training_loss(
        {"loss": "focal", "focal_gamma": 0.5},
        positive=2.0, negative=6.0, device=torch.device("cpu"),
    )
    observed = criterion(logits, targets)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor(metadata["positive_weight"]),
        reduction="none",
    )
    probability = torch.sigmoid(logits)
    probability_of_target = (
        targets * probability + (1.0 - targets) * (1.0 - probability)
    )
    expected = (bce * (1.0 - probability_of_target).pow(0.5)).mean()
    assert torch.allclose(observed, expected, rtol=1e-6, atol=1e-7)


def test_raw_hardness_weights_are_mean_one_detached_and_monotone():
    raw_logits = torch.tensor([4.0, 0.0, -4.0], requires_grad=True)
    targets = torch.ones(3)
    weights = _raw_hardness_weights(
        raw_logits,
        targets,
        {"raw_hardness_scale": 4.0, "raw_hardness_power": 1.0},
    )
    assert weights.requires_grad is False
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]


def test_raw_hardness_control_returns_none_and_preserves_loss():
    raw_logits = torch.tensor([2.0, -1.0])
    targets = torch.tensor([1.0, 0.0])
    assert _raw_hardness_weights(raw_logits, targets, {}) is None
    criterion, _ = _build_training_loss(
        {}, positive=1.0, negative=1.0, device=torch.device("cpu")
    )
    assert torch.equal(
        criterion(raw_logits, targets, sample_weights=None),
        F.binary_cross_entropy_with_logits(raw_logits, targets),
    )


def test_weighted_loss_matches_manual_mean_and_error_boost_is_paired():
    raw_logits = torch.tensor([2.0, -1.0, -2.0, 1.0])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    training = {
        "raw_hardness_scale": 2.0,
        "raw_hardness_power": 2.0,
        "raw_hardness_temperature": 0.5,
        "raw_error_boost": 3.0,
    }
    first = _raw_hardness_weights(raw_logits, targets, training)
    second = _raw_hardness_weights(raw_logits, targets, training)
    assert torch.equal(first, second)
    criterion, _ = _build_training_loss(
        training, positive=2.0, negative=2.0, device=torch.device("cpu")
    )
    observed = criterion(raw_logits, targets, sample_weights=first)
    expected = (
        F.binary_cross_entropy_with_logits(
            raw_logits, targets, reduction="none"
        ) * first
    ).mean()
    assert torch.allclose(observed, expected)


@pytest.mark.parametrize(
    "training,match",
    [
        ({"raw_hardness_scale": -1.0}, "raw_hardness_scale"),
        ({"raw_hardness_power": 0.0}, "raw_hardness_power"),
        ({"raw_hardness_temperature": 0.0}, "raw_hardness_temperature"),
        ({"raw_error_boost": -1.0}, "raw_error_boost"),
    ],
)
def test_raw_hardness_rejects_invalid_hyperparameters(training, match):
    with pytest.raises(ValueError, match=match):
        _build_training_loss(
            training, positive=1.0, negative=1.0,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf")])
def test_loss_rejects_invalid_positive_weight_scale(scale):
    with pytest.raises(ValueError, match="positive_weight_scale"):
        _build_training_loss(
            {"positive_weight_scale": scale},
            positive=1.0, negative=1.0, device=torch.device("cpu"),
        )


def test_optimization_config_is_paired_and_does_not_open_confirmation():
    spec = _load_optimization_spec(CONFIG)
    assert len(spec["datasets"]) == 10
    assert len(spec["optimization_candidates"]) == 25
    assert spec["paired_seeds"] == [40, 41, 42]
    assert spec["optimization_selection"]["confirmation_opened"] is False
    assert spec["model_override"]["head_architecture"] == "deep_mlp"


def test_base_config_hash_is_independent_of_newline_style(tmp_path):
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    lf.write_bytes(b"alpha: 1\nbeta: 2\n")
    crlf.write_bytes(b"alpha: 1\r\nbeta: 2\r\n")
    assert _canonical_text_sha256(lf) == _canonical_text_sha256(crlf)


def test_recipe_merges_upstream_training_and_model_without_arm_specific_keys():
    spec = _load_optimization_spec(CONFIG)
    group = next(
        name for name, row in spec["groups"].items()
        if "channel" in row["datasets"]
    )
    manifest = {
        "base_candidate_by_group": {
            group: {
                "id": "old",
                "fmt_feature": "aivd1w3",
                "upstream_candidate_id": "upstream",
                "training": {"learning_rate": 0.0003},
                **spec["model_override"],
            }
        }
    }
    candidate = _optimization_candidate(spec, manifest, "channel", 3)
    assert candidate["id"] == "o03_posweight075"
    assert candidate["training"]["learning_rate"] == pytest.approx(0.0003)
    assert candidate["training"]["positive_weight_scale"] == pytest.approx(0.75)
    assert candidate["head_architecture"] == "deep_mlp"
    assert "auxiliary_source" not in candidate


@pytest.mark.parametrize(
    "error",
    [
        FloatingPointError("non-finite training gradient"),
        ValueError("Input contains NaN."),
        ValueError("array contains inf"),
        ValueError("Input contains infinity"),
    ],
)
def test_numerical_instability_errors_are_explicitly_classified(error):
    assert _is_numerical_instability(error)


def test_unrelated_value_error_is_not_silently_downgraded():
    assert not _is_numerical_instability(ValueError("wrong tensor shape"))


def test_numerical_instability_marker_is_hash_bound_and_idempotent(tmp_path):
    manifest_path = tmp_path / "preflight_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    spec = {
        "output_root": str(tmp_path),
        "optimization_config_sha256": "optimization-hash",
    }
    manifest = {"upstream_selection_sha256": "selection-hash"}
    candidate = {"id": "o07_focal_g05"}

    marker = _write_numerical_instability(
        spec, manifest, candidate, "cylinder3d", 40, "fmt",
        FloatingPointError("non-finite residual validation logits"),
    )
    first_text = marker.read_text(encoding="utf-8")
    loaded = _load_numerical_instability(
        spec, manifest, candidate, "cylinder3d"
    )
    assert loaded["status"] == "invalid_numerical_instability"
    assert loaded["failed_seed"] == 40
    assert loaded["failed_source"] == "fmt"

    repeated = _write_numerical_instability(
        spec, manifest, candidate, "cylinder3d", 41, "raw_pca",
        FloatingPointError("second failure must not rewrite evidence"),
    )
    assert repeated == marker
    assert marker.read_text(encoding="utf-8") == first_text

    payload = json.loads(first_text)
    payload["dataset"] = "tampered"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker changed"):
        _load_numerical_instability(
            spec, manifest, candidate, "cylinder3d"
        )
