from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from Search_Task3_LossOptimization_7_1 import (
    _load_optimization_spec,
    _optimization_candidate,
)
from Verify_Task3_FMTResidual import _build_training_loss


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
