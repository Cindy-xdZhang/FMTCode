from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
)
from Search_Task3_NetworkArchitecture_6_1 import (
    _architecture_candidate,
    _decode_job,
    _load_architecture_spec,
)


CONFIG = "config/Verify_Task3_NetworkArchitecture_6.1.yaml"


def test_grid_and_job_mapping_are_frozen():
    spec = _load_architecture_spec(CONFIG)
    assert len(spec["datasets"]) == 10
    assert len(spec["architectures"]) == 7
    assert _decode_job(spec, 0) == ("channel", 0)
    assert _decode_job(spec, 69) == ("smokeBuoyancy", 6)
    assert spec["paired_seeds"] == [40, 41, 42]
    assert spec["architecture_selection"]["confirmation_opened"] is False
    assert (
        len(spec["datasets"]) * len(spec["architectures"])
        * len(spec["paired_seeds"]) * 2
    ) == 420


def test_all_architectures_preserve_raw_at_alpha_zero_and_fit_budget():
    torch.manual_seed(7068)
    pathlines = torch.randn(3, 7, 32, 3)
    auxiliary = torch.randn(3, 173)
    for index, architecture in enumerate(
        _load_architecture_spec(CONFIG)["architectures"]
    ):
        raw = PathlineBinaryClassifier3D("raw", fmt_dim=173).eval()
        model = PathlineFMTResidualClassifier3D(
            raw, fmt_dim=173,
            **residual_model_kwargs({
                "embedding_dim": 128,
                "auxiliary_dim": 64,
                "residual_input": "geometry_fmt",
                **architecture,
            }),
        ).eval()
        assert torch.equal(model(pathlines, auxiliary, alpha=0.0), raw(pathlines))
        assert model(pathlines, auxiliary).shape == (3,)
        assert sum(parameter.numel() for parameter in model.parameters()) < 148225


def test_default_historical_mlp_checkpoint_contract_is_unchanged():
    raw_a = PathlineBinaryClassifier3D("raw", fmt_dim=161)
    raw_b = PathlineBinaryClassifier3D("raw", fmt_dim=161)
    old_default = PathlineFMTResidualClassifier3D(raw_a, fmt_dim=161)
    explicit = PathlineFMTResidualClassifier3D(
        raw_b, fmt_dim=161, head_architecture="mlp",
        head_hidden_dim=128, head_depth=2,
    )
    assert list(old_default.state_dict()) == list(explicit.state_dict())
    assert sum(parameter.numel() for parameter in old_default.parameters()) == 125506


if __name__ == "__main__":
    test_grid_and_job_mapping_are_frozen()
    test_all_architectures_preserve_raw_at_alpha_zero_and_fit_budget()
    test_default_historical_mlp_checkpoint_contract_is_unchanged()
    print("TASK3 NETWORK ARCHITECTURE 6.1 TEST PASSED")
