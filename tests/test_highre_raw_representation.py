import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.RawPathline_3D import (
    normalize_raw_train_eval, raw_pathline_representation,
    raw_representation_group_split,
)


def test_raw_representations_have_expected_shapes_and_are_translation_invariant():
    rng = np.random.default_rng(7068)
    pathlines = rng.normal(size=(5, 7, 32, 3)).astype(np.float32)
    shifted = pathlines + np.array([10.0, -2.0, 5.0], dtype=np.float32)
    local = (pathlines - pathlines[:, :1, :1]).reshape(5, -1)
    shifted_local = (shifted - shifted[:, :1, :1]).reshape(5, -1)
    expected = {
        "positions": 672, "center": 96, "center_delta": 93, "relative": 576,
        "relative_delta": 558, "dynamics": 651, "relative_distance": 192,
        "relative_distance_delta": 186, "pair_distance": 672,
        "pair_distance_delta": 651, "invariant_dynamics": 682,
        "center_relative": 672, "center_delta_relative": 669,
        "center_relative_delta": 654,
    }
    for name, width in expected.items():
        value = raw_pathline_representation(local, name)
        shifted_value = raw_pathline_representation(shifted_local, name)
        assert value.shape == (5, width)
        assert np.allclose(value, shifted_value, atol=1e-5)

    assert raw_representation_group_split("center_relative", 32) == 96
    assert raw_representation_group_split("dynamics", 32) == 93
    assert raw_representation_group_split("pair_distance", 32) == 192
    train = raw_pathline_representation(local[:3], "center_relative")
    evaluate = raw_pathline_representation(local[3:], "center_relative")
    train_n, evaluate_n = normalize_raw_train_eval(
        train, evaluate, "center_relative", 32, "pre_group_rms"
    )
    assert train_n.shape == train.shape and evaluate_n.shape == evaluate.shape
    assert np.isfinite(train_n).all() and np.isfinite(evaluate_n).all()


if __name__ == "__main__":
    test_raw_representations_have_expected_shapes_and_are_translation_invariant()
    print("HIGH-RE RAW REPRESENTATION TEST PASSED")
