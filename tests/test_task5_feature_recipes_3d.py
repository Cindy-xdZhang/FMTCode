import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.DFT_FMT_3D import pathline_velocity_gradient_dft_features_3d
from FMT_Utils.Task5FeatureRecipes_3D import (
    parse_task5_feature_recipe,
    task5_sample_times,
)


def _isotropic_expansion(length=9):
    index = torch.arange(length, dtype=torch.float64)
    scale = torch.exp(0.08 * index)
    primitive = torch.zeros(2, 7, length, 3, dtype=torch.float64)
    axes = torch.eye(3, dtype=torch.float64)
    for axis in range(3):
        primitive[:, 1 + 2 * axis] = 0.5 * scale[:, None] * axes[axis]
        primitive[:, 2 + 2 * axis] = -0.5 * scale[:, None] * axes[axis]
    return primitive


def test_physical_time_rescales_velocity_gradient_features():
    primitive = _isotropic_expansion()
    index = torch.arange(primitive.shape[2], dtype=torch.float64)
    unit = pathline_velocity_gradient_dft_features_3d(
        primitive, num_freq=1, sample_times=index, return_numpy=False
    )
    double = pathline_velocity_gradient_dft_features_3d(
        primitive, num_freq=1, sample_times=2.0 * index, return_numpy=False
    )
    # num_freq=1 yields the four real DC values in this order:
    # IVD-like, strain norm, divergence, Q-like balance.
    torch.testing.assert_close(double[:, 0], unit[:, 0])
    torch.testing.assert_close(double[:, 1:3], 0.5 * unit[:, 1:3])
    torch.testing.assert_close(double[:, 3], 0.25 * unit[:, 3])


def test_task5_sample_times_match_rounded_cache_indices():
    times = task5_sample_times(
        np.array([32, 40, 64]), np.array([0.1, 0.2, 0.05]), 32
    )
    expected = np.rint(
        np.array([32, 40, 64])[:, None] * np.linspace(0.0, 1.0, 32)[None]
    ) * np.array([0.1, 0.2, 0.05])[:, None]
    np.testing.assert_array_equal(times, expected)
    assert np.all(np.diff(times, axis=1) > 0)


def test_recipe_parser_preserves_legacy_and_physical_names():
    assert parse_task5_feature_recipe("all_plus_gram_kinematic") == (
        "all", True, "index"
    )
    assert parse_task5_feature_recipe(
        "real_neighbor_plus_gram_physical_kinematic"
    ) == ("real_neighbor", True, "physical")
    assert parse_task5_feature_recipe("physical_kinematic") == (
        None, False, "physical"
    )
