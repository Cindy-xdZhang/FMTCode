"""Tests for the Task3 seed-anchored kinematic Fourier block."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.DFT_FMT_3D import (
    pathline_anchored_kinematic_dft_features_3d,
)


def _rotation(seed=0):
    matrix, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return torch.tensor(matrix, dtype=torch.float64)


def _primitives(count=12, length=32):
    generator = torch.Generator().manual_seed(73)
    velocity = torch.randn(
        count, 7, length - 1, 3, generator=generator, dtype=torch.float64
    )
    points = torch.cat(
        (torch.zeros(count, 7, 1, 3, dtype=torch.float64), velocity), dim=2
    ).cumsum(dim=2)
    offsets = torch.tensor([
        [0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.2],
        [0.0, 0.0, -0.2],
    ], dtype=points.dtype)
    return points + offsets[None, :, None, :]


def test_shape_and_constant_rigid_invariance():
    primitives = _primitives()
    rotation = _rotation(93)
    translation = torch.tensor([2.0, -3.0, 0.5], dtype=primitives.dtype)
    before = pathline_anchored_kinematic_dft_features_3d(
        primitives, num_freq=4, window=16, channels=(0, 3),
        return_numpy=False,
    )
    after = pathline_anchored_kinematic_dft_features_3d(
        primitives @ rotation.T + translation,
        num_freq=4,
        window=16,
        channels=(0, 3),
        return_numpy=False,
    )
    # Two channels x ((4 real + 3 imaginary) + 7 anchors).
    assert before.shape == (12, 28)
    torch.testing.assert_close(before, after, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    test_shape_and_constant_rigid_invariance()
    print("ANCHORED KINEMATIC FEATURE TESTS PASSED")
