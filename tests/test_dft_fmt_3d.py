import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.DFT_FMT_3D import (
    dft_rotation_invariants_3d,
    pathline_dft_features_3d,
    pathline_velocity_gradient_dft_features_3d,
    time_local_gram_dft_features_3d,
)


def random_rotation(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q, dtype=torch.float64)


def make_primitives(n=8, lines=7, length=32):
    rng = torch.Generator().manual_seed(4)
    velocity = torch.randn(n, lines, length - 1, 3, generator=rng, dtype=torch.float64)
    points = torch.cat((torch.zeros(n, lines, 1, 3, dtype=torch.float64), velocity), dim=2)
    points = points.cumsum(dim=2)
    return points


def test_rotation_invariance():
    seq = make_primitives()[:, 0]
    rotation = random_rotation()
    for mode in ("magnitude", "gram"):
        before = dft_rotation_invariants_3d(seq, 6, mode, include_chirality=True)
        after = dft_rotation_invariants_3d(seq @ rotation.T, 6, mode, include_chirality=True)
        torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-10)


def test_translation_and_neighbour_permutation_invariance():
    primitives = make_primitives()
    translated = primitives + torch.tensor([4.0, -2.0, 7.0])
    permutation = torch.tensor([0, 4, 2, 6, 1, 5, 3])
    base = pathline_dft_features_3d(primitives, neighbor_pool="sort", return_numpy=False)
    moved = pathline_dft_features_3d(
        translated[:, permutation], neighbor_pool="sort", return_numpy=False
    )
    torch.testing.assert_close(base, moved, rtol=1e-10, atol=1e-10)


def test_reflection_flips_nonzero_chirality_slots():
    seq = make_primitives()[:, 0]
    reflected = seq * torch.tensor([-1.0, 1.0, 1.0])
    proper = dft_rotation_invariants_3d(seq, 6, "gram", True)
    mirror = dft_rotation_invariants_3d(reflected, 6, "gram", True)
    width_without_chirality = 18
    torch.testing.assert_close(
        proper[:, :width_without_chirality], mirror[:, :width_without_chirality]
    )
    torch.testing.assert_close(
        proper[:, width_without_chirality:], -mirror[:, width_without_chirality:]
    )
    assert proper[:, width_without_chirality:].abs().max() > 1e-4


def test_time_local_gram_is_invariant_to_time_varying_rigid_observer():
    primitives = make_primitives(n=3)
    transformed = torch.empty_like(primitives)
    for time_index in range(primitives.shape[2]):
        rotation = random_rotation(100 + time_index)
        translation = torch.tensor(
            [0.1 * time_index, -0.03 * time_index ** 2, 2.0],
            dtype=primitives.dtype,
        )
        transformed[:, :, time_index] = (
            primitives[:, :, time_index] @ rotation.T + translation
        )
    before = time_local_gram_dft_features_3d(
        primitives, num_freq=6, return_numpy=False
    )
    after = time_local_gram_dft_features_3d(
        transformed, num_freq=6, return_numpy=False
    )
    assert before.shape == (3, 231)
    torch.testing.assert_close(before, after, rtol=1e-9, atol=1e-9)


def test_velocity_gradient_block_constant_rigid_invariance_and_shape():
    primitives = make_primitives(n=12)
    offsets = torch.tensor([
        [0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.2],
        [0.0, 0.0, -0.2],
    ], dtype=primitives.dtype)
    primitives = primitives + offsets[None, :, None, :]
    rotation = random_rotation(91)
    translation = torch.tensor([3.0, -4.0, 2.0], dtype=primitives.dtype)
    before = pathline_velocity_gradient_dft_features_3d(
        primitives, num_freq=6, return_numpy=False
    )
    after = pathline_velocity_gradient_dft_features_3d(
        primitives @ rotation.T + translation, num_freq=6,
        return_numpy=False,
    )
    assert before.shape == (12, 44)
    torch.testing.assert_close(before, after, rtol=1e-8, atol=1e-8)




if __name__ == "__main__":
    test_rotation_invariance()
    test_translation_and_neighbour_permutation_invariance()
    test_reflection_flips_nonzero_chirality_slots()
    test_time_local_gram_is_invariant_to_time_varying_rigid_observer()
    test_velocity_gradient_block_constant_rigid_invariance_and_shape()
    print("ALL DFT_FMT_3D TESTS PASSED")
