"""Checks for the no-derivative / DCT feature variants.

The load-bearing claim is that the variant encoder is a strict generalisation of
p35: with ``derivative=True, transform="dft"`` it must reproduce the original
features exactly, so any measured difference in the other three settings is
attributable to the setting and not to a reimplementation.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.Task4C_FeatureVariants_1_1 import (
    dct_ii, dct_rotation_invariants_3d, pathline_features_variant_3d, variant_block_width)


def test_dct_matches_scipy():
    from scipy.fft import dct as scipy_dct
    torch.manual_seed(0)
    seq = torch.randn(4, 31, 3, dtype=torch.float64)
    defect = np.abs(dct_ii(seq, dim=1).numpy()
                    - scipy_dct(seq.numpy(), type=2, axis=1, norm=None)).max()
    assert defect < 1e-12, f"DCT-II disagrees with scipy by {defect:.2e}"
    print(f"  DCT-II vs scipy: {defect:.2e}")
    print("  test_dct_matches_scipy ok")


def test_variant_reproduces_p35_exactly():
    torch.manual_seed(1)
    lines = torch.randn(6, 7, 31, 3, dtype=torch.float64)
    reference = pathline_dft_features_3d(
        lines, num_freq=6, neighbor_scale=1.0, neighbor_weight=1.0,
        neighbor_pool="none", mode="gram", include_chirality=True, return_numpy=False)
    mine = pathline_features_variant_3d(lines, 6, derivative=True, transform="dft",
                                        neighbor_scale=1.0)
    defect = (reference - mine).abs().max().item()
    assert defect < 1e-10, f"variant path diverges from p35 by {defect:.2e}"
    assert mine.shape[-1] == 7 * variant_block_width(6, "dft")
    print(f"  variant(derivative, dft) vs p35: {defect:.2e}, width {mine.shape[-1]}")
    print("  test_variant_reproduces_p35_exactly ok")


def test_dct_descriptor_is_rotation_invariant_and_reflection_odd():
    from scipy.spatial.transform import Rotation
    torch.manual_seed(2)
    seq = torch.randn(8, 31, 3, dtype=torch.float64)
    rotation = torch.as_tensor(Rotation.random(4, random_state=7).as_matrix(),
                               dtype=torch.float64)
    base = dct_rotation_invariants_3d(seq, 6)
    for matrix in rotation:
        defect = (base - dct_rotation_invariants_3d(seq @ matrix.T, 6)).abs().max().item()
        assert defect < 1e-10, f"DCT descriptor is not rotation invariant: {defect:.2e}"
    mirrored = dct_rotation_invariants_3d(seq * torch.tensor([1.0, 1.0, -1.0],
                                                            dtype=torch.float64), 6)
    norms_and_cosines, triples = 11, slice(11, None)
    assert torch.allclose(base[:, :norms_and_cosines], mirrored[:, :norms_and_cosines],
                          atol=1e-10), "reflection changed a norm or cosine"
    assert torch.allclose(base[:, triples], -mirrored[:, triples], atol=1e-10), \
        "reflection did not negate the chirality slots"
    print("  test_dct_descriptor_is_rotation_invariant_and_reflection_odd ok")


def test_no_derivative_is_translation_invariant():
    torch.manual_seed(3)
    lines = torch.randn(5, 7, 31, 3, dtype=torch.float64)
    shift = torch.randn(5, 1, 1, 3, dtype=torch.float64) * 10.0
    for transform in ("dft", "dct"):
        a = pathline_features_variant_3d(lines, 6, derivative=False, transform=transform)
        b = pathline_features_variant_3d(lines + shift, 6, derivative=False,
                                         transform=transform)
        defect = (a - b).abs().max().item()
        assert defect < 1e-9, f"{transform} without dt is not translation invariant: {defect:.2e}"
        print(f"  no-derivative {transform}: translation defect {defect:.2e}")
    print("  test_no_derivative_is_translation_invariant ok")


if __name__ == "__main__":
    test_dct_matches_scipy()
    test_variant_reproduces_p35_exactly()
    test_dct_descriptor_is_rotation_invariant_and_reflection_odd()
    test_no_derivative_is_translation_invariant()
    print("TASK4C FEATURE VARIANT TEST PASSED")
