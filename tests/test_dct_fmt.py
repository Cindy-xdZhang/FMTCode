"""Tests for the DCT_FMT spin-direction (negative-frequency) bug fix.

The old implementation kept only DC + positive-frequency DFT magnitudes of the
complex signal z = x + i*y. Clockwise motion lives in NEGATIVE frequencies, so
clockwise vortices were encoded as "almost no rotation". These tests pin the
fixed behavior:
  1. clockwise and counter-clockwise primitives get features of EQUAL norm
     (mirror symmetry), and the clockwise one is far from zero;
  2. spin direction remains distinguishable (features differ);
  3. invariance to constant rotation and constant translation;
  4. declared out_dim matches the forward output.

Run:  python tests/test_dct_fmt.py   (from the repo root)
"""
import os
import sys
import math

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from FMT_Utils.DCT_FMT_encoder import DCT_FMT  # noqa: E402
from FMT_Utils.DCT_utils import dft_complex_lowfreq_mag  # noqa: E402

K, L = 5, 16


def make_rigid_rotation_primitive(omega_sign: float, cycles: float = 1.0,
                                  seed_xy=(0.3, 0.0), offset: float = 0.1):
    """One cross primitive [1, 1, K, L, 2] rigidly rotating about the origin.

    Relative vectors (neighbor - center) rotate at the same rate, completing
    `cycles` turns over the L-sample window; omega_sign=+1 -> counter-clockwise,
    -1 -> clockwise.
    """
    sx, sy = seed_xy
    pts0 = torch.tensor([
        [sx, sy], [sx + offset, sy], [sx - offset, sy],
        [sx, sy + offset], [sx, sy - offset]], dtype=torch.float64)  # [K, 2]
    t = torch.arange(L, dtype=torch.float64)
    theta = omega_sign * 2.0 * math.pi * cycles * t / L  # [L]
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # [L,2,2]
    traj = torch.einsum('lij,kj->kli', rot, pts0)  # [K, L, 2]
    return traj.unsqueeze(0).unsqueeze(0).float()  # [1, 1, K, L, 2]


def rotate_all(pathlines: torch.Tensor, theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]], dtype=pathlines.dtype)
    return torch.einsum('ij,bmklj->bmkli', rot, pathlines)


def test_lowfreq_mag_layout():
    """|Z| bins must be picked at [0, +1, -1, ..., +k, -k] of torch.fft.fft order."""
    N, k = 9, 3
    x = torch.randn(2, N, 2)
    got = dft_complex_lowfreq_mag(x, k)
    z = x[..., 0] + 1j * x[..., 1]
    Z = torch.fft.fft(z, dim=-1).abs()
    expect = torch.stack([Z[:, 0], Z[:, 1], Z[:, N - 1], Z[:, 2], Z[:, N - 2],
                          Z[:, 3], Z[:, N - 3]], dim=-1)
    assert got.shape == (2, 1 + 2 * k)
    assert torch.allclose(got, expect, atol=1e-6)
    print("ok: bidirectional bin layout [0, +1, -1, ..., +k, -k]")


def test_cw_vortex_no_longer_invisible():
    enc = DCT_FMT(nerbors=K, L=L, dct_k=6)
    f_ccw = enc(make_rigid_rotation_primitive(+1.0)).squeeze(0)
    f_cw = enc(make_rigid_rotation_primitive(-1.0)).squeeze(0)

    n_ccw, n_cw = f_ccw.norm().item(), f_cw.norm().item()
    # mirror symmetry: swapping spin swaps +m/-m bins -> identical norms
    assert abs(n_ccw - n_cw) <= 1e-4 * max(n_ccw, n_cw), (n_ccw, n_cw)
    # anti-regression: the old code encoded CW as near-zero rotation
    assert n_cw > 0.5 * n_ccw and n_cw > 1e-3, (n_ccw, n_cw)
    # spin direction still distinguishable
    assert not torch.allclose(f_ccw, f_cw, rtol=1e-3, atol=1e-5)
    print(f"ok: CW visible (||f_ccw||={n_ccw:.4f}, ||f_cw||={n_cw:.4f}), spin distinguishable")


def test_constant_rotation_invariance():
    enc = DCT_FMT(nerbors=K, L=L, dct_k=6)
    p = make_rigid_rotation_primitive(-1.0)
    f0 = enc(p)
    f1 = enc(rotate_all(p, 1.234))
    assert torch.allclose(f0, f1, rtol=1e-4, atol=1e-5)
    print("ok: constant-rotation invariance")


def test_constant_translation_invariance():
    enc = DCT_FMT(nerbors=K, L=L, dct_k=6)
    p = make_rigid_rotation_primitive(+1.0)
    f0 = enc(p)
    f1 = enc(p + torch.tensor([3.7, -1.2]))
    assert torch.allclose(f0, f1, rtol=1e-4, atol=1e-5)
    print("ok: constant-translation invariance")


def test_out_dim_and_shapes():
    for dct_k, Lc in [(6, 16), (6, 4), (2, 16)]:
        enc = DCT_FMT(nerbors=K, L=Lc, dct_k=dct_k)
        B, M = 3, 7
        tok = enc(torch.randn(B, M, K, Lc, 3))
        assert tok.shape == (B, enc.out_dim), (tok.shape, enc.out_dim)
        expected_pairs = min(dct_k, (Lc - 2) // 2)
        assert enc.per_signal_dim == 1 + 2 * expected_pairs
    print("ok: out_dim consistent (incl. clamped small-L case)")


def test_L_mismatch_raises():
    enc = DCT_FMT(nerbors=K, L=16, dct_k=6)
    try:
        enc(torch.randn(1, 2, K, 8, 2))  # wrong L
    except AssertionError:
        print("ok: L mismatch raises")
        return
    raise AssertionError("expected AssertionError for L mismatch")


if __name__ == "__main__":
    test_lowfreq_mag_layout()
    test_cw_vortex_no_longer_invisible()
    test_constant_rotation_invariance()
    test_constant_translation_invariance()
    test_out_dim_and_shapes()
    test_L_mismatch_raises()
    print("ALL DCT_FMT TESTS PASSED")
