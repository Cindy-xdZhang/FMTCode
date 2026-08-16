"""Tests for FMT_Utils.FMT_encoder after removing GeoLinePicker / HierachyFMT_encoder.

Run:  python tests/test_fmt_encoder.py   (from the repo root)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from FMT_Utils.FMT_encoder import FMT, group_same_timestep  # noqa: E402


def test_group_same_timestep_matches_bruteforce():
    """Neighbor j of point (line i, timestep t) must be point (line j, timestep t)."""
    torch.manual_seed(0)
    B, K, L, C = 3, 5, 7, 4
    N = K * L
    xyz = torch.randn(B, N, 3)
    x = torch.randn(B, N, C)

    _, _, knn_xyz, knn_x = group_same_timestep(xyz, x, L)
    assert knn_xyz.shape == (B, N, K, 3)
    assert knn_x.shape == (B, N, K, C)

    for b in range(B):
        for i in range(K):
            for t in range(L):
                n = i * L + t  # flat convention: line-major
                for j in range(K):
                    assert torch.equal(knn_xyz[b, n, j], xyz[b, j * L + t])
                    assert torch.equal(knn_x[b, n, j], x[b, j * L + t])
    print("ok: group_same_timestep matches brute force")


def test_group_same_timestep_rejects_bad_N():
    xyz = torch.randn(2, 11, 3)  # 11 not divisible by L=5
    try:
        group_same_timestep(xyz, xyz.clone(), 5)
    except AssertionError:
        print("ok: non-divisible N raises")
        return
    raise AssertionError("expected AssertionError for N % L != 0")


def test_fmt_forward_shapes():
    torch.manual_seed(0)
    B, K, L = 4, 5, 6
    N = K * L
    xyz = torch.randn(B, N, 3)
    x = xyz.permute(0, 2, 1).contiguous()

    for head, name in [(None, "none"), ("dft", "dft")]:
        enc = FMT(PathlineLtimesteps=L, num_stages=2, embed_dim=24,
                  alpha=1000, beta=19, temporal_head=head)
        enc.eval()
        with torch.no_grad():
            out = enc(xyz, x)
        assert out.shape == (B, enc.out_dim) == (B, 24 * 2 ** 2), \
            f"temporal_head={name}: got {tuple(out.shape)}"
        assert torch.isfinite(out).all()
    print("ok: FMT forward shapes (temporal_head None / dft)")


if __name__ == "__main__":
    test_group_same_timestep_matches_bruteforce()
    test_group_same_timestep_rejects_bad_N()
    test_fmt_forward_shapes()
    print("ALL FMT_encoder TESTS PASSED")
