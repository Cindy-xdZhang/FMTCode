"""
Fourier / DCT based temporal feature extraction utilities.

These helpers are the building blocks for the training-free ``DCT_FMT`` encoder
(see :mod:`FMT_Utils.DCT_FMT_encoder`). The key idea (suggested by a colleague,
originally prototyped in ``DCT_utils_xx.py`` / ``FMT_Clustering_xx.py``) is to
treat a 2D pathline as a *complex* signal ``z[n] = x[n] + i*y[n]`` and use the
magnitude of its DFT as a rotation-invariant temporal descriptor:

    rotating the pathline by theta maps  z -> e^{i*theta} * z,
    hence                                Z[k] -> e^{i*theta} * Z[k],
    so |Z[k]| is unchanged  ==> rotation invariant.

This addresses FMT's weak temporal fusion: instead of pooling points as an
unordered cloud, we explicitly encode the per-frequency content along time.
"""

import math
import torch


def create_dct_matrix(N: int, device=None, dtype=None):
    """
    Create an orthonormal DCT-II transform matrix [N x N].
    Equivalent to scipy.fft.dct(type=2, norm='ortho').
    """
    n = torch.arange(N, device=device, dtype=dtype).reshape(1, N)
    k = torch.arange(N, device=device, dtype=dtype).reshape(N, 1)
    dct_mat = torch.cos(math.pi / N * (n + 0.5) * k) * math.sqrt(2.0 / N)
    dct_mat[0, :] /= math.sqrt(2.0)
    return dct_mat  # [N, N]


def dct_1d(x: torch.Tensor):
    """
    Orthonormal DCT-II for batched input.
    x: [B, N, C]
    """
    N = x.shape[1]
    dct_mat = create_dct_matrix(N, device=x.device, dtype=x.dtype)
    X = torch.einsum('nk,bkc->bnc', dct_mat, x)
    return X


def idct_1d(X: torch.Tensor):
    """
    Orthonormal inverse DCT (type III), inverse of :func:`dct_1d`.
    X: [B, N, C]
    """
    N = X.shape[1]
    dct_mat = create_dct_matrix(N, device=X.device, dtype=X.dtype)
    x = torch.einsum('kn,bkc->bnc', dct_mat, X)
    return x


def dft_complex_lowfreq_mag(x: torch.Tensor, k_pairs: int) -> torch.Tensor:
    """
    Low-frequency DFT magnitudes of a complex pathline signal z[n] = x[n] + i*y[n],
    keeping BOTH spin directions.

    Why both signs: for a complex signal the spectrum is NOT conjugate-symmetric.
    Counter-clockwise rotation (z ~ e^{+i w t}) puts its energy in POSITIVE
    frequency bins, clockwise rotation (z ~ e^{-i w t}) in NEGATIVE bins.
    Keeping only Z[0..k] (as the old code did) makes clockwise vortices nearly
    invisible to the descriptor. Here we return, per signal:

        [ |Z[0]|, |Z[+1]|, |Z[-1]|, |Z[+2]|, |Z[-2]|, ..., |Z[+k]|, |Z[-k]| ]

    shape [..., 1 + 2*k_pairs]. All entries are invariant to a constant rotation
    of the trajectory (global phase e^{i*theta}); the +m/-m pair encodes spin
    direction. (Their symmetric/antisymmetric combinations are an orthogonal
    change of basis up to a uniform sqrt(2) factor, so KMeans assignments are
    identical under either representation.)

    Args:
        x:       [..., N, 2] with x[...,0]=x-coord, x[...,1]=y-coord.
        k_pairs: number of +/- frequency pairs to keep; requires
                 0 <= k_pairs <= (N-1)//2 so that +m and -m are distinct bins.
    """
    z = x[..., 0].to(torch.float32) + 1j * x[..., 1].to(torch.float32)  # [..., N]
    Z = torch.fft.fft(z, dim=-1)  # bin order: [0, +1, ..., +floor(N/2), ..., -2, -1]
    N = Z.shape[-1]
    k = int(k_pairs)
    assert 0 <= k <= (N - 1) // 2, \
        f"k_pairs={k} out of range for N={N}; need k_pairs <= (N-1)//2 = {(N - 1) // 2}"
    bins = [0]
    for m in range(1, k + 1):
        bins.extend([m, N - m])  # +m, then -m
    index = torch.as_tensor(bins, dtype=torch.long, device=Z.device)
    return Z.abs().index_select(-1, index)  # [..., 1 + 2k]


def dft_complex_1d(x: torch.Tensor, return_mag: bool = True):
    """
    DFT on a complex pathline signal z[n] = x[n] + i*y[n].

    Args:
        x:          [..., N, 2] where x[...,0]=x-coord, x[...,1]=y-coord.
                    Any number of leading (batch) dimensions is supported.
        return_mag: if True, return the rotation-invariant magnitudes |Z[k]|
                    of shape [..., N]; otherwise return the complex Z[k].

    Rotation by theta maps z -> e^{i*theta} z, so Z[k] -> e^{i*theta} Z[k];
    the magnitudes are therefore rotation invariant.
    """
    z = x[..., 0].to(torch.float32) + 1j * x[..., 1].to(torch.float32)  # [..., N] complex
    Z = torch.fft.fft(z, dim=-1)  # [..., N] complex
    if return_mag:
        return Z.abs()  # [..., N] real, rotation-invariant
    return Z
