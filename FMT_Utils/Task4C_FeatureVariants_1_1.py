"""FMT feature variants: no time derivative, and DCT in place of the DFT.

The p35 encoder takes a discrete time derivative before transforming --
``center_delta = x[t] - x[t-1]`` -- which removes constant translation for free
and makes the descriptor depend on the *shape of the motion* rather than on
where the line sits.  This module provides the two variants that isolate those
choices:

* ``derivative=False`` transforms the raw spatial track with the root's own
  ``t=0`` sample moved to the origin, ``x[t] - x[0]``.  Translation invariance is
  still exact, but the low frequencies now carry absolute excursion rather than
  velocity.
* ``transform="dct"`` replaces the complex DFT with a real DCT-II.

Both are new files; `FMT_Utils/DFT_FMT_3D.py` is untouched.

Feature width.  A complex spectrum gives two real vectors per frequency (Re, Im)
and hence the p35 block ``3k + (k-1) = 4k-1`` (23 for k=6).  A real transform
gives **one** vector per frequency, so the within-frequency Gram invariants
collapse to a norm and the natural analogue uses consecutive frequency pairs:

    ‖C_k‖ (k)  +  cos∠(C_k, C_{k+1}) (k-1)  +  det[C_k,C_{k+1},C_{k+2}] (k-2)

giving ``3k-3`` (15 for k=6).  The invariance argument is unchanged: a constant
rotation R maps every C_k to R C_k, so norms, pairwise cosines and normalised
triple products are preserved, and the triple products still flip sign under
reflection.
"""
from __future__ import annotations

import math

import torch

EPS = 1e-8


def dct_ii(seq: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Unnormalised DCT-II along ``dim``, matching ``scipy.fft.dct(type=2)``."""
    work = seq if seq.dtype in (torch.float32, torch.float64) else seq.float()
    work = work.movedim(dim, -1)
    length = work.shape[-1]
    reordered = torch.cat((work[..., 0::2], work[..., 1::2].flip(-1)), dim=-1)
    spectrum = torch.fft.fft(reordered, dim=-1)
    index = torch.arange(length, device=work.device, dtype=work.dtype)   # keep float64 in float64
    twiddle = torch.exp(-1j * math.pi * index / (2 * length))
    return (2 * (spectrum * twiddle).real).movedim(-1, dim)


def dct_rotation_invariants_3d(seq: torch.Tensor, num_freq: int,
                               include_chirality: bool = True,
                               eps: float = EPS) -> torch.Tensor:
    """``[B,T,3]`` real sequences -> ``3k-3`` SO(3) invariants of the DCT spectrum."""
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"seq must be [B,T,3], got {tuple(seq.shape)}")
    if not 1 <= int(num_freq) <= seq.shape[1]:
        raise ValueError(f"num_freq={num_freq} must be in [1,{seq.shape[1]}]")
    if include_chirality and num_freq < 3:
        raise ValueError("include_chirality=True requires num_freq >= 3")

    coefficients = dct_ii(seq, dim=1)[:, :num_freq, :]                 # [B,k,3]
    norm = torch.linalg.vector_norm(coefficients, dim=-1)              # [B,k]
    left, right = coefficients[:, :-1], coefficients[:, 1:]
    cosine = (left * right).sum(-1) / (norm[:, :-1] * norm[:, 1:]).clamp_min(eps)
    features = torch.cat((norm, cosine), dim=-1)
    if include_chirality:
        triple = (torch.cross(coefficients[:, :-2], coefficients[:, 1:-1], dim=-1)
                  * coefficients[:, 2:]).sum(-1)
        denominator = (norm[:, :-2] * norm[:, 1:-1] * norm[:, 2:]).clamp_min(eps)
        features = torch.cat((features, triple / denominator), dim=-1)
    return features


def variant_block_width(num_freq: int, transform: str, include_chirality: bool = True) -> int:
    if transform == "dft":
        return 4 * num_freq - 1 if include_chirality else 3 * num_freq
    return 3 * num_freq - 3 if include_chirality else 2 * num_freq - 1


def pathline_features_variant_3d(pathlines: torch.Tensor, num_freq: int = 6,
                                 derivative: bool = True, transform: str = "dft",
                                 neighbor_scale: float = 1.0,
                                 include_chirality: bool = True) -> torch.Tensor:
    """Per-primitive blocks ``[N, (1+K-1) * block]`` with the centre block first.

    Mirrors ``pathline_dft_features_3d(..., neighbor_pool="none", mode="gram")``
    so the two can be swapped without touching the graph model.
    """
    from FMT_Utils.DFT_FMT_3D import dft_rotation_invariants_3d

    if pathlines.ndim != 4:
        raise ValueError(f"pathlines must be [N,K,L,C], got {tuple(pathlines.shape)}")
    xyz = pathlines[..., :3]
    if not xyz.is_floating_point():
        xyz = xyz.float()

    centre, neighbours = xyz[:, :1], xyz[:, 1:]
    relative = neighbours - centre
    if derivative:
        centre_signal = centre[:, :, 1:] - centre[:, :, :-1]
        neighbour_signal = (relative[:, :, 1:] - relative[:, :, :-1]) * neighbor_scale
    else:                                   # raw track, the root's t=0 at the origin
        centre_signal = centre - centre[:, :, :1]
        neighbour_signal = relative * neighbor_scale

    if transform == "dft":
        encode = lambda x: dft_rotation_invariants_3d(x, num_freq, "gram", include_chirality)
    elif transform == "dct":
        encode = lambda x: dct_rotation_invariants_3d(x, num_freq, include_chirality)
    else:
        raise ValueError(f"transform must be 'dft' or 'dct', got {transform!r}")

    count, neighbour_count, length, _ = neighbour_signal.shape
    centre_features = encode(centre_signal[:, 0])
    neighbour_features = encode(
        neighbour_signal.reshape(count * neighbour_count, length, 3)
    ).reshape(count, neighbour_count, -1)
    return torch.cat((centre_features, neighbour_features.flatten(1)), dim=-1)
