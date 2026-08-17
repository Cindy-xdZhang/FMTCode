"""Training-free Fourier descriptors for 3D pathline-cross primitives.

The primitive layout is ``[N, K, L, C]`` where line 0 is the center and
lines 1..K-1 are spatial neighbours.  The intended 3D cross has K=7 in the
order center, x+, x-, y+, y-, z+, z-.  C is either (x,y,z) or (x,y,z,t).
"""

from __future__ import annotations

import torch


def dft_rotation_invariants_3d(
    seq: torch.Tensor,
    num_freq: int,
    mode: str = "gram",
    include_chirality: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encode ``[B,T,3]`` real vector sequences with SO(3)-invariant features.

    A constant rotation R maps each complex Fourier vector F[k] to R F[k].
    Therefore its norm and the Gram invariants of (Re F, Im F) are unchanged.
    Optional triple products preserve handedness under proper rotations while
    changing sign under reflections.
    """
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"seq must be [B,T,3], got {tuple(seq.shape)}")
    if seq.shape[1] < 1:
        raise ValueError("sequence length must be positive")

    independent_bins = seq.shape[1] // 2 + 1
    if not 1 <= int(num_freq) <= independent_bins:
        raise ValueError(
            f"num_freq={num_freq} must be in [1,{independent_bins}] for T={seq.shape[1]}"
        )
    if include_chirality and num_freq < 2:
        raise ValueError("include_chirality=True requires num_freq >= 2")

    work = seq if seq.dtype in (torch.float32, torch.float64) else seq.float()
    spectrum = torch.fft.rfft(work, dim=1)[:, :num_freq, :]
    real, imag = spectrum.real, spectrum.imag

    if mode == "magnitude":
        features = torch.linalg.vector_norm(spectrum, dim=-1)
    elif mode == "gram":
        real_norm = torch.linalg.vector_norm(real, dim=-1)
        imag_norm = torch.linalg.vector_norm(imag, dim=-1)
        cosine = (real * imag).sum(dim=-1) / (real_norm * imag_norm).clamp_min(eps)
        features = torch.stack((real_norm, imag_norm, cosine), dim=-1).flatten(1)
    else:
        raise ValueError(f"mode must be 'magnitude' or 'gram', got {mode!r}")

    if include_chirality:
        # The first value is necessarily zero because Im(F[0])=0.  Keep the
        # fixed-width slot explicitly so feature dimensions remain predictable.
        cross = torch.cross(real[:, :-1], imag[:, :-1], dim=-1)
        triple = (cross * real[:, 1:]).sum(dim=-1)
        denom = (
            torch.linalg.vector_norm(real[:, :-1], dim=-1)
            * torch.linalg.vector_norm(imag[:, :-1], dim=-1)
            * torch.linalg.vector_norm(real[:, 1:], dim=-1)
        ).clamp_min(eps)
        features = torch.cat((features, triple / denom), dim=-1)
    return features


def pathline_dft_features_3d(
    pathlines: torch.Tensor,
    valid_index: torch.Tensor | None = None,
    num_freq: int = 6,
    neighbor_weight: float = 0.5,
    neighbor_scale: float = 100.0,
    neighbor_pool: str = "sort",
    mode: str = "gram",
    include_chirality: bool = True,
    return_numpy: bool = True,
):
    """Encode 3D pathline primitives into one feature vector per seed.

    Temporal differences remove constant translations.  Neighbour trajectories
    are first expressed relative to the center pathline.  ``sort`` pools each
    feature slot across neighbours and is invariant to neighbour relabelling.
    """
    pathlines = torch.as_tensor(pathlines)
    if pathlines.ndim != 4:
        raise ValueError(f"pathlines must be [N,K,L,C], got {tuple(pathlines.shape)}")
    _, line_count, length, channels = pathlines.shape
    if line_count < 2 or length < 2 or channels not in (3, 4):
        raise ValueError(
            f"expected K>=2, L>=2 and C in {{3,4}}, got {tuple(pathlines.shape)}"
        )
    xyz = pathlines[..., :3]
    if not xyz.is_floating_point():
        xyz = xyz.float()
    if valid_index is not None:
        index = torch.as_tensor(valid_index, device=xyz.device)
        xyz = xyz[index]

    center = xyz[:, :1]
    neighbours = xyz[:, 1:]
    center_delta = center[:, :, 1:] - center[:, :, :-1]
    relative = neighbours - center
    neighbour_delta = (relative[:, :, 1:] - relative[:, :, :-1]) * neighbor_scale

    n_valid, neighbour_count, seq_len, _ = neighbour_delta.shape
    center_features = dft_rotation_invariants_3d(
        center_delta[:, 0], num_freq, mode, include_chirality
    )
    block_width = center_features.shape[-1]
    neighbour_features = dft_rotation_invariants_3d(
        neighbour_delta.reshape(n_valid * neighbour_count, seq_len, 3),
        num_freq,
        mode,
        include_chirality,
    ).reshape(n_valid, neighbour_count, block_width)

    if neighbor_pool == "none":
        pooled = neighbour_features.flatten(1)
    elif neighbor_pool == "sort":
        pooled = neighbour_features.sort(dim=1, descending=True).values.flatten(1)
    elif neighbor_pool == "mean":
        pooled = neighbour_features.mean(dim=1)
    elif neighbor_pool == "max":
        pooled = neighbour_features.amax(dim=1)
    else:
        raise ValueError("neighbor_pool must be one of: none, sort, mean, max")

    result = torch.cat((center_features, neighbor_weight * pooled), dim=-1)
    return result.detach().cpu().numpy() if return_numpy else result

