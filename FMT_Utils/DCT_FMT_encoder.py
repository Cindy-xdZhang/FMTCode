"""
DCT_FMT: a training-free, Fourier-based flowmap tokenizer.

(Naming note: the class historically carries "DCT" in its name but the transform
used is the complex DFT/FFT along time, not a discrete cosine transform.)

Motivation
----------
The original FMT encoder (``pnn.models.point_nn.EncNPNew`` / ``FMT_encoder.FMT``)
treats a window of pathlines as an *unordered point cloud* (KNN + positional
encoding + max/mean pooling). This makes the temporal axis of each pathline a
weak signal: ordering along time is essentially discarded by the global pooling,
and KNN over ``M*K*L`` points is slow.

DCT_FMT keeps the same *role* (window of pathlines -> single feature token of a
fixed dimension) but extracts the token in the **frequency domain along time**:

  * Each 2D pathline is read as a complex signal  z[n] = x[n] + i*y[n].
  * We take DFT magnitudes of
        - the center pathline's per-step velocity   (center_dt),
        - each neighbor's relative displacement rate (d(neighbor-center)/dt).
  * Keeping magnitudes makes the descriptor invariant to a constant rotation
    of the trajectory (rotation by theta multiplies every DFT coefficient by
    e^{i*theta}).
  * Per signal we keep |Z[0]| plus the first ``dct_k`` POSITIVE **and** NEGATIVE
    frequency magnitudes, interleaved as
        [ |Z[0]|, |Z[+1]|, |Z[-1]|, ..., |Z[+k]|, |Z[-k]| ].

    Why both signs (bug fix, 2026-08): the spectrum of a *complex* signal is not
    conjugate-symmetric. Counter-clockwise rotation concentrates energy in
    positive-frequency bins, clockwise rotation in negative-frequency bins.
    The previous implementation kept only ``|Z[0..k-1]|`` (DC + positive bins),
    so clockwise vortices were encoded as "almost no rotation" -- fatal for
    e.g. von Karman streets where shed vortices alternate spin. The +m/-m pair
    preserves spin information; their sum/difference (spin-invariant magnitude /
    signed chirality) is an orthogonal linear recombination, so Euclidean
    distances -- and hence KMeans -- are unaffected by this basis choice.

  * The per-seeding vectors are max/mean pooled across the window.

The encoder has **no learnable parameters** (it is an ``nn.Module`` only so it
moves with ``.to(device)`` and plugs into the trainable UNet/ViT backends the
same way ``EncNPNew`` does).

Interface
---------
``forward(pathlines)`` expects *structured* pathlines (NOT a flat point cloud),
shape ``[B, M, K, L, D]`` with ``D >= 2``:
    B = batch, M = #seedings in the window, K = #cross pathlines per seeding
    (index 0 is the seeding's own / "center" line, 1.. are neighbors),
    L = #time samples per line, D = point dim (x, y, [t]).
Returns a token of shape ``[B, out_dim]`` with
    out_dim = per_signal_dim * (1 if use_center else 0) + (K-1) * per_signal_dim,
    per_signal_dim = 1 + 2 * k_pairs,  k_pairs = min(dct_k, (L-2)//2).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from FMT_Utils.DCT_utils import dft_complex_lowfreq_mag


class DCT_FMT(nn.Module):
    def __init__(self,
                 nerbors: int,
                 L: int,
                 dct_k: int = 6,
                 dct_weight: float = 0.5,
                 neighbor_diff_scale: float = 100.0,
                 use_center: bool = True):
        """
        Args:
            nerbors:             #cross pathlines per seeding (K), incl. the center line.
            L:                   #time samples per pathline.
            dct_k:               #(+m, -m) frequency PAIRS to keep per signal (plus DC,
                                 which is always kept). Clamped to (seq_len-1)//2 so that
                                 +m and -m are distinct bins; a warning is logged when
                                 clamping changes the requested value.
            dct_weight:          scale applied to the neighbor block when concatenating
                                 (matches the colleague's clustering recipe).
            neighbor_diff_scale: multiplier on the neighbor relative-velocity signal
                                 (the relative displacement rate is small, so it is
                                 amplified before the DFT, as in the prototype).
            use_center:          include the center pathline's velocity-spectrum block.
        """
        super().__init__()
        self.nerbors = int(nerbors)
        self.L = int(L)
        self.dct_weight = float(dct_weight)
        self.neighbor_diff_scale = float(neighbor_diff_scale)
        self.use_center = bool(use_center)

        # DFT is applied to per-step differences, so the usable length is L-1.
        self.seq_len = max(1, self.L - 1)

        # +/- pairs require distinct bins: m and N-m coincide beyond (N-1)//2.
        max_pairs = max(0, (self.seq_len - 1) // 2)
        self.k_pairs = int(max(0, min(int(dct_k), max_pairs)))
        if self.k_pairs != int(dct_k):
            logging.warning(
                "[DCT_FMT] dct_k=%d clamped to k_pairs=%d (seq_len=%d supports at most "
                "(seq_len-1)//2 distinct +/- pairs); per-signal dim = %d.",
                int(dct_k), self.k_pairs, self.seq_len, 1 + 2 * self.k_pairs)

        # DC + interleaved (+m, -m) magnitudes per signal.
        self.per_signal_dim = 1 + 2 * self.k_pairs

        self.num_neighbors = max(0, self.nerbors - 1)

        # Output token dimension (deterministic; no learnable params).
        center_dim = self.per_signal_dim if self.use_center else 0
        neighbor_dim = self.num_neighbors * self.per_signal_dim
        self.out_dim = int(center_dim + neighbor_dim)
        assert self.out_dim > 0, "DCT_FMT produced an empty feature; check nerbors/L/dct_k."

    def extra_repr(self) -> str:
        return (f"nerbors={self.nerbors}, L={self.L}, k_pairs={self.k_pairs}, "
                f"per_signal_dim={self.per_signal_dim}, dct_weight={self.dct_weight}, "
                f"out_dim={self.out_dim}")

    def forward(self, pathlines: torch.Tensor) -> torch.Tensor:
        """
        pathlines: [B, M, K, L, D]  (structured window of pathlines, D >= 2)
        returns:   [B, out_dim]
        """
        assert pathlines.dim() == 5, \
            f"DCT_FMT expects [B, M, K, L, D], got {tuple(pathlines.shape)}"
        B, M, K, L, D = pathlines.shape
        assert K == self.nerbors, f"K mismatch: expected {self.nerbors}, got {K}"
        assert L == self.L, f"L mismatch: expected {self.L}, got {L}"
        assert D >= 2, f"point dim must be >= 2 (x,y), got {D}"

        # keep only (x, y) -> complex signal channels
        xy = pathlines[..., :2].to(torch.float32)          # [B, M, K, L, 2]
        center = xy[:, :, 0:1, :, :]                        # [B, M, 1, L, 2]

        feats = []

        if self.use_center:
            # center per-step velocity, then bidirectional low-freq |DFT| of z=x+iy
            center_dt = center[:, :, :, 1:, :] - center[:, :, :, :-1, :]  # [B,M,1,L-1,2]
            ce = center_dt.reshape(B * M * 1, self.seq_len, 2)
            ce_feat = dft_complex_lowfreq_mag(ce, self.k_pairs)            # [B*M, 1+2k]
            feats.append(ce_feat.reshape(B, M, 1 * self.per_signal_dim))

        if self.num_neighbors > 0:
            neighbor = xy[:, :, 1:, :, :]                   # [B, M, K-1, L, 2]
            nd = neighbor - center                          # relative pos  [B,M,K-1,L,2]
            nd_dt = (nd[:, :, :, 1:, :] - nd[:, :, :, :-1, :]) * self.neighbor_diff_scale
            ne = nd_dt.reshape(B * M * self.num_neighbors, self.seq_len, 2)
            ne_feat = dft_complex_lowfreq_mag(ne, self.k_pairs)            # [.., 1+2k]
            ne_feat = ne_feat.reshape(B, M, self.num_neighbors * self.per_signal_dim)
            feats.append(self.dct_weight * ne_feat)

        per_seed = torch.cat(feats, dim=-1)                 # [B, M, out_dim]

        # Aggregate the window's seedings into one token (same max+mean as FMT).
        token = per_seed.max(dim=1)[0] + per_seed.mean(dim=1)  # [B, out_dim]
        return token
