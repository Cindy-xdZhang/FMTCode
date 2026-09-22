"""Octahedral SIREN-VAE encoder transferred from Task1 to Task4-c bundles.

Task1 encodes a 7-line pathline primitive (centre plus `x+-, y+-, z+-`) with an
`O_h`-equivariant SIREN-VAE (`FMT_Utils/SirenVAE_3D.py`).  Task4-c classifies a
bundle of 10-27 vortex lines as Hairpin / Non-hairpin.  The two line up better
than they look: c156 already forms, for **each valid line as centre**, a group of
centre + 6 farthest-point-sampled neighbours -- exactly the 7-line shape the
Task1 encoder consumes.  So the transfer replaces c156's fixed 141-D FMT plus
shared per-line network with the Task1 encoder at the same position, and keeps
c156's mean+max pooling over valid lines and its classification head.

Two structural differences from Task1, both important:

**1. The channel permutation disappears.**  Task1's six neighbours sit at the
signed axis directions, so a group element permutes them.  Here the six
neighbours are FPS-selected at arbitrary positions.  FPS depends only on
pairwise distances, which are rotation-invariant, so a global rotation selects
the *same* neighbours in the *same* order: the induced channel permutation is the
identity and the action is a pure component rotation.  A pleasant side effect is
that the group is no longer restricted to `O_h` -- any rotation is admissible,
so the coarse `SO(3)` coverage that limits Task1 (40.7 degrees mean for the
octahedral group) does not bind here.

**2. The label is NOT rotation-invariant.**  Task1's IVD is an `O(3)` invariant,
so all 48 augmentations are label-exact.  A hairpin is defined in a wall-bounded
flow with a canonical orientation -- legs downstream along `+x`, head lifted away
from the wall at low `z` -- so rotating a bundle does not preserve "is a
hairpin".  c156 accordingly uses no rotation augmentation at all.

The split latent is what makes the transfer safe anyway: the contrastive loss
pushes `z_inv` toward invariance while `z_eq` stays free and absorbs
orientation, and the classifier reads **both**.  Shape information is
regularised, orientation is retained, and nothing label-relevant is discarded.
`--group reflect` restricts augmentation to the spanwise reflection `y -> -y`,
which is a genuine statistical symmetry of both channel and TBL flow, for
comparison against the full group.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from FMT_Utils.OctahedralGroup_3D import octahedral_group
from FMT_Utils.SirenVAE_3D import ResTransformerEncoder3D
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import Residual, mlp
from FMT_Utils.Task4C_NeighborSelection_1_1 import neighbor_indices

LINES_PER_GROUP = 7
MAX_LINES = 27


def rotation_set(group="oh"):
    """Rotations used for augmentation, as ``[G, 3, 3]`` float32.

    ``oh``      : the 48 signed permutation matrices (the Task1 group).
    ``o``       : its 24 proper rotations.
    ``reflect`` : identity and the spanwise reflection ``y -> -y`` only -- the
                  one transform that is a genuine statistical symmetry of a
                  wall-bounded flow and therefore exactly label-preserving.
    ``none``    : identity only, i.e. augmentation off (c156's setting).
    """
    if group in ("oh", "o"):
        return octahedral_group(group)[0]
    if group == "reflect":
        return np.stack([np.eye(3, dtype=np.float32),
                         np.diag([1.0, -1.0, 1.0]).astype(np.float32)])
    if group == "none":
        return np.eye(3, dtype=np.float32)[None]
    raise ValueError(f"unknown group: {group!r}")


def rotate_groups(signal, element, matrices):
    """Rotate every point of ``[B, L, 7, T, 3]`` by a per-sample group element.

    No channel permutation: see the module docstring -- FPS neighbour order is
    rotation-invariant, so the induced permutation is the identity.
    """
    rotation = matrices.to(signal.dtype)[element]                # [B, 3, 3]
    return torch.einsum("bij,blktj->blkti", rotation, signal)


def build_groups(geometry, seeds, counts, neighbours=6, device="cpu"):
    """Turn cached bundles into per-centre 7-line groups.

    ``geometry`` ``[B, 27, T, 3]`` and ``seeds`` ``[B, 27, 3]`` are the frozen
    cache arrays; ``counts`` ``[B]`` is the valid-line count.  Returns

        signal ``[B, 27, 7, T, 3]`` , mask ``[B, 27]``

    with padding left at zero and excluded by the mask.  Channel 0 of each group
    is the centre line about its own centroid; channels 1-6 are neighbour minus
    centre at the same arclength index.  Both are translation-invariant and
    rotate as vectors, which is what the group action needs.
    """
    def _tensor(values, dtype):
        if torch.is_tensor(values):
            return values.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)

    geometry = _tensor(geometry, torch.float32)
    seeds = _tensor(seeds, torch.float32)
    counts = _tensor(counts, torch.long)
    batch, lines, steps, _ = geometry.shape
    if lines != MAX_LINES:
        raise ValueError(f"expected {MAX_LINES} padded lines, got {lines}")
    if int(counts.min()) < LINES_PER_GROUP:
        raise ValueError("every bundle needs at least seven valid lines for FPS")

    index = neighbor_indices(seeds, counts, "fps6")[..., :neighbours]   # [B, 27, 6]
    centre = torch.arange(lines, device=device)[None, :, None].expand(batch, -1, -1)
    ids = torch.cat((centre, index), dim=-1)                            # [B, 27, 7]

    gather = ids.view(batch, lines * LINES_PER_GROUP, 1, 1).expand(-1, -1, steps, 3)
    grouped = torch.gather(geometry, 1, gather).view(batch, lines, LINES_PER_GROUP, steps, 3)

    centre_line = grouped[:, :, :1]                                     # [B,27,1,T,3]
    signal = torch.cat((centre_line - centre_line.mean(dim=3, keepdim=True),
                        grouped[:, :, 1:] - centre_line), dim=2)
    mask = torch.arange(lines, device=device)[None] < counts[:, None]
    return signal * mask[:, :, None, None, None], mask


def masked_mean_max(values, mask):
    """Pool ``[B, L, D]`` over valid lines into ``[B, 2D]`` (mean then max)."""
    weight = mask[..., None].to(values.dtype)
    mean = (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    maximum = values.masked_fill(~mask[..., None], float("-inf")).max(dim=1).values
    return torch.cat((mean, torch.nan_to_num(maximum, neginf=0.0)), dim=-1)


class OctVAEBundleClassifier(nn.Module):
    """Task1 encoder per 7-line group, c156 pooling, c156 or linear head.

    ``head='mlp'`` reproduces c156's classifier exactly -- ``mlp(512,256)``,
    ``mlp(256,128)``, ``Linear(128,2)`` with the same LayerNorm/GELU/Dropout
    blocks -- which is why the latent is sized so that mean+max pooling gives
    512.  ``head='linear'`` is a single ``Linear(512,2)``, i.e. a linear probe on
    the same pooled representation.
    """

    def __init__(self, steps=32, z_inv_dim=144, z_eq_dim=112, head="mlp",
                 conv_channels=(64, 128), d_model=128, nhead=4, trans_layers=3,
                 dim_ff=256, dropout=0.15):
        super().__init__()
        self.z_inv_dim, self.z_eq_dim = int(z_inv_dim), int(z_eq_dim)
        self.z_dim = self.z_inv_dim + self.z_eq_dim
        self.encoder = ResTransformerEncoder3D(
            LINES_PER_GROUP * 3, int(steps), self.z_dim, conv_channels=conv_channels,
            d_model=d_model, nhead=nhead, num_layers=trans_layers, dim_ff=dim_ff)
        pooled = 2 * self.z_dim
        if head == "mlp":
            self.head = nn.Sequential(mlp(pooled, 256, dropout), mlp(256, 128, dropout),
                                      nn.Linear(128, 2))
        elif head == "linear":
            self.head = nn.Linear(pooled, 2)
        else:
            raise ValueError(f"unknown head: {head!r}")
        self.head_kind = head

    def encode_groups(self, signal):
        """``[B, L, 7, T, 3]`` -> ``mu [B, L, z_dim]``."""
        batch, lines, group, steps, comps = signal.shape
        flat = signal.permute(0, 1, 2, 4, 3).reshape(batch * lines, group * comps, steps)
        mu, _ = self.encoder(flat)
        return mu.view(batch, lines, -1)

    def forward(self, signal, mask):
        mu = self.encode_groups(signal)
        pooled = masked_mean_max(mu, mask)
        return self.head(pooled), mu

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"total": int(total), "encoder": int(total - head), "head": int(head)}
