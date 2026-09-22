"""Hierarchical FMT graph: coarse bundle centre, fine FPS-neighbour sub-stars.

The p35 baseline makes **every** valid line a centre, encodes each 7-line group
to 141-D and mean+max pools ~25 of them.  That is flat: every line contributes
symmetrically and the bundle's own seeding centre carries no special status.

This replaces the flat pooling with a two-level graph rooted at the lattice
centre, the line actually seeded at the cube's origin:

    coarse   centre c  <--GATv2--  its 6 FPS neighbours j
    fine     each j    <--GATv2--  j's own 6 nearest neighbours

and gives every node a role-aware feature, using the fact that `FMT` produces
two *different* descriptors for the same line depending on the role it plays:

* **as centre** -- `dft_rotation_invariants_3d` of the line's own arclength
  differences (23-D for k=6);
* **as neighbour of p** -- the same invariants computed on `(line - p)`
  differences, so the descriptor encodes the line *relative to its parent*.

A fine node therefore carries [as-neighbour-of-c, as-own-centre, GATv2 over its
own 6 nearest neighbours], exactly the three components requested.  The root
additionally carries the 72-D direction spectrum, the one non-rotation-invariant
block of p35, so bundle orientation is not discarded.

Why this can use the whole dataset: it needs only the centre line to exist, not
the complete 6-face octahedral star.  The centre survives in 99.55% of bundles
(against 33% for a complete star), and every bundle has >= 10 valid lines, so
FPS-6 and nearest-6 are always defined.  Results are therefore directly
comparable to the .888 baseline on the full 193,000 / 10,000 split.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_P35_NormFrequency_3_1 import direction_spectrum, normalize_geometry
from FMT_Utils.Task4C_NeighborSelection_1_1 import neighbor_indices
from FMT_Utils.Task4C_FeatureVariants_1_1 import (
    pathline_features_variant_3d, variant_block_width)
from FMT_Utils.Task4C_OctVAE_2_1 import lattice_codes

FREQUENCIES = 6
BLOCK = 4 * FREQUENCIES - 1          # 23, one line's invariant descriptor
DIRECTION = 12 * FREQUENCIES         # 72, the orientation-carrying block
NEIGHBOURS = 6


def centre_rows(seeds, meta):
    """Row index of the lattice-centre line per bundle, with a fallback flag.

    The centre is the line seeded at cube origin (0,0,0).  It is cleaned away in
    ~0.45% of bundles; those fall back to the first valid row so that no bundle
    is dropped and the split stays comparable to the published baseline.
    """
    codes, valid = lattice_codes(seeds, meta)
    is_centre = valid & (codes == 0).all(-1)
    found = is_centre.any(1)
    rows = np.where(found, is_centre.argmax(1), 0).astype(np.int64)
    return rows, ~found


@torch.no_grad()
def precompute(geometry, seeds, counts, meta, device, chunk=512):
    """Role-aware FMT blocks for the two-level graph.

    Returns a dict of tensors, per bundle:
      centre_inv [23]      centre as its own centre
      centre_dir [72]      centre direction spectrum (not rotation invariant)
      fps_nb  [6, 23]      each FPS neighbour, as a neighbour of the centre
      fps_ctr [6, 23]      each FPS neighbour, as its own centre
      fps_nn  [6, 6, 23]   each FPS neighbour's 6 nearest neighbours, as its neighbours
    """
    rows, fell_back = centre_rows(seeds, meta)
    out = {k: [] for k in ("centre_inv", "centre_dir", "fps_nb", "fps_ctr", "fps_nn")}
    for start in range(0, len(geometry), chunk):
        stop = start + chunk
        g = torch.as_tensor(np.asarray(geometry[start:stop]), device=device)
        c = torch.as_tensor(np.asarray(counts[start:stop]), device=device, dtype=torch.long)
        s = torch.as_tensor(np.asarray(seeds[start:stop]), device=device)
        x, _ = normalize_geometry(g, c, "max_radius")
        batch = len(x)
        index = torch.arange(batch, device=device)
        root = torch.as_tensor(rows[start:stop], device=device)

        fps = neighbor_indices(s, c, "fps6")[index, root]                 # [B, 6]
        near = neighbor_indices(s, c, "nearest6")                          # [B, 27, 6]
        fine_near = near[index[:, None], fps]                              # [B, 6, 6]

        coarse_ids = torch.cat((root[:, None], fps), 1)                    # [B, 7]
        coarse = pathline_dft_features_3d(
            x[index[:, None], coarse_ids], num_freq=FREQUENCIES, neighbor_scale=1.,
            neighbor_weight=1., neighbor_pool="none", mode="gram",
            include_chirality=True, return_numpy=False)
        out["centre_inv"].append(coarse[:, :BLOCK].float().cpu())
        out["fps_nb"].append(coarse[:, BLOCK:].reshape(batch, NEIGHBOURS, BLOCK).float().cpu())
        out["centre_dir"].append(direction_spectrum(x[index, root], FREQUENCIES).float().cpu())

        fine_ids = torch.cat((fps[:, :, None], fine_near), 2)              # [B, 6, 7]
        fine = pathline_dft_features_3d(
            x[index[:, None, None], fine_ids].reshape(batch * NEIGHBOURS, 7, x.shape[2], 3),
            num_freq=FREQUENCIES, neighbor_scale=1., neighbor_weight=1.,
            neighbor_pool="none", mode="gram", include_chirality=True, return_numpy=False)
        out["fps_ctr"].append(fine[:, :BLOCK].reshape(batch, NEIGHBOURS, BLOCK).float().cpu())
        out["fps_nn"].append(fine[:, BLOCK:].reshape(
            batch, NEIGHBOURS, NEIGHBOURS, BLOCK).float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}, int(fell_back.sum())


class GATv2(nn.Module):
    """Star-topology GATv2: one centre attends over a fixed set of neighbours.

    GATv2 applies the nonlinearity *after* the two linear maps and before the
    attention vector, so the ranking of neighbours can depend on the centre --
    the fix to GAT's static attention.
    """

    def __init__(self, centre_dim, neighbour_dim, out_dim, heads=4, dropout=0.15,
                 slope=0.2):
        super().__init__()
        if out_dim % heads:
            raise ValueError("out_dim must divide by heads")
        self.heads, self.per_head = heads, out_dim // heads
        self.centre = nn.Linear(centre_dim, out_dim)
        self.neighbour = nn.Linear(neighbour_dim, out_dim)
        self.attention = nn.Parameter(torch.empty(heads, self.per_head))
        nn.init.xavier_uniform_(self.attention)
        self.project = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.slope = slope

    def forward(self, centre, neighbours, mask=None):
        batch, count, _ = neighbours.shape
        c = self.centre(centre).view(batch, 1, self.heads, self.per_head)
        n = self.neighbour(neighbours).view(batch, count, self.heads, self.per_head)
        score = (F.leaky_relu(c + n, self.slope) * self.attention).sum(-1)      # [B,N,H]
        if mask is not None:
            score = score.masked_fill(~mask[..., None], float("-inf"))
        weight = torch.softmax(score, dim=1)
        if mask is not None:                    # a row with no valid node -> all -inf
            weight = torch.nan_to_num(weight)
        weight = self.drop(weight)
        pooled = (weight.unsqueeze(-1) * n).sum(1).reshape(batch, -1)
        return self.norm(self.project(pooled)), weight


def block(cin, cout, dropout):
    return nn.Sequential(nn.Linear(cin, cout), nn.LayerNorm(cout), nn.GELU(),
                         nn.Dropout(dropout))


class HierarchicalFMTGraph(nn.Module):
    """Two-level FMT graph -> `representation_dim` -> binary head."""

    def __init__(self, fine_dim=128, representation_dim=256, heads=4, dropout=0.15,
                 use_direction=True, use_fine=True, head_hidden=128):
        super().__init__()
        self.use_direction, self.use_fine = use_direction, use_fine
        self.fine_query = block(BLOCK, fine_dim, dropout)
        self.fine_key = block(BLOCK, fine_dim, dropout)
        self.fine_gat = GATv2(fine_dim, fine_dim, fine_dim, heads, dropout)
        fps_in = BLOCK + BLOCK + (fine_dim if use_fine else 0)
        self.fps_proj = nn.Sequential(block(fps_in, representation_dim, dropout),
                                      block(representation_dim, representation_dim, dropout))
        centre_in = BLOCK + (DIRECTION if use_direction else 0)
        self.centre_proj = nn.Sequential(block(centre_in, representation_dim, dropout),
                                         block(representation_dim, representation_dim, dropout))
        self.coarse_gat = GATv2(representation_dim, representation_dim,
                                representation_dim, heads, dropout)
        self.fuse = block(2 * representation_dim, representation_dim, dropout)
        self.head = nn.Sequential(block(representation_dim, head_hidden, dropout),
                                  nn.Linear(head_hidden, 2))

    def forward(self, batch, return_attention=False):
        fps_ctr, fps_nn = batch["fps_ctr"], batch["fps_nn"]
        size, count = fps_ctr.shape[0], fps_ctr.shape[1]
        parts = [batch["fps_nb"], fps_ctr]
        fine_weight = None
        if self.use_fine:
            query = self.fine_query(fps_ctr.reshape(size * count, BLOCK))
            key = self.fine_key(fps_nn.reshape(size * count, NEIGHBOURS, BLOCK))
            fine, fine_weight = self.fine_gat(query, key)
            parts.append(fine.reshape(size, count, -1))
        fps = self.fps_proj(torch.cat(parts, -1))

        centre = batch["centre_inv"]
        if self.use_direction:
            centre = torch.cat((centre, batch["centre_dir"]), -1)
        centre = self.centre_proj(centre)
        coarse, coarse_weight = self.coarse_gat(centre, fps)
        representation = self.fuse(torch.cat((centre, coarse), -1))
        logits = self.head(representation)
        if return_attention:
            return logits, representation, coarse_weight, fine_weight
        return logits

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"total": int(total), "backbone": int(total - head), "head": int(head)}


# ---------------------------------------------------------------------------
# Multi-root variant
# ---------------------------------------------------------------------------
# Round 2 showed the single-root hierarchy plateauing near .79 at every learning
# rate, well short of the .888 baseline.  The likely reason is coverage, not the
# graph: p35 makes *every* valid line a centre and mean+max pools ~25 overlapping
# 7-line views of the bundle, whereas the single-root design sees the bundle
# through one window plus two hops.
#
# This variant keeps p35's coverage exactly and changes only how the six
# neighbours are combined: p35 reduces them with a fixed mean and max, here a
# GATv2 lets the centre weight its own neighbours.  Node features stay
# role-aware (root as centre + direction, neighbours as neighbours-of-root), and
# the per-root representations are mean+max pooled as before.


@torch.no_grad()
def precompute_multiroot(geometry, seeds, counts, device, strategy="fps6", chunk=256,
                         dtype=torch.float16, derivative=True, transform="dft",
                         frequencies=FREQUENCIES):
    """Per-root blocks for every valid line.

    Returns centre_inv [B,27,23], centre_dir [B,27,72], nb [B,27,6,23], mask [B,27].
    Stored in half precision: float32 would be ~5 GB for 206k bundles.
    """
    inv, dirs, nbs, masks = [], [], [], []
    for start in range(0, len(geometry), chunk):
        stop = start + chunk
        g = torch.as_tensor(np.asarray(geometry[start:stop]), device=device)
        c = torch.as_tensor(np.asarray(counts[start:stop]), device=device, dtype=torch.long)
        s = torch.as_tensor(np.asarray(seeds[start:stop]), device=device)
        x, mask = normalize_geometry(g, c, "max_radius")
        batch, lines = x.shape[0], x.shape[1]
        index = neighbor_indices(s, c, strategy)                       # [B, 27, 6]
        roots = torch.arange(lines, device=device)[None, :, None].expand(batch, -1, -1)
        ids = torch.cat((roots, index), -1)                            # [B, 27, 7]
        primitives = x[torch.arange(batch, device=device)[:, None, None], ids].reshape(
            batch * lines, 7, x.shape[2], 3)
        if derivative and transform == "dft":
            # keep the original call byte-for-byte so existing caches stay valid
            flat = pathline_dft_features_3d(
                primitives, num_freq=frequencies, neighbor_scale=1., neighbor_weight=1.,
                neighbor_pool="none", mode="gram", include_chirality=True, return_numpy=False)
        else:
            flat = pathline_features_variant_3d(
                primitives, num_freq=frequencies, derivative=derivative,
                transform=transform, neighbor_scale=1.)
        width = variant_block_width(frequencies, transform)
        inv.append(flat[:, :width].reshape(batch, lines, width).to(dtype).cpu())
        nbs.append(flat[:, width:].reshape(batch, lines, NEIGHBOURS, width).to(dtype).cpu())
        dirs.append(direction_spectrum(x.reshape(batch * lines, x.shape[2], 3), FREQUENCIES)
                    .reshape(batch, lines, DIRECTION).to(dtype).cpu())
        masks.append(mask.cpu())
    return {"centre_inv": torch.cat(inv), "centre_dir": torch.cat(dirs),
            "nb": torch.cat(nbs), "mask": torch.cat(masks)}


def precompute_geometry(geometry, counts, device, chunk=256, dtype=torch.float16):
    """Normalised tracks for a learned descriptor: geometry [B,27,T,3], dir, mask.

    Stores the same normalised coordinates `precompute_multiroot` transforms, so
    the learned and fixed descriptors see identical input.
    """
    tracks, dirs, masks = [], [], []
    for start in range(0, len(geometry), chunk):
        g = torch.as_tensor(np.asarray(geometry[start:start + chunk]), device=device)
        c = torch.as_tensor(np.asarray(counts[start:start + chunk]), device=device,
                            dtype=torch.long)
        x, mask = normalize_geometry(g, c, "max_radius")
        batch, lines = x.shape[0], x.shape[1]
        tracks.append(x.to(dtype).cpu())
        dirs.append(direction_spectrum(x.reshape(batch * lines, x.shape[2], 3), FREQUENCIES)
                    .reshape(batch, lines, DIRECTION).to(dtype).cpu())
        masks.append(mask.cpu())
    return {"geometry": torch.cat(tracks), "centre_dir": torch.cat(dirs),
            "mask": torch.cat(masks)}


class MultiRootFMTGraph(nn.Module):
    """p35 coverage, GATv2 instead of the fixed mean+max over the six neighbours."""

    def __init__(self, representation_dim=256, heads=4, dropout=0.15,
                 use_direction=True, head_hidden=128, pool="meanmax", use_orbit=False,
                 root_dropout=0.0, use_fine=False, fine_dim=128, block_width=BLOCK):
        super().__init__()
        self.block_width = block_width
        # Fine level.  Every line is already processed as a root, so the token of
        # line j *is* the "j as its own centre, having aggregated its own 6
        # neighbours" summary the two-level design asks for.  The second hop is
        # therefore a gather on neighbour indices rather than a second
        # [B,27,6,6,23] feature tensor (which would be ~8.6 GB for this cache).
        self.use_fine = use_fine
        self.use_direction, self.pool, self.use_orbit = use_direction, pool, use_orbit
        # Train loss reaches 1e-4 while validation F1 sits near 0.91: the model
        # memorises bundles.  Hiding a random subset of the roots each step is the
        # set-model analogue of DropNode -- the bundle stays the same object, but is
        # never seen through the same 25-line view twice.
        self.root_dropout = root_dropout
        # Every root is a 3x3x3 lattice site, but FMT descriptors are computed
        # relative to the root, so the model cannot tell a cube-centre root from a
        # corner one.  The orbit of a site under O_h (centre / face / edge / corner)
        # is a group invariant -- the octahedral group permutes sites within an
        # orbit and never across -- so embedding it adds the missing radial cue
        # without breaking the octahedral symmetry the features are built on.
        if use_orbit:
            self.orbit_embed = nn.Embedding(5, representation_dim)
            nn.init.normal_(self.orbit_embed.weight, std=0.02)
        centre_in = block_width + (DIRECTION if use_direction else 0)
        self.centre_proj = block(centre_in, representation_dim, dropout)
        self.neighbour_proj = block(block_width, representation_dim, dropout)
        self.gat = GATv2(representation_dim, representation_dim, representation_dim,
                         heads, dropout)
        self.token = nn.Sequential(block(2 * representation_dim, representation_dim, dropout),
                                   block(representation_dim, representation_dim, dropout))
        if use_fine:
            self.fine_proj = nn.Linear(representation_dim, fine_dim)
            self.fine_self = nn.Linear(representation_dim, fine_dim)
            self.fine_gat = GATv2(fine_dim, fine_dim, fine_dim, heads, dropout)
            self.fine_merge = block(representation_dim + fine_dim, representation_dim, dropout)
        # The neighbour level already replaced fixed pooling with GATv2; mean+max over
        # the 27 lattice roots is the last hand-chosen aggregation left, so allow the
        # same treatment one level up.  The query is the masked mean of the root
        # tokens, which keeps the read-out invariant to root ordering.
        if pool in ("attn", "attnmax"):
            self.root_gat = GATv2(representation_dim, representation_dim,
                                  representation_dim, heads, dropout)
        width = representation_dim * (2 if pool in ("meanmax", "attnmax") else 1)
        self.fuse = block(width, representation_dim, dropout)
        self.head = nn.Sequential(block(representation_dim, head_hidden, dropout),
                                  nn.Linear(head_hidden, 2))

    def forward(self, batch):
        centre, nb, mask = batch["centre_inv"], batch["nb"], batch["mask"]
        size, lines = centre.shape[0], centre.shape[1]
        if self.training and self.root_dropout > 0:
            kept = mask & (torch.rand_like(mask, dtype=torch.float) >= self.root_dropout)
            mask = torch.where(kept.any(1, keepdim=True), kept, mask)   # never empty a bundle
        if self.use_direction:
            centre = torch.cat((centre, batch["centre_dir"]), -1)
        c = self.centre_proj(centre.reshape(size * lines, -1))
        if self.use_orbit:                       # before the GATv2, so a corner root
            c = c + self.orbit_embed(            # can attend differently from the centre
                batch["orbit"].reshape(size * lines).long())
        n = self.neighbour_proj(nb.reshape(size * lines, nb.shape[2], self.block_width))
        pooled, _ = self.gat(c, n)
        tokens = self.token(torch.cat((c, pooled), -1))
        if self.use_fine:
            grid = tokens.reshape(size, lines, -1)
            neighbour_tokens = self.fine_proj(grid)
            index = batch["nbr_index"].long()                         # [B, L, 6]
            rows = torch.arange(size, device=index.device)[:, None, None]
            gathered = neighbour_tokens[rows, index]                  # [B, L, 6, fine]
            fine, _ = self.fine_gat(
                self.fine_self(grid).reshape(size * lines, -1),
                gathered.reshape(size * lines, index.shape[2], -1))
            tokens = self.fine_merge(torch.cat((tokens, fine), -1))
        tokens = tokens.reshape(size, lines, -1)
        weight = mask[..., None].to(tokens.dtype)
        mean = (tokens * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        summary = mean
        if self.pool in ("attn", "attnmax"):
            summary, _ = self.root_gat(mean, tokens, mask)
        if self.pool in ("meanmax", "attnmax"):
            maximum = tokens.masked_fill(~mask[..., None], float("-inf")).amax(1)
            merged = torch.cat((summary, torch.nan_to_num(maximum, neginf=0.0)), -1)
        else:
            merged = summary
        return self.head(self.fuse(merged))

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"total": int(total), "backbone": int(total - head), "head": int(head)}
