"""Lattice-exact octahedral primitives for Task4-c, replacing FPS neighbours.

Version 1.1 picked the six neighbours of each centre by farthest-point sampling,
copying c156.  That throws away the structure the data already has.  The Task4-c
bundles are **not** an unordered cloud of lines: `Task4C_Multiscale_4_1.
center_and_neighbors` seeds them on a regular 3x3x3 cubic lattice,

    offsets = [[x, y, z] for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)]
    seeds   = centre + offsets * distance,   distance = (prod(spacing)**(1/3)) * s

with `s` in {0.25, 0.5, 1.0} and the centre swapped into slot 0.  The six
**face-adjacent** sites are therefore exactly Task1's octahedral star
`+-h e_x, +-h e_y, +-h e_z`, at a single common distance from the centre.

That matters for equivariance.  With FPS neighbours a rotation induces the
identity channel permutation, so only the component rotation survives and the
group is a weak constraint.  On the lattice the full Task1 action applies: a
group element rotates the components **and permutes the six neighbour channels**,
because it maps the axis directions onto each other.  `O_h` is the exact
symmetry group of the seeding template, so
`FMT_Utils/OctahedralGroup_3D.apply_group` can be used unchanged.

The cost is coverage.  A line is dropped if its seed left the candidate head or
had non-positive `oyf`, and again if the integrated curve failed cleaning.  A
primitive is only usable when the centre **and all six faces** survive; measured
on the 40,000-bundle local rebuild only **30.4%** qualify, and the rate is set
almost entirely by the offset scale (61.7% at 0.25h, 24.3% at 0.5h, 5.2% at 1h).
Per the design decision, an incomplete primitive is excluded from both training
and prediction rather than back-filled.

Note one consequence that must be carried into any result: excluding bundles
changes the evaluation set, so scores computed this way are **not** comparable to
c156 or any handoff table row, which are measured on all 10,000 test bundles.

A closed-orbit alternative, not implemented here: the full 27-site cube is also
`O_h`-closed (1 centre + 6 faces + 12 edges + 8 corners, each orbit closed), so a
27-channel primitive with a 27-way permutation would keep the dropped 70% at the
price of a much wider input and a padding-aware action.
"""

from __future__ import annotations

import numpy as np
import torch

# Canonical order, identical to FMT_Utils.OctahedralGroup_3D.NEIGHBOUR_DIRECTIONS
# so that module's channel permutation applies to these primitives unchanged.
FACE_ORDER = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
CENTRE = (0, 0, 0)
PRIMITIVE_ORDER = (CENTRE,) + FACE_ORDER
LINES_PER_PRIMITIVE = len(PRIMITIVE_ORDER)
LATTICE_TOLERANCE = 1e-3


def lattice_codes(seeds, meta, tolerance=LATTICE_TOLERANCE):
    """Recover each retained line's 3x3x3 lattice offset.

    The cache compacts surviving lines to the front and zero-pads, so the seeding
    slot is not stored.  It is recoverable exactly, because the seed positions are
    the lattice itself:

        physical_seed = seeds * radius + centroid
        offset        = (physical_seed - centre) / neighbour_distance

    Returns ``codes`` ``[N, 27, 3]`` int8 with ``127`` marking padding or any row
    that is not within ``tolerance`` of a lattice site, and ``valid`` ``[N, 27]``.
    """
    seeds = np.asarray(seeds, dtype=np.float64)
    count = np.asarray(meta["counts"]).astype(int)
    physical = seeds * meta["radius"][:, None, None] + meta["centroid"][:, None, :]
    offset = (physical - meta["center"][:, None, :]) / meta["neighbor_distance"][:, None, None]
    rounded = np.rint(offset)
    on_lattice = (np.abs(offset - rounded) <= tolerance).all(axis=-1)
    on_lattice &= (np.abs(rounded) <= 1).all(axis=-1)
    within = np.arange(seeds.shape[1])[None, :] < count[:, None]
    valid = on_lattice & within
    codes = np.where(valid[..., None], rounded, 127).astype(np.int8)
    return codes, valid


def octahedral_selection(seeds, meta, tolerance=LATTICE_TOLERANCE):
    """Row index of each required lattice site, and which bundles are complete.

    Returns ``select`` ``[N, 7]`` (row indices into the padded line axis, in
    `PRIMITIVE_ORDER`; ``-1`` where the site is missing) and ``usable`` ``[N]``.
    """
    codes, valid = lattice_codes(seeds, meta, tolerance)
    bundles, lines = valid.shape
    select = np.full((bundles, LINES_PER_PRIMITIVE), -1, dtype=np.int64)
    for slot, site in enumerate(PRIMITIVE_ORDER):
        match = valid & (codes == np.asarray(site, dtype=np.int8)).all(axis=-1)
        any_match = match.any(axis=1)
        select[any_match, slot] = match.argmax(axis=1)[any_match]
    usable = (select >= 0).all(axis=1)
    return select, usable


def completeness_report(seeds, meta, tolerance=LATTICE_TOLERANCE):
    """Per-bundle diagnostics for how much of the star survived."""
    codes, valid = lattice_codes(seeds, meta, tolerance)
    select, usable = octahedral_selection(seeds, meta, tolerance)
    faces_present = (select[:, 1:] >= 0).sum(axis=1)
    return {
        "bundles": int(len(usable)),
        "usable": int(usable.sum()),
        "ratio": float(usable.mean()),
        "centre_missing": int((select[:, 0] < 0).sum()),
        "faces_present_mean": float(faces_present.mean()),
        "missing_histogram": np.bincount(6 - faces_present, minlength=7).tolist(),
        "valid_lines_mean": float(np.asarray(meta["counts"]).mean()),
        "off_lattice_rows": int((~codes.astype(np.int16).__eq__(127).all(-1) & ~valid).sum()),
        "usable_mask": usable,
    }


def build_primitives(geometry, seeds, meta, select=None, usable=None,
                     tolerance=LATTICE_TOLERANCE, device="cpu"):
    """Assemble ``[M, 7, T, 3]`` octahedral primitives for the usable bundles.

    Channel 0 is the centre line about its own centroid; channels 1-6 are the
    face neighbour minus the centre at the same arclength index, in
    `PRIMITIVE_ORDER`.  Both are translation-invariant and rotate as vectors, and
    the channel order matches `OctahedralGroup_3D`, so the full Task1 group
    action -- rotation **and** channel permutation -- applies directly.
    """
    if select is None or usable is None:
        select, usable = octahedral_selection(seeds, meta, tolerance)
    rows = np.flatnonzero(usable)
    geometry = torch.as_tensor(np.asarray(geometry)[rows], dtype=torch.float32, device=device)
    picks = torch.as_tensor(select[rows], dtype=torch.long, device=device)
    steps = geometry.shape[2]
    gather = picks[:, :, None, None].expand(-1, -1, steps, 3)
    star = torch.gather(geometry, 1, gather)                       # [M, 7, T, 3]
    centre = star[:, :1]
    signal = torch.cat((centre - centre.mean(dim=2, keepdim=True), star[:, 1:] - centre), dim=1)
    return signal, rows
