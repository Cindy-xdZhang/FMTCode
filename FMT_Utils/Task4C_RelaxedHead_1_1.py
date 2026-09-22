"""Seed the 3x3x3 lattice without demanding head membership of the neighbours.

`Task4C_Multiscale_4_1.center_and_neighbors` accepts a lattice seed only if it is
inside the domain, has interpolated `oyf > 0`, **and** its cell belongs to the
same connected lambda2 head component as the bundle's centre.  That last test is
responsible for 80-86% of all excluded face neighbours
(`docs/exp_Task4C_OctVAE.md` §8.1), and the curves it removes are full length
(arc fraction .999) and diverge from the centre line only 11-13% more than the
retained ones (§8.2).  It is region bookkeeping, not a quality filter.

This variant keeps every other rule and relaxes exactly that one, for neighbours
only:

* the **centre** must still lie in its own head component, so the bundle stays
  anchored to one candidate head and the label logic is untouched;
* **neighbours** must still be inside the domain with `oyf > 0`, so the physical
  sense of spanwise rotation is still enforced;
* neighbours are no longer required to share the centre's head component.

The random draws are made in the same order with the same seed sequence as the
original, so the sampled cell and the within-cell centre are **identical** to the
frozen build.  Only which neighbours survive changes, which is what makes the two
caches comparable.
"""

from __future__ import annotations

import numpy as np

from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar


def center_and_neighbors_relaxed(row, component_grid, axes, oyf, center_number, scale,
                                 base_seed):
    """Drop-in replacement for `center_and_neighbors` with neighbour head test removed."""
    rng = np.random.default_rng(
        np.random.SeedSequence([base_seed, row["head_component"], center_number]))
    cells = row["cell_ids"]
    cell = int(rng.choice(cells))
    iz, iy, ix = np.unravel_index(cell, component_grid.shape)
    spacing = np.array([axes[0][ix + 1] - axes[0][ix],
                        axes[1][iy + 1] - axes[1][iy],
                        axes[2][iz + 1] - axes[2][iz]])
    center = np.array([axes[0][ix], axes[1][iy], axes[2][iz]]) + rng.uniform(.25, .75, 3) * spacing
    offsets = np.array([[x, y, z] for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)],
                       float)
    center_index = int(np.flatnonzero((offsets == 0).all(1))[0])
    offsets[[0, center_index]] = offsets[[center_index, 0]]
    distance = float(np.prod(spacing) ** (1 / 3)) * scale["neighbor_grid_scale"]
    seeds = center + offsets * distance
    fluct, inside, cell_id = interpolate_scalar(seeds, axes, oyf)

    same_head = component_grid.ravel()[cell_id] == row["head_component"]
    valid = inside & (fluct > 0)
    valid[0] = valid[0] & same_head[0]          # the centre keeps the original rule
    if not valid[0]:
        return None
    return {"center": center, "seeds": seeds[valid], "source_cell": cell,
            "offset_grid_ids": np.flatnonzero(valid), "neighbor_distance": distance,
            "seed_rms_distance": float(np.sqrt(np.mean(np.sum((seeds[valid] - center) ** 2,
                                                              axis=1))))}
