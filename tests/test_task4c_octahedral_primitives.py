"""Gate for the lattice-exact Task4-c octahedral primitives.

The claim under test is the one that version 1.1 could not make: because the
Task4-c bundles are seeded on a 3x3x3 cubic lattice, rotating the underlying
geometry permutes the six face neighbours exactly as it does in Task1, so
`OctahedralGroup_3D.apply_group` -- rotation *and* channel permutation -- is the
correct action on these primitives.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.OctahedralGroup_3D import (  # noqa: E402
    NEIGHBOUR_DIRECTIONS, apply_group, group_tensors, octahedral_group,
)
from FMT_Utils.Task4C_OctVAE_2_1 import (  # noqa: E402
    FACE_ORDER, PRIMITIVE_ORDER, build_primitives, lattice_codes,
    octahedral_selection,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/exp_Task4C_LocalRebuild_1.0/physical/channel/validation"


def _load():
    seeds = np.load(CACHE / "seeds.npy")
    geometry = np.load(CACHE / "geometry.npy")
    with np.load(CACHE / "metadata.npz") as data:
        meta = {key: data[key] for key in data.files}
    return geometry, seeds, meta


def test_channel_order_matches_task1():
    assert tuple(map(tuple, NEIGHBOUR_DIRECTIONS)) == FACE_ORDER
    assert PRIMITIVE_ORDER[0] == (0, 0, 0) and len(PRIMITIVE_ORDER) == 7


def test_every_retained_line_sits_on_the_lattice():
    if not CACHE.exists():
        print("  (no local Task4C rebuild; skipped)"); return
    _, seeds, meta = _load()
    codes, valid = lattice_codes(seeds, meta)
    count = np.asarray(meta["counts"]).astype(int)
    # every line inside `counts` must be recognised as a lattice site
    within = np.arange(seeds.shape[1])[None, :] < count[:, None]
    assert valid.sum() == within.sum(), "some retained line is off the 3x3x3 lattice"
    assert set(np.unique(codes[valid])) <= {-1, 0, 1}
    # sites are unique within a bundle
    for row in range(len(count)):
        sites = [tuple(v) for v in codes[row][valid[row]]]
        assert len(sites) == len(set(sites)), f"duplicate lattice site in bundle {row}"


def test_selection_is_consistent_with_completeness():
    if not CACHE.exists():
        print("  (no local Task4C rebuild; skipped)"); return
    _, seeds, meta = _load()
    select, usable = octahedral_selection(seeds, meta)
    codes, valid = lattice_codes(seeds, meta)
    for row in np.flatnonzero(usable)[:200]:
        for slot, site in enumerate(PRIMITIVE_ORDER):
            picked = select[row, slot]
            assert valid[row, picked]
            assert tuple(codes[row, picked]) == site
    assert usable.sum() == sum(
        all(any((codes[r][valid[r]] == np.asarray(s, np.int8)).all(-1))
            for s in PRIMITIVE_ORDER) for r in range(len(usable)))


def test_rotating_the_geometry_equals_the_group_action():
    """The property version 1.1 lacked: a real channel permutation.

    Rotate the bundle in space -- geometry, seeds, centroid and seeding centre --
    then rebuild the primitive from scratch.  The lattice site (1,0,0) becomes
    M(1,0,0), so the rebuild reorders the neighbour channels by itself.  That must
    agree with applying `apply_group` to the original primitive.
    """
    if not CACHE.exists():
        print("  (no local Task4C rebuild; skipped)"); return
    geometry, seeds, meta = _load()
    select, usable = octahedral_selection(seeds, meta)
    rows = np.flatnonzero(usable)[:64]
    subset = {key: value[rows] for key, value in meta.items()}
    base, _ = build_primitives(geometry[rows], seeds[rows], subset)
    base = base.double()

    matrices, permutation = octahedral_group("oh")
    torch_matrices, torch_permutation = group_tensors("oh", dtype=torch.float64)
    worst = 0.0
    for index, matrix in enumerate(matrices):
        moved = {key: value.copy() for key, value in subset.items()}
        moved["centroid"] = subset["centroid"] @ matrix.T
        moved["center"] = subset["center"] @ matrix.T
        rotated_geometry = geometry[rows] @ matrix.T
        rotated_seeds = seeds[rows] @ matrix.T
        rebuilt, kept = build_primitives(rotated_geometry, rotated_seeds, moved)
        assert len(kept) == len(rows), "rotation must not change which bundles are usable"
        element = torch.zeros(len(rows), dtype=torch.long) + index
        expected = apply_group(base, element, torch_matrices, torch_permutation)
        worst = max(worst, float((rebuilt.double() - expected).abs().max()))
    assert worst < 1e-5, f"lattice action disagrees with apply_group by {worst:g}"
    print(f"  rotation-vs-action agreement over all 48 elements: max |diff| {worst:.2e}")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function(); print(f"  {name} ok")
    print("TASK4C OCTAHEDRAL PRIMITIVE TEST PASSED")
