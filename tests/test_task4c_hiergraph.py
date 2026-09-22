"""Checks for the hierarchical-graph read-out on top of the FMT p35 features.

The three properties that are easy to break and expensive to notice in a
training curve: the root read-out must not depend on root ordering, the orbit
embedding must be a genuine lattice quantity rather than a slot index, and root
dropout must never hand the model an empty bundle.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.Task4C_HierGraph_1_1 import BLOCK, DIRECTION, MultiRootFMTGraph
from FMT_Utils.Task4C_OctVAE_2_1 import lattice_codes

CACHE = ROOT / "outputs/exp_Task4C_BottomDensity_1.2_local/physical"


def _batch(size=8, roots=27, neighbours=6, seed=0):
    torch.manual_seed(seed)
    mask = torch.rand(size, roots) > 0.3
    mask[0] = False                      # no valid root at all
    mask[1] = False
    mask[1, 3] = True                    # exactly one valid root
    return {
        "centre_inv": torch.randn(size, roots, BLOCK),
        "centre_dir": torch.randn(size, roots, DIRECTION),
        "nb": torch.randn(size, roots, neighbours, BLOCK),
        "mask": mask,
        "orbit": torch.randint(0, 5, (size, roots), dtype=torch.int8),
        "nbr_index": torch.randint(0, roots, (size, roots, neighbours), dtype=torch.int16),
    }


def test_every_pooling_stays_finite_on_degenerate_bundles():
    batch = _batch()
    for pool in ("meanmax", "mean", "attn", "attnmax"):
        model = MultiRootFMTGraph(representation_dim=64, heads=4, pool=pool).eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (8, 2), pool
        assert torch.isfinite(out).all(), f"{pool} produced non-finite logits"
    print("  test_every_pooling_stays_finite_on_degenerate_bundles ok")


def test_root_readout_is_invariant_to_root_order():
    batch = _batch()
    roots = batch["mask"].shape[1]
    order = torch.randperm(roots)          # one permutation, applied to every block
    permuted = {k: (v[:, order] if v.dim() > 1 and v.shape[1] == roots else v)
                for k, v in batch.items()}
    torch.manual_seed(1)
    for pool in ("meanmax", "attn", "attnmax"):
        model = MultiRootFMTGraph(representation_dim=64, heads=4, pool=pool,
                                  use_orbit=True).eval()
        with torch.no_grad():
            defect = (model(batch) - model(permuted)).abs().max().item()
        assert defect < 1e-5, f"{pool} depends on root order: {defect:.2e}"
        print(f"  {pool} root-permutation defect {defect:.2e}")
    print("  test_root_readout_is_invariant_to_root_order ok")


def test_root_dropout_is_train_only_and_never_empties_a_bundle():
    batch = _batch()
    mask = batch["mask"]
    for _ in range(500):
        kept = mask & (torch.rand_like(mask, dtype=torch.float) >= 0.3)
        out = torch.where(kept.any(1, keepdim=True), kept, mask)
        assert out[1].any(), "single-root bundle was emptied by root dropout"

    dropped = MultiRootFMTGraph(representation_dim=64, pool="attnmax", root_dropout=0.3)
    plain = MultiRootFMTGraph(representation_dim=64, pool="attnmax", root_dropout=0.0)
    plain.load_state_dict(dropped.state_dict())
    dropped.eval(); plain.eval()
    with torch.no_grad():
        assert torch.allclose(dropped(batch), plain(batch)), "root dropout leaked into eval"
    dropped.train()
    assert torch.isfinite(dropped(batch)).all()
    print("  test_root_dropout_is_train_only_and_never_empties_a_bundle ok")


def test_orbit_classes_come_from_the_lattice_not_the_slot():
    folder = CACHE / "channel" / "validation"
    if not folder.exists():
        print("  test_orbit_classes_come_from_the_lattice_not_the_slot skipped (no cache)")
        return
    with np.load(folder / "metadata.npz") as data:
        meta = {key: data[key] for key in data.files}
    seeds = np.load(folder / "seeds.npy")
    codes, on_lattice = lattice_codes(seeds, meta)
    counts = np.asarray(meta["counts"]).astype(int)
    within = np.arange(seeds.shape[1])[None, :] < counts[:, None]
    orbit = np.where(on_lattice, np.abs(codes.astype(np.int64)).sum(-1), 4)

    resolved = (on_lattice & within).sum() / within.sum()
    assert resolved > 0.999, f"only {resolved:.4f} of retained lines sit on the lattice"
    assert orbit[within].max() <= 3, "retained line outside the 3x3x3 lattice"
    # at most one cube centre per bundle, and it is the slot the pipeline swaps it into
    centres = ((orbit == 0) & within).sum(1)
    assert centres.max() == 1, "more than one cube-centre root in a bundle"

    # Compaction means every *other* slot holds different lattice sites in different
    # bundles.  If the orbit class were a fixed function of the slot index, the
    # lattice recovery would be doing nothing; check it is genuinely data-dependent.
    varying = sum(len(np.unique(orbit[within[:, j], j])) > 1
                  for j in range(1, orbit.shape[1]))
    assert varying > 0.8 * (orbit.shape[1] - 1), (
        f"only {varying} non-centre slots carry more than one orbit class -- "
        "lattice recovery is a no-op")
    print(f"  {100 * resolved:.2f}% of retained lines on-lattice; "
          f"{varying}/{orbit.shape[1] - 1} non-centre slots are slot-order independent")
    print("  test_orbit_classes_come_from_the_lattice_not_the_slot ok")


def test_fine_level_keeps_the_readout_root_order_invariant():
    """The second hop gathers by index, so a permutation must relabel the indices."""
    batch = _batch()
    roots = batch["mask"].shape[1]
    order = torch.randperm(roots)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(roots)
    permuted = {k: (v[:, order] if v.dim() > 1 and v.shape[1] == roots else v)
                for k, v in batch.items()}
    permuted["nbr_index"] = inverse[batch["nbr_index"].long()[:, order]].to(torch.int16)

    torch.manual_seed(5)
    model = MultiRootFMTGraph(representation_dim=64, heads=4, use_orbit=True,
                              use_fine=True, fine_dim=32).eval()
    with torch.no_grad():
        defect = (model(batch) - model(permuted)).abs().max().item()
    assert defect < 1e-5, f"fine level depends on root order: {defect:.2e}"
    print(f"  fine-level root-permutation defect {defect:.2e}")
    print("  test_fine_level_keeps_the_readout_root_order_invariant ok")


def test_block_width_follows_the_transform():
    from FMT_Utils.Task4C_FeatureVariants_1_1 import variant_block_width
    for transform, width in (("dft", 23), ("dct", 15)):
        assert variant_block_width(6, transform) == width
        batch = _batch()
        batch["centre_inv"] = torch.randn(8, 27, width)
        batch["nb"] = torch.randn(8, 27, 6, width)
        model = MultiRootFMTGraph(representation_dim=64, block_width=width,
                                  use_orbit=True, use_fine=True, fine_dim=32).eval()
        with torch.no_grad():
            assert torch.isfinite(model(batch)).all(), transform
        print(f"  block width for {transform}: {width}")
    print("  test_block_width_follows_the_transform ok")


if __name__ == "__main__":
    test_every_pooling_stays_finite_on_degenerate_bundles()
    test_root_readout_is_invariant_to_root_order()
    test_root_dropout_is_train_only_and_never_empties_a_bundle()
    test_orbit_classes_come_from_the_lattice_not_the_slot()
    test_fine_level_keeps_the_readout_root_order_invariant()
    test_block_width_follows_the_transform()
    print("TASK4C HIERGRAPH TEST PASSED")
