"""Phase 0 gate for the octahedral group action on 7-line 3D primitives."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.OctahedralGroup_3D import (  # noqa: E402
    NEIGHBOUR_DIRECTIONS, apply_group, apply_norm_stats_oh, build_signal_from_raw,
    channel_balance_weights_oh, group_orbit, group_tensors, normalize_signal_oh,
    octahedral_group,
)


ROOT = Path(__file__).resolve().parents[1]


def _elements(group):
    matrices, permutation = octahedral_group(group)
    return matrices, permutation


def test_group_is_closed_and_correctly_sized():
    for group, order, proper in (("oh", 48, 24), ("o", 24, 24)):
        matrices, _ = _elements(group)
        assert len(matrices) == order
        keys = {tuple(np.asarray(m, dtype=np.int64).ravel().tolist()) for m in matrices}
        assert len(keys) == order, "group elements must be distinct"
        determinants = np.round([np.linalg.det(m) for m in matrices]).astype(int)
        assert set(determinants.tolist()) <= {-1, 1}
        assert int((determinants == 1).sum()) == proper
        # every element is a signed permutation matrix
        for matrix in matrices:
            assert np.array_equal(np.abs(matrix).sum(axis=0), np.ones(3))
            assert np.array_equal(np.abs(matrix).sum(axis=1), np.ones(3))
        # closure and inverses
        for left in matrices:
            for right in matrices:
                assert tuple((left @ right).astype(np.int64).ravel().tolist()) in keys
            assert tuple(left.T.astype(np.int64).ravel().tolist()) in keys
        assert np.array_equal(matrices[0], np.eye(3, dtype=np.float32))


def test_direction_star_is_a_fixed_point_of_the_whole_group():
    """A primitive whose neighbour channels carry their own directions is fixed.

    ``augmented[j] = M @ original[permutation[j]]`` and ``permutation`` is built
    so that ``direction(permutation[j]) = M^T direction(j)``; substituting gives
    ``M M^T direction(j) = direction(j)``.  This is the sharpest single check
    that the component rotation and the channel permutation agree -- getting
    either the gather direction or the transpose wrong breaks it.
    """
    for group in ("oh", "o"):
        matrices, permutation = group_tensors(group)
        steps = 5
        signal = torch.zeros(1, 7, steps, 3, dtype=torch.float64)
        signal[0, 1:, :, :] = torch.as_tensor(
            NEIGHBOUR_DIRECTIONS, dtype=torch.float64
        )[:, None, :].expand(-1, steps, -1)
        for index in range(len(matrices)):
            element = torch.zeros(1, dtype=torch.long) + index
            moved = apply_group(signal, element, matrices.double(), permutation)
            assert torch.equal(moved[0, 1:], signal[0, 1:]), f"{group} element {index}"


def test_action_matches_rotating_the_primitive_in_space():
    """End-to-end: rotating the geometry then re-caching equals the channel action.

    Builds real primitives whose initial stencil is the integrator's own offset
    table, rotates every point by M, re-labels the lines into the canonical
    ``x+, x-, y+, y-, z+, z-`` order that the rotated stencil now occupies, and
    checks the cached-signal path agrees with ``apply_group``.
    """
    rng = np.random.default_rng(11)
    matrices, permutation = octahedral_group("oh")
    offset, samples, count = 0.017, 32, 24

    paths = np.empty((count, 7, samples, 3), dtype=np.float64)
    drift = np.cumsum(rng.normal(0, 0.01, (count, 1, samples, 3)), axis=2)
    paths[:, 0] = drift[:, 0]
    for channel in range(6):
        wobble = np.cumsum(rng.normal(0, 0.002, (count, samples, 3)), axis=1)
        paths[:, 1 + channel] = drift[:, 0] + offset * NEIGHBOUR_DIRECTIONS[channel] + wobble
        paths[:, 1 + channel, 0] = drift[:, 0, 0] + offset * NEIGHBOUR_DIRECTIONS[channel]

    def cache(array):
        return (array - array[:, :1, :1, :]).reshape(len(array), -1)

    signal = torch.from_numpy(build_signal_from_raw(cache(paths), samples).astype(np.float64))
    torch_matrices, torch_permutation = group_tensors("oh", dtype=torch.float64)

    for index in range(len(matrices)):
        matrix, gather = matrices[index], permutation[index]
        rotated = np.empty_like(paths)
        rotated[:, 0] = paths[:, 0] @ matrix.T
        for channel in range(6):
            rotated[:, 1 + channel] = paths[:, 1 + gather[channel]] @ matrix.T
        # the rotated stencil must again be the canonical axis-aligned star
        initial = rotated[:, 1:, 0, :] - rotated[:, :1, 0, :]
        assert np.allclose(initial, offset * NEIGHBOUR_DIRECTIONS[None], atol=1e-12)

        expected = build_signal_from_raw(cache(rotated), samples).astype(np.float64)
        element = torch.zeros(len(paths), dtype=torch.long) + index
        actual = apply_group(signal, element, torch_matrices, torch_permutation)
        assert np.abs(actual.numpy() - expected).max() < 1e-12, f"element {index}"


def test_normalisation_is_exactly_equivariant():
    """The reason the 2D mean subtraction and componentwise clamp were dropped."""
    rng = np.random.default_rng(3)
    signal = torch.from_numpy(
        (rng.normal(0, 1, (64, 7, 31, 3)) + np.array([4.0, 0.3, -0.2])).astype(np.float64)
    )
    matrices, permutation = group_tensors("oh", dtype=torch.float64)
    normalized, stats = normalize_signal_oh(signal, clip_sigma=5.0)

    worst = 0.0
    for index in range(len(matrices)):
        element = torch.zeros(len(signal), dtype=torch.long) + index
        moved = apply_group(signal, element, matrices, permutation)
        # statistics are scalar RMS values, invariant under the group
        _, moved_stats = normalize_signal_oh(moved, clip_sigma=5.0)
        for before, after in zip(stats, moved_stats):
            assert torch.abs(before - after) < 1e-12
        left = apply_norm_stats_oh(moved, stats, clip_sigma=5.0)
        right = apply_group(normalized, element, matrices, permutation)
        worst = max(worst, float((left - right).abs().max()))
    assert worst < 1e-6, f"normalisation equivariance defect {worst:g}"


def test_clipping_actually_engages_and_stays_equivariant():
    """Guard against the equivariance test passing only because nothing clipped."""
    rng = np.random.default_rng(5)
    signal = torch.from_numpy(rng.normal(0, 1, (32, 7, 31, 3)).astype(np.float64))
    signal[0, 3, 4] = torch.tensor([90.0, 60.0, 30.0], dtype=torch.float64)
    matrices, permutation = group_tensors("oh", dtype=torch.float64)
    normalized, stats = normalize_signal_oh(signal, clip_sigma=5.0)
    unclipped = torch.cat(
        (signal[:, :1] / stats[0], signal[:, 1:] / stats[1]), dim=1
    )
    assert float((unclipped - normalized).abs().max()) > 1.0, "clipping never engaged"

    for index in range(len(matrices)):
        element = torch.zeros(len(signal), dtype=torch.long) + index
        left = apply_norm_stats_oh(
            apply_group(signal, element, matrices, permutation), stats, 5.0
        )
        right = apply_group(normalized, element, matrices, permutation)
        assert float((left - right).abs().max()) < 1e-9


def test_channel_balance_weights_are_group_invariant():
    rng = np.random.default_rng(7)
    signal = torch.from_numpy(rng.normal(0, 1, (48, 7, 31, 3)).astype(np.float64))
    signal[:, 0] *= 0.05          # make the centre genuinely different
    matrices, permutation = group_tensors("oh", dtype=torch.float64)
    weights = channel_balance_weights_oh(signal)
    assert torch.abs(weights.mean() - 1.0) < 1e-9
    assert torch.allclose(weights[0, 1:], weights[0, 1].expand(6, 1, 1))
    for index in range(len(matrices)):
        element = torch.zeros(len(signal), dtype=torch.long) + index
        moved = channel_balance_weights_oh(
            apply_group(signal, element, matrices, permutation)
        )
        assert float((moved - weights).abs().max()) < 1e-9


def test_group_orbit_enumerates_every_element_once():
    """`group_orbit` is the reference iterator the batched readout must match."""
    rng = np.random.default_rng(17)
    signal = torch.from_numpy(rng.normal(0, 1, (8, 7, 31, 3)).astype(np.float64))
    matrices, permutation = group_tensors("oh", dtype=torch.float64)
    orbit = list(group_orbit(signal, matrices, permutation))
    assert len(orbit) == len(matrices)
    for index, moved in enumerate(orbit):
        element = torch.zeros(len(signal), dtype=torch.long) + index
        assert torch.equal(moved, apply_group(signal, element, matrices, permutation))
    # the identity must appear exactly once, and the orbit must not repeat
    keys = {moved.numpy().tobytes() for moved in orbit}
    assert len(keys) == len(matrices)
    assert torch.equal(orbit[0], signal)


def test_signal_builder_matches_direct_recomputation():
    rng = np.random.default_rng(13)
    paths = rng.normal(0, 1, (16, 7, 32, 3)).astype(np.float32)
    raw = (paths - paths[:, :1, :1, :]).reshape(len(paths), -1)
    signal = build_signal_from_raw(raw, 32)
    assert signal.shape == (16, 7, 31, 3)
    centre = np.diff(paths[:, 0], axis=1)
    assert np.abs(signal[:, 0] - centre).max() < 1e-5
    for channel in range(6):
        relative = np.diff(paths[:, 1 + channel] - paths[:, 0], axis=1)
        assert np.abs(signal[:, 1 + channel] - relative).max() < 1e-5
    try:
        build_signal_from_raw(raw[:, :-3], 32)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a shape error on truncated raw features")


def test_runs_on_real_cached_primitives():
    """Only asserts if a Task1 cache is present; skips quietly otherwise."""
    caches = sorted(ROOT.glob("outputs/exp_Task1_*/development_cache/*/slice_*.npz"))
    if not caches:
        print("  (no Task1 cache found; real-data check skipped)")
        return
    record = np.load(caches[0])
    signal = build_signal_from_raw(record["raw_features"], 32)
    assert signal.shape == (len(record["raw_features"]), 7, 31, 3)
    assert np.isfinite(signal).all()
    tensor = torch.from_numpy(signal[:256].astype(np.float64))
    matrices, permutation = group_tensors("oh", dtype=torch.float64)
    normalized, stats = normalize_signal_oh(tensor)
    worst = 0.0
    for index in range(len(matrices)):
        element = torch.zeros(len(tensor), dtype=torch.long) + index
        left = apply_norm_stats_oh(
            apply_group(tensor, element, matrices, permutation), stats
        )
        right = apply_group(normalized, element, matrices, permutation)
        worst = max(worst, float((left - right).abs().max()))
    assert worst < 1e-6, f"real-data equivariance defect {worst:g}"
    print(f"  real cache {caches[0].parent.name}/{caches[0].name}: defect {worst:.2e}")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"  {name} ok")
    print("OCTAHEDRAL GROUP 3D TEST PASSED")
