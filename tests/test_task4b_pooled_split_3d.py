"""Tests for FMT_Utils.Task4B_PooledSplit_3D (runnable with pytest or directly)."""

from __future__ import annotations

import numpy as np

from FMT_Utils.Task4B_PooledSplit_3D import (
    SPLIT_CODES,
    buffer_mask_against_test,
    nearest_instance_ids,
    size_stratified_instance_split,
    split_volume,
)


def test_block_pattern_proportions_and_uniqueness():
    ids = np.arange(1, 74)
    sizes = np.arange(73, 0, -1)
    assignment = size_stratified_instance_split(ids, sizes, seed=7068)
    merged = np.concatenate(list(assignment.values()))
    assert len(merged) == 73 and len(np.unique(merged)) == 73
    # seven full blocks: 14 test, 7 validation, 49 train, plus 3 random slots
    assert 14 <= len(assignment["test"]) <= 17
    assert 7 <= len(assignment["validation"]) <= 10
    assert 49 <= len(assignment["train"]) <= 52
    # every full size block of ten contributes exactly two test instances
    order = np.argsort(-sizes)
    for start in range(0, 70, 10):
        block_ids = set(ids[order[start:start + 10]].tolist())
        assert len(block_ids & set(assignment["test"].tolist())) == 2
        assert len(block_ids & set(assignment["validation"].tolist())) == 1


def test_split_is_deterministic_in_seed():
    ids = np.arange(1, 31)
    sizes = np.random.default_rng(0).integers(10, 500, size=30)
    first = size_stratified_instance_split(ids, sizes, seed=5)
    second = size_stratified_instance_split(ids, sizes, seed=5)
    third = size_stratified_instance_split(ids, sizes, seed=6)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert any(not np.array_equal(first[k], third[k]) for k in first)


def test_periodic_nearest_instance_wraps_seam():
    hairpin_seeds = np.asarray([[0.1, 0.5, 0.5], [2.0, 0.5, 0.5]])
    hairpin_ids = np.asarray([7, 9])
    query = np.asarray([[3.05, 0.5, 0.5]])  # near the x seam of a period-3.1 box
    nearest, distance = nearest_instance_ids(query, hairpin_seeds, hairpin_ids, periods_xy=(3.1, 1.0))
    assert nearest[0] == 7 and np.isclose(distance[0], 0.15)
    nearest_np, _ = nearest_instance_ids(query, hairpin_seeds, hairpin_ids, periods_xy=None)
    assert nearest_np[0] == 9


def test_buffer_removes_only_nontest_neighbours():
    seeds = np.asarray([[0.0, 0, 0], [0.05, 0, 0], [0.5, 0, 0], [1.0, 0, 0]], dtype=float)
    codes = np.asarray([2, 0, 0, 1])  # test, train(near), train(far), validation(far)
    keep, distance = buffer_mask_against_test(seeds, codes, buffer_distance=0.1, periods_xy=None)
    assert keep.tolist() == [True, False, True, True]
    assert np.isclose(distance[1], 0.05)


def test_split_volume_instance_purity_and_certificate():
    rng = np.random.default_rng(1)
    centers = rng.uniform(0.5, 9.5, size=(20, 3))
    seeds, ids, labels = [], [], []
    for instance, center in enumerate(centers, start=1):
        cubes = center[None, :] + rng.normal(scale=0.05, size=(30, 3))
        seeds.append(cubes)
        ids.append(np.full(30, instance))
        labels.append(np.where(rng.random(30) < 0.3, 2, 3))
    ordinary = rng.uniform(0, 10, size=(2000, 3))
    seeds.append(ordinary)
    ids.append(np.zeros(2000, dtype=int))
    labels.append(rng.integers(0, 2, size=2000))
    seeds = np.concatenate(seeds)
    ids = np.concatenate(ids)
    labels = np.concatenate(labels)
    result = split_volume(seeds, ids, labels, seed=3, buffer_distance=0.2, periods_xy=None)
    hairpin = ids > 0
    for instance in np.unique(ids[hairpin]):
        assert len(np.unique(result.split_codes[ids == instance])) == 1
    kept_nontest = result.keep_mask & (result.split_codes != SPLIT_CODES["test"])
    assert result.distance_to_test[kept_nontest].min() >= 0.2
    assert result.keep_mask[result.split_codes == SPLIT_CODES["test"]].all()
    assert result.summary["instances_per_split"]["test"] == 4


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("PASS", name)
