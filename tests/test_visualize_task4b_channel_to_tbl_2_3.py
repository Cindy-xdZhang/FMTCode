from __future__ import annotations

import numpy as np

from experiments.Visualize_Task4B_ChannelToTBL_2_3 import (
    _array_sha256,
    labels_to_projected_image,
    project_frontmost_rows,
    true_hairpin_support_subset,
)


def _synthetic_voxels() -> np.ndarray:
    # Three rows share the same x-y ray; two rows share a y-z ray.
    return np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )


def test_projection_uses_every_row_and_selects_frontmost_depth() -> None:
    voxels = _synthetic_voxels()
    resolution = np.asarray([2, 2, 3], dtype=np.int64)
    plus_z = project_frontmost_rows(voxels, resolution, "+z")
    minus_z = project_frontmost_rows(voxels, resolution, "-z")
    assert plus_z["input_row_count"] == len(voxels)
    assert plus_z["visible_row_count"] == 3
    assert plus_z["occluded_row_count"] == 2
    assert plus_z["excluded_row_count"] == 0
    assert plus_z["all_input_rows_accounted_for"] is True
    # Row 2 is z=2 (nearest from +z); row 0 is z=0 (nearest from -z).
    assert 2 in plus_z["visible_rows"].tolist()
    assert 0 not in plus_z["visible_rows"].tolist()
    assert 0 in minus_z["visible_rows"].tolist()
    assert 2 not in minus_z["visible_rows"].tolist()


def test_all_methods_reuse_identical_visible_geometry() -> None:
    voxels = _synthetic_voxels()
    resolution = np.asarray([2, 2, 3], dtype=np.int64)
    projection = project_frontmost_rows(voxels, resolution, "+z")
    proxy = np.asarray([0, 1, 2, 3, 0], dtype=np.int8)
    model = np.asarray([3, 2, 1, 0, 3], dtype=np.int8)
    proxy_image = labels_to_projected_image(proxy, projection)
    model_image = labels_to_projected_image(model, projection)
    assert np.array_equal(proxy_image >= 0, model_image >= 0)
    assert np.count_nonzero(proxy_image >= 0) == projection["visible_row_count"]
    assert proxy_image[0, 0] == proxy[2]
    assert model_image[0, 0] == model[2]


def test_typed_array_hash_distinguishes_order_and_dtype() -> None:
    values = np.asarray([1, 2, 3], dtype=np.int32)
    assert _array_sha256(values) == _array_sha256(values.copy())
    assert _array_sha256(values) != _array_sha256(values[::-1])
    assert _array_sha256(values) != _array_sha256(values.astype(np.int64))


def test_true_hairpin_subset_uses_all_positive_ids_and_global_bbox() -> None:
    voxels = np.asarray(
        [[0, 0, 0], [2, 1, 1], [4, 3, 2], [3, 2, 1]], dtype=np.int64
    )
    data = {
        "voxels": voxels,
        "resolution": np.asarray([5, 4, 3], dtype=np.int64),
        "domain_min": np.asarray([10.0, 20.0, 30.0]),
        "domain_max": np.asarray([15.0, 24.0, 33.0]),
        "voxel_size": np.ones(3),
        "candidate_indices": np.asarray([5, 7, 9, 11], dtype=np.int64),
        "vortex_ids": np.asarray([0, 4, 7, 4], dtype=np.int32),
        "labels": {
            "proxy": np.asarray([0, 2, 3, 3], dtype=np.int8),
            "raw": np.asarray([1, 0, 3, 1], dtype=np.int8),
        },
    }
    subset = true_hairpin_support_subset(data)
    assert subset["selection_record"]["selected_row_count"] == 3
    assert subset["selection_record"]["selected_vortex_id_count"] == 2
    assert subset["selection_record"]["global_voxel_index_min_xyz_inclusive"] == [2, 1, 1]
    assert subset["selection_record"]["global_voxel_index_max_xyz_inclusive"] == [4, 3, 2]
    assert subset["resolution"].tolist() == [3, 3, 2]
    assert subset["voxels"].tolist() == [[0, 0, 0], [2, 2, 1], [1, 1, 0]]
    assert set(subset["labels"]["proxy"].tolist()) == {2, 3}
    # Model ordinary predictions are intentionally retained as visible errors.
    assert subset["labels"]["raw"].tolist() == [0, 3, 1]
