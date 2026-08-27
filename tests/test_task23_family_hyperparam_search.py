import json

import numpy as np
import pytest

from FMT_Utils.Task12Data_3D import load_cache_records
from Search_Task2_FMTVAE_3D import (
    _decode_job as decode_task2_job,
    _load_spec as load_task2_spec,
)
from Search_Task3_FMTResidual_3D import (
    _decode_job as decode_task3_job,
    _load_spec as load_task3_spec,
)


TASK2_CONFIG = "config/Verify_Task2_FMTVAEFamilySearch_4.1.yaml"
TASK3_CONFIG = "config/Verify_Task3_FMTResidualFamilySearch_4.1.yaml"


def _write_cache(path, value):
    raw = np.full((2, 7 * 4 * 3), value, dtype=np.float32)
    np.savez_compressed(
        path,
        raw_features=raw,
        fmt_features=np.full((2, 161), value, dtype=np.float32),
        reference=np.asarray([False, True]),
        metadata_json=np.asarray(json.dumps({"value": value})),
    )


def test_selected_cache_loader_does_not_open_outer_files(tmp_path):
    _write_cache(tmp_path / "slice_00.npz", 0.0)
    _write_cache(tmp_path / "slice_01.npz", 1.0)
    (tmp_path / "slice_02.npz").write_bytes(b"sealed-not-an-npz")

    records = load_cache_records(
        tmp_path, expected_count=3, ordinals=[0, 1]
    )

    assert [record["ordinal"] for record in records] == [0, 1]
    assert [record["metadata"]["value"] for record in records] == [0.0, 1.0]
    with pytest.raises(Exception):
        load_cache_records(tmp_path, expected_count=3)


def test_task2_stage1_array_mapping_covers_exact_cartesian_product():
    spec = load_task2_spec(TASK2_CONFIG)
    assert len(spec["datasets"]) == 10
    assert len(spec["features"]) == 14
    assert len(spec["architectures"]) == 4
    assert decode_task2_job(spec, "raw", 0) == ("channel", 0, None)
    assert decode_task2_job(spec, "raw", 39) == ("smokeBuoyancy", 3, None)
    assert decode_task2_job(spec, "fmt", 0) == ("channel", 0, 0)
    assert decode_task2_job(spec, "fmt", 559) == ("smokeBuoyancy", 3, 13)
    with pytest.raises(IndexError):
        decode_task2_job(spec, "fmt", 560)


def test_task3_stage1_array_mapping_and_split_are_frozen():
    spec = load_task3_spec(TASK3_CONFIG)
    assert len(spec["datasets"]) == 10
    assert len(spec["candidates"]) == 14
    assert decode_task3_job(spec, 0) == ("channel", 0)
    assert decode_task3_job(spec, 139) == ("smokeBuoyancy", 13)
    assert set(spec["screen_split"]["train_ordinals"]) == set(range(6))
    assert spec["screen_split"]["validation_ordinals"] == [6, 7]
    assert spec["outer_ordinals"] == [8, 9]
    with pytest.raises(IndexError):
        decode_task3_job(spec, 140)

