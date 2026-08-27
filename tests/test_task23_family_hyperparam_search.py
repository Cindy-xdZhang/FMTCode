import json

import numpy as np
import pytest

from FMT_Utils.Task12Data_3D import load_cache_records
from FMT_Utils.FMT_3D_pipeline import generate_seeding_grid_3d
from Search_Task2_FMTVAE_3D import (
    _decode_job as decode_task2_job,
    _load_spec as load_task2_spec,
    _selection_key as task2_selection_key,
)
from Search_Task2_FMTVAE_Stage2_3D import _decode_job as decode_task2_stage2_job
from Search_Task3_FMTResidual_3D import (
    _decode_job as decode_task3_job,
    _load_spec as load_task3_spec,
    _selection_key as task3_selection_key,
)
from Search_Task3_FMTResidual_Stage2_3D import (
    _decode_job as decode_task3_stage2_job,
)
from Build_Task23_FamilySearch_Confirmation import SETTINGS, SEED_GRID_PHASE


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


def test_stage2_arrays_expand_only_three_features_per_family():
    task2 = load_task2_spec(TASK2_CONFIG)
    assert len(task2["stage2_architectures"]) == 12
    assert decode_task2_stage2_job(task2, "raw", 119) == (
        "smokeBuoyancy", 11, None
    )
    assert decode_task2_stage2_job(task2, "fmt", 359) == (
        "smokeBuoyancy", 11, 2
    )
    task3 = load_task3_spec(TASK3_CONFIG)
    assert len(task3["stage2_networks"]) == 10
    assert decode_task3_stage2_job(task3, 299) == ("smokeBuoyancy", 29)


def test_task2_selection_maximizes_same_vae_gain_not_task1_diagnostic():
    high_gain = {
        "fmt_minus_raw_f1_macro": 0.16,
        "worst_seed_f1_gain": 0.12,
        "fmt_f1_macro": 0.60,
        "absolute_fmt_guard_passed": False,
    }
    low_gain = {
        "fmt_minus_raw_f1_macro": 0.03,
        "worst_seed_f1_gain": 0.02,
        "fmt_f1_macro": 0.80,
        "absolute_fmt_guard_passed": True,
    }
    assert task2_selection_key(high_gain) > task2_selection_key(low_gain)


def test_task3_selection_maximizes_same_structure_raw_pca_gain():
    high_gain = {
        "fmt_minus_raw_pca_f1_macro": 0.17,
        "fmt_minus_raw_pca_ap_macro": 0.11,
        "worst_seed_f1_gain": 0.10,
        "fmt_f1_macro": 0.65,
        "strong_raw_guard_passed": False,
    }
    low_gain = {
        "fmt_minus_raw_pca_f1_macro": 0.04,
        "fmt_minus_raw_pca_ap_macro": 0.03,
        "worst_seed_f1_gain": 0.02,
        "fmt_f1_macro": 0.82,
        "strong_raw_guard_passed": True,
    }
    assert task3_selection_key(high_gain) > task3_selection_key(low_gain)


def test_phased_confirmation_grid_preserves_count_and_changes_every_axis():
    field = type("Field", (), {
        "domainMinBoundary": np.asarray([0.0, -1.0, 2.0]),
        "domainMaxBoundary": np.asarray([4.0, 3.0, 8.0]),
    })()
    regular, regular_axes = generate_seeding_grid_3d(
        field, [4, 3, 2], 0.1, 0.01
    )
    phased, phased_axes = generate_seeding_grid_3d(
        field, [4, 3, 2], 0.1, 0.01, grid_phase=[0.31, -0.23, 0.17]
    )
    assert regular.shape == phased.shape == (24, 3)
    assert all(not np.array_equal(a, b) for a, b in zip(regular_axes, phased_axes))
    assert not np.any(np.all(regular[:, None, :] == phased[None, :, :], axis=-1))
    with pytest.raises(ValueError, match="grid_phase"):
        generate_seeding_grid_3d(field, [4, 3, 2], 0.1, 0.01,
                                 grid_phase=[0.5, 0.0, 0.0])


def test_confirmation_start_indices_leave_the_full_pathline_window_available():
    source_frames = {
        "cylinder3d": 151, "halfcylinderRe640": 76,
        "halfcylinderRe6400": 151, "tangaroa": 201,
        "deltaWing_resampled": 171, "deltaWing_LBM": 234,
        "f22raptor": 159, "channel": 159, "boeing747": 199,
        "smokeBuoyancy": 160,
    }
    frame_count = 14
    assert all(abs(value) < 0.5 for value in SEED_GRID_PHASE)
    for settings in SETTINGS.values():
        for dataset, starts in settings["indices"].items():
            assert len(starts) == len(set(starts)) == 4
            assert min(starts) >= int(np.floor(0.2 * source_frames[dataset]))
            assert max(starts) + frame_count <= source_frames[dataset]
