import hashlib
import json
from pathlib import Path

import numpy as np

from Build_Task3_AnchoredRobust_Confirmation_5_1 import (
    SETTINGS as SETTINGS_5_1,
    SEED_GRID_PHASE as PHASE_5_1,
)
from Build_Task3_SpatialRobust_Confirmation_5_2 import (
    SETTINGS as SETTINGS_5_2,
    SEED_GRID_PHASE as PHASE_5_2,
    _jobs,
)
from FMT_Utils.Task12Data_3D import feature_matrix
from Search_Task3_FMTResidual_3D import (
    _decode_job,
    _load_search_splits,
    _load_spec,
)
from Search_Task3_FMTResidual_Stage2_3D import _decode_job as decode_stage2
from Verify_Task3_FMTClassifier import _normalize_train_only


CONFIG = "config/Verify_Task3_SpatialRobust_5.2.yaml"


def _radical_inverse(index, base):
    value, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def _write_population(source_root, label_root, count, value_offset=0.0):
    dataset = "flow"
    source_dir = Path(source_root) / dataset
    label_dir = Path(label_root) / dataset
    source_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for ordinal in range(count):
        name = f"slice_{ordinal:02d}.npz"
        raw = np.full(
            (2, 7 * 4 * 3), value_offset + ordinal, dtype=np.float32
        )
        reference = np.asarray([False, True])
        np.savez_compressed(
            source_dir / name,
            raw_features=raw,
            fmt_features=np.full((2, 8), ordinal, dtype=np.float32),
            reference=reference,
            metadata_json=np.asarray(json.dumps({"ordinal": ordinal})),
        )
        np.savez_compressed(
            label_dir / name,
            labels=reference.astype(np.float32),
            metadata_json=np.asarray(json.dumps({
                "source_cache": name,
                "label_value": 95.0,
            })),
        )


def test_5_2_config_cartesian_products_and_exposed_populations_are_frozen():
    spec = _load_spec(CONFIG)
    assert len(spec["candidates"]) == 30
    assert len(spec["stage2_networks"]) == 18
    assert spec["selection"]["stage2_top_k"] == 4
    assert _decode_job(spec, 0) == ("channel", 0)
    assert _decode_job(spec, 299) == ("smokeBuoyancy", 29)
    assert decode_stage2(spec, 719) == ("smokeBuoyancy", 71)
    assert spec["exposed_training"]["source_experiment"] == (
        "mainExp_Task3_3D_4.1"
    )
    assert spec["robust_validation"]["source_experiment"] == (
        "mainExp_Task3_3D_5.1"
    )
    assert spec["exposed_training"]["seed_grid_phase"] != (
        spec["robust_validation"]["seed_grid_phase"]
    )


def test_5_2_search_appends_only_declared_exposed_populations(tmp_path):
    roots = {
        key: tmp_path / key
        for key in ("base", "base_labels", "train", "train_labels",
                    "validation", "validation_labels")
    }
    _write_population(roots["base"], roots["base_labels"], 10, 0.0)
    _write_population(roots["train"], roots["train_labels"], 4, 20.0)
    _write_population(
        roots["validation"], roots["validation_labels"], 4, 40.0
    )
    spec = {
        "expected_slices": 10,
        "expected_ivd_percentile": 95.0,
        "require_source_reference_match": True,
        "datasets": ["flow"],
        "groups": {"family": {
            "datasets": ["flow"],
            "source_cache_root": str(roots["base"]),
            "label_cache_root": str(roots["base_labels"]),
            "exposed_training_source_cache_root": str(roots["train"]),
            "exposed_training_label_cache_root": str(roots["train_labels"]),
            "exposed_spatial_source_cache_root": str(roots["validation"]),
            "exposed_spatial_label_cache_root": str(roots["validation_labels"]),
        }},
        "screen_split": {
            "train_ordinals": list(range(6)),
            "validation_ordinals": list(range(6, 10)),
        },
        "exposed_training": {"expected_slices": 4, "ordinals": list(range(4))},
        "robust_validation": {"expected_slices": 4, "ordinals": list(range(4))},
    }
    train, validation = _load_search_splits(
        spec, "flow", {"fmt_feature": "fmt_all"}, "cpu"
    )
    assert train[0].shape == (20, 7, 4, 3)
    assert validation[0].shape == (16, 7, 4, 3)
    assert np.all(train[0][-2:] == 23.0)
    assert np.all(validation[0][-2:] == 43.0)


def test_frozen_raw_normalization_is_not_recomputed_after_augmentation():
    train = (
        np.full((3, 7, 4, 3), 100.0, dtype=np.float32),
        np.arange(24, dtype=np.float32).reshape(3, 8),
        np.asarray([0, 1, 0], dtype=np.float32),
    )
    validation = tuple(value.copy() for value in train)
    frozen = {
        "raw_mean": np.zeros((1, 1, 1, 3), dtype=np.float32),
        "raw_std": np.full((1, 1, 1, 3), 2.0, dtype=np.float32),
    }
    normalized_train, _, _, stats = _normalize_train_only(
        train, validation, raw_stats=frozen
    )
    assert np.all(normalized_train[0] == 50.0)
    assert np.array_equal(stats["raw_mean"], frozen["raw_mean"])
    assert np.array_equal(stats["raw_std"], frozen["raw_std"])


def test_second_order_endpoint_feature_is_finite_and_not_silent_default_change():
    sample_count, length = 4, 8
    pathlines = np.zeros((sample_count, 7, length, 3), dtype=np.float32)
    basis = np.eye(3, dtype=np.float32)
    for sample in range(sample_count):
        for step in range(length):
            t = float(step)
            skew = np.asarray([
                [0.0, -(sample + 1) * (0.02 * t + 0.003 * t * t), 0.0],
                [(sample + 1) * (0.02 * t + 0.003 * t * t), 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
            transform = basis + skew
            for axis, (plus, minus) in enumerate(((1, 2), (3, 4), (5, 6))):
                vector = 0.5 * transform[:, axis]
                pathlines[sample, plus, step] = vector
                pathlines[sample, minus, step] = -vector
    record = {
        "raw": pathlines.reshape(sample_count, -1),
        "fmt": np.zeros((sample_count, 161), dtype=np.float32),
        "features": {},
    }
    first_order = feature_matrix(record, "aivd1w3", "cpu")
    second_order = feature_matrix(record, "aivd1w3d2", "cpu")
    assert first_order.shape == second_order.shape == (sample_count, 8)
    assert np.isfinite(second_order).all()
    assert not np.array_equal(first_order, second_order)


def test_5_2_final_phase_is_hash_derived_new_population():
    digest = hashlib.sha256(
        b"mainExp_Task3_3D_5.2|final-phase-v1"
    ).hexdigest()
    assert digest == (
        "45f7218a508f675d58750bd33b41c0718c398fbeaa11fc0b225ddf914b0df655"
    )
    expected = [
        _radical_inverse(395, base) - 0.5 for base in (2, 3, 5)
    ]
    assert np.allclose(PHASE_5_2, expected, rtol=0.0, atol=1e-15)
    assert PHASE_5_2 != PHASE_5_1
    assert all(abs(value) < 0.5 for value in PHASE_5_2)
    assert len(_jobs()) == len(set(_jobs())) == 10
    for group, settings in SETTINGS_5_2.items():
        assert settings["indices"] == SETTINGS_5_1[group]["indices"]
