from pathlib import Path

import yaml

from FMT_Utils.MultiscalePathline_3D import parse_scale_table


ROOT = Path(__file__).resolve().parents[1]


def _master():
    return yaml.safe_load(
        (ROOT / "config/mainExp_Task5_3D_1.1.yaml").read_text(encoding="utf-8")
    )


def test_scale_tuple_splits_are_disjoint_and_fixed_width():
    spec = _master()
    sampled_steps = spec["pathlines"]["sampled_steps"]
    parsed = {
        name: parse_scale_table(rows, sampled_steps)
        for name, rows in spec["scale_sets"].items()
    }
    assert {name: len(rows) for name, rows in parsed.items()} == {
        "train": 18, "validation": 6, "confirmation": 9,
    }
    tuples = {
        name: {
            (row.offset_grid_scale, row.dt_scale, row.integration_steps)
            for row in rows
        }
        for name, rows in parsed.items()
    }
    assert tuples["train"].isdisjoint(tuples["validation"])
    assert tuples["train"].isdisjoint(tuples["confirmation"])
    assert tuples["validation"].isdisjoint(tuples["confirmation"])
    assert max(row.horizon_in_source_frames for rows in parsed.values() for row in rows) == 12


def test_development_and_confirmation_source_windows_do_not_overlap():
    spec = _master()
    maximum_horizon = 12
    development = spec["phases"]["development"]["time_indices_by_dataset"]
    confirmation = spec["phases"]["confirmation"]["time_indices_by_dataset"]
    assert set(development) == set(confirmation)
    for dataset in development:
        assert len(development[dataset]) == 6
        assert len(confirmation[dataset]) == 4
        assert max(development[dataset]) + maximum_horizon < min(confirmation[dataset])


def test_training_configs_freeze_fixed_tensor_and_five_paired_seeds():
    for group in ("old8", "new2"):
        for method in ("baselines", "fmt", "raw_pca"):
            path = ROOT / f"config/mainExp_Task5_3D_1.1_{method}_{group}.yaml"
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert spec["expected_slices"] == 6
            assert spec["sampled_steps"] == 32
            assert spec["split"]["train_ordinals"] == [0, 1, 2, 3]
            assert spec["split"]["validation_ordinals"] == [4, 5]
            assert spec["training"]["seeds"] == [40, 41, 42, 43, 44]
