import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Run_Pathline_Hyperparams_3D import _load_common_slices


def _write_slice(path, mask):
    valid_indices = np.flatnonzero(mask)
    count = len(valid_indices)
    np.savez_compressed(
        path,
        raw_features=np.column_stack((valid_indices, valid_indices)).astype(np.float32),
        fmt_features=np.column_stack((valid_indices, valid_indices + 10)).astype(np.float32),
        reference=(valid_indices % 2 == 1),
        valid_mask=np.asarray(mask, dtype=bool),
        metadata_json=np.asarray(json.dumps({"source_start_index": 7})),
    )


def test_common_valid_seed_intersection():
    with tempfile.TemporaryDirectory() as temporary:
        tmp_path = Path(temporary)
        masks = {"a": np.r_[np.ones(180, dtype=bool), np.zeros(20, dtype=bool)],
                 "b": np.r_[np.zeros(20, dtype=bool), np.ones(180, dtype=bool)]}
        for variant, mask in masks.items():
            directory = tmp_path / variant / "flow"; directory.mkdir(parents=True)
            for index in range(10):
                _write_slice(directory / f"slice_{index:02d}.npz", mask)
        records = _load_common_slices(tmp_path, ["a", "b"], "a", "flow")
        assert len(records) == 10
        np.testing.assert_array_equal(records[0]["raw"][:, 0], np.arange(20, 180))
        np.testing.assert_array_equal(records[0]["fmt"][:, 0], np.arange(20, 180))
        np.testing.assert_array_equal(records[0]["reference"], np.arange(20, 180) % 2 == 1)
        assert records[0]["metadata"]["common_valid_fraction"] == 0.8


def test_three_axes_change_only_the_intended_pathline_quantity():
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "config/Verify_PathlineHyperparams3D_1.1.yaml").read_text())
    variants = {item["id"]: item for item in spec["variants"]}
    assert len({(item["dt_scale"], item["integration_steps"], item["sampled_steps"])
                for item in variants.values()}) == len(variants)
    assert {variants[name]["dt_scale"] * variants[name]["integration_steps"]
            for name in ("dt_small", "baseline", "dt_large")} == {12.0}
    assert {(variants[name]["dt_scale"], variants[name]["sampled_steps"])
            for name in ("steps_short", "baseline", "steps_long")} == {(0.25, 32)}
    assert {(variants[name]["dt_scale"], variants[name]["integration_steps"])
            for name in ("samples_16", "baseline", "samples_48")} == {(0.25, 48)}
    assert [spec["screening_seed"], *spec["confirmation_seeds"]] == [7068, 7069, 7070]
    physical = sum(spec["comparison"]["physical_families"].values(), [])
    assert len(physical) == len(set(physical)) == 7
    assert set(physical) | {spec["comparison"]["synthetic_control"]} == set(spec["datasets"])


if __name__ == "__main__":
    test_common_valid_seed_intersection()
    test_three_axes_change_only_the_intended_pathline_quantity()
    print("PATHLINE HYPERPARAMETER TEST PASSED")
