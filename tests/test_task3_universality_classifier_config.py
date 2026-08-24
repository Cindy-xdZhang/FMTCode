import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Verify_Task3_FMTClassifier import _load_dataset, _portable_basename


def test_portable_basename_accepts_windows_and_posix_cache_paths():
    expected = "slice_00_index_0040.npz"
    assert _portable_basename(
        r"C:\work\cache\boeing747\slice_00_index_0040.npz"
    ) == expected
    assert _portable_basename(
        "/work/cache/boeing747/slice_00_index_0040.npz"
    ) == expected


def test_all_configured_datasets_have_ten_aligned_slices():
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load(
        (root / "config/Verify_Task3UniversalityClassifier_1.1.yaml").read_text(
            encoding="utf-8"
        )
    )
    for dataset in spec["datasets"]:
        source = sorted((root / spec["source_cache_root"] / dataset).glob("slice_*.npz"))
        labels = sorted((root / spec["label_cache_root"] / dataset).glob("slice_*.npz"))
        assert len(source) == len(labels) == 10
        assert [path.name for path in source] == [path.name for path in labels]
    # Exercise the production loader on two structurally different datasets.
    for dataset in ("cylinder3d", "channel"):
        records = _load_dataset(
            root / spec["source_cache_root"] / dataset,
            root / spec["label_cache_root"] / dataset,
            spec["sampled_steps"], spec["fmt_subset"],
        )
        assert len(records) == 10


if __name__ == "__main__":
    test_portable_basename_accepts_windows_and_posix_cache_paths()
    test_all_configured_datasets_have_ten_aligned_slices()
    print("TASK3 UNIVERSALITY CLASSIFIER CONFIG TEST PASSED")
