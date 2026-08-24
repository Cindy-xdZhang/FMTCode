import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.LocalIVDLabel_3D import (
    local_ivd_labels_3d, local_ivd_threshold_3d, sample_local_ivd_labels_3d,
)


def test_literal_mean_fraction_and_local_percentile_are_not_confused():
    constant = np.ones((5, 5, 5), dtype=np.float32)
    mean_labels, mean_threshold = local_ivd_labels_3d(
        constant, 3, "mean_fraction", 0.9
    )
    percentile_labels, percentile_threshold = local_ivd_labels_3d(
        constant, 3, "percentile", 90.0
    )
    assert mean_labels.all() and np.allclose(mean_threshold, 0.9)
    assert not percentile_labels.any() and np.allclose(percentile_threshold, 1.0)


def test_spike_is_above_both_local_thresholds():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[2, 2, 2] = 10.0
    for mode, value in (("mean_fraction", 0.9), ("percentile", 90.0)):
        labels, _ = local_ivd_labels_3d(volume, 3, mode, value)
        assert labels[2, 2, 2]


def test_seed_sampling_uses_xyz_coordinates_for_zyx_volume():
    z, y, x = np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij")
    volume = (100 * z + 10 * y + x).astype(np.float32)
    labels, sampled, threshold = sample_local_ivd_labels_3d(
        volume, (np.arange(3), np.arange(3), np.arange(3)),
        np.array([[2.0, 1.0, 0.0]]), 1, "mean_fraction", 0.5,
    )
    assert sampled[0] == 12.0 and threshold[0] == 6.0 and labels[0]


def test_even_neighborhood_is_rejected():
    try:
        local_ivd_threshold_3d(np.ones((3, 3, 3)), 4)
    except ValueError as error:
        assert "odd" in str(error)
    else:
        raise AssertionError("even neighborhood size was accepted")


if __name__ == "__main__":
    test_literal_mean_fraction_and_local_percentile_are_not_confused()
    test_spike_is_above_both_local_thresholds()
    test_seed_sampling_uses_xyz_coordinates_for_zyx_volume()
    test_even_neighborhood_is_rejected()
    print("LOCAL IVD LABEL 3D TEST PASSED")
