import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Verify_3DFMTHyperparam import _score


def test_score_matches_arbitrary_cluster_id():
    reference = np.array([True, True, False, False, False])
    labels = np.array([1, 1, 0, 0, 0])
    result = _score(labels, reference)
    assert result["cluster_as_vortex"] == 1
    assert result["f1"] == 1.0
    assert result["predicted_fraction"] == 0.4


if __name__ == "__main__":
    test_score_matches_arbitrary_cluster_id()
    print("3D FMT HYPERPARAMETER TEST PASSED")
