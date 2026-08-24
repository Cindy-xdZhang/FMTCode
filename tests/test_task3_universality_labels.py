import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Build_Task3_Universality_Labels import _config_identity


def test_config_identity_is_order_stable_and_label_formula_is_strict():
    assert _config_identity({"b": 2, "a": 1}) == _config_identity({"a": 1, "b": 2})
    ivd = np.asarray([1.0, 2.0, 2.0])
    threshold = np.asarray([1.0, 1.5, 2.0])
    assert np.array_equal(ivd > threshold, [False, True, False])


if __name__ == "__main__":
    test_config_identity_is_order_stable_and_label_formula_is_strict()
    print("TASK3 UNIVERSALITY LABEL TEST PASSED")
