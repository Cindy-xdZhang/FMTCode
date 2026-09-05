import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Build_Task2_Universality_Cache import resolve_primitive_offset


def test_primitive_offset_modes_on_anisotropic_grid():
    spacing = np.array([0.08, 0.04, 0.01])
    assert np.isclose(resolve_primitive_offset(spacing, 0.5, "min"), 0.005)
    assert np.isclose(resolve_primitive_offset(spacing, 0.5, "max"), 0.04)
    expected = 0.5 * np.prod(spacing) ** (1.0 / 3.0)
    assert np.isclose(resolve_primitive_offset(spacing, 0.5, "geometric_mean"), expected)


def test_primitive_offset_rejects_unknown_mode():
    try:
        resolve_primitive_offset([1, 1, 1], 0.5, "median")
    except ValueError as error:
        assert "offset_mode" in str(error)
    else:
        raise AssertionError("unknown offset mode was accepted")


if __name__ == "__main__":
    test_primitive_offset_modes_on_anisotropic_grid()
    test_primitive_offset_rejects_unknown_mode()
    print("TASK2 PRIMITIVE OFFSET TEST PASSED")
