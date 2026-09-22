import unittest
import numpy as np
from FMT_Utils.Task6Refinement_1_3 import accepted_refinement


class RefinementTests(unittest.TestCase):
    def test_reject_finer_failure_even_when_coarser_curves_agree(self):
        a = np.zeros((3, 65, 3)); b = a.copy(); c = a.copy()
        reasons = [np.zeros((3, 2), np.int8) for _ in range(3)]
        reasons[2][1, 1] = 3; c[2, :, 0] = .1
        ok, _ = accepted_refinement(a, b, c, reasons, np.ones((3, 2)), 1.)
        np.testing.assert_array_equal(ok, [True, False, False])

    def test_retains_original_error_bound_and_short_curve_filter(self):
        a = np.zeros((2, 65, 3)); b = a.copy(); c = a.copy(); c[:, :, 0] = .05
        reasons = [np.zeros((2, 2), np.int8) for _ in range(3)]
        lengths = np.array([[.6, .6], [.5, .5]])
        ok, error = accepted_refinement(a, b, c, reasons, lengths, 1.)
        np.testing.assert_array_equal(ok, [True, False]); np.testing.assert_allclose(error, .05)


if __name__ == '__main__': unittest.main()
