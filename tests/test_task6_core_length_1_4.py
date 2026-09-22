"""Check the physical definition, exact cutoff, and curved-line arclength."""
import unittest
import numpy as np
from FMT_Utils.Task6CoreLength_3D import mean_grid_spacing, filter_core_lengths


class TestCoreLength(unittest.TestCase):
    def test_nonuniform_grid_uses_full_mean_not_smallest_cell(self):
        axes = [[0, .001, .2], [-1, 0, 1], [0, .02, 1]]
        np.testing.assert_allclose(mean_grid_spacing(axes), [.1, 1, .5])

    def test_cutoff_includes_equality_and_uses_arc_not_chord(self):
        cores = [np.array([[0, 0, 0], [9.999, 0, 0]]),
                 np.array([[0, 0, 0], [10, 0, 0]]),
                 np.array([[0, 0, 0], [3, 4, 0], [0, 0, 0]])]
        retained, lengths, keep = filter_core_lengths(cores, 1)
        np.testing.assert_array_equal(keep, [False, True, True])
        np.testing.assert_allclose(lengths, [9.999, 10, 10])
        self.assertEqual(len(retained), 2)


if __name__ == '__main__':
    unittest.main()
