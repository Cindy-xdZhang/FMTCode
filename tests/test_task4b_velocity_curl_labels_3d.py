"""Physical orientation, unit changes and strict-coverage regression checks."""
import unittest
import numpy as np
from FMT_Utils.Task4B_VelocityCurlLabels_3D import coverage_threshold, velocity_curl_labels, spatial_mean


class LabelsTest(unittest.TestCase):
    def test_classes_sign_ties_and_scale(self):
        u = np.tile([1., 0, 0], (8, 1))
        w = np.array([[1,0,0], [0,1,0], [0,-1,0], [-1,0,0], [1,1,0], [1,1,0], [0,0,1], [0,0,-1]], float)
        h = [False, False, True, True, False, True, False, True]
        expected = [0,1,2,3,0,3,1,2]
        for us, ws in [(1,1), (100,0.01), (0.01,100)]:
            labels, _, valid = velocity_curl_labels(u * us, w * ws, h)
            np.testing.assert_array_equal(labels, expected)
            self.assertTrue(valid.all())

    def test_invalid_vectors_only(self):
        labels, _, valid = velocity_curl_labels([[0,0,0],[1,0,0]], [[1,0,0],[np.nan,0,0]], [True,False])
        np.testing.assert_array_equal(labels, [-1,-1])
        self.assertFalse(valid.any())

    def test_strict_coverage_and_speed_length_units(self):
        values = np.array([0.1, 10., 2.])
        threshold = coverage_threshold(values)
        self.assertTrue(np.all(values > threshold['a']))
        self.assertEqual(threshold['vortex_threshold'], 0.9 * threshold['a'])
        scaled = coverage_threshold(values * 100)
        self.assertAlmostEqual(scaled['vortex_threshold'], threshold['vortex_threshold'] * 100)
        for invalid in ([0,1], [], [np.nan], [-1]):
            with self.assertRaises(ValueError):
                coverage_threshold(invalid)

    def test_nonuniform_volume_mean(self):
        z, y, x = np.array([0.,1.,4.]), np.array([0.,2.,3.]), np.array([0.,1.,2.])
        field = np.zeros((3,3,3,3))
        field[..., 0] = z[:,None,None]
        field[..., 1] = y[None,:,None]
        field[..., 2] = x[None,None,:]
        np.testing.assert_allclose(spatial_mean(field, (z,y,x)), [2.,1.5,1.])


if __name__ == '__main__':
    unittest.main()
