import unittest
import numpy as np
from FMT_Utils.GlobalMeanObserver_3D import volume_mean


class MeanObserverTests(unittest.TestCase):
    def test_nonuniform_grid_linear_field(self):
        coords = [np.array([0., .1, 1.]), np.array([0., 2., 3.]), np.array([-1., 0., 4.])]
        z, y, x = np.meshgrid(*coords, indexing='ij')
        mean, missing = volume_mean(np.stack([x, y, z], -1), coords)
        np.testing.assert_allclose(mean, [1.5, 1.5, .5], atol=1e-14)
        self.assertEqual(missing, 0)

    def test_missing_component_excludes_whole_vector(self):
        data = np.ma.array(np.full((2, 2, 2, 3), 3.), mask=False)
        data[0, 0, 0] = [100., 200., 300.]
        data.mask[0, 0, 0, 1] = True
        mean, missing = volume_mean(data, [np.arange(2.)] * 3)
        np.testing.assert_allclose(mean, [3., 3., 3.])
        self.assertEqual(missing, 1)


if __name__ == '__main__':
    unittest.main()
