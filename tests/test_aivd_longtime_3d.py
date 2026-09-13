"""Window-only extension and synchronous multiscale feature context checks."""
import unittest
import numpy as np
from FMT_Utils.Task12Data_3D import _anchored_recipe, feature_matrix, feature_block_dims
from tests.test_aivd_transfer_3d import fixture, reference_numpy


def encode(x, name):
    return feature_matrix({'raw': x.astype(np.float32).reshape(len(x), -1),
                           'features': {}}, name)


class LongtimeTests(unittest.TestCase):
    def test_only_window_changes(self):
        short = _anchored_recipe('aivd1w3_dft')
        long = _anchored_recipe('aivd1w3_dft_longtime')
        self.assertEqual(short['window'], 3)
        self.assertEqual(long, {**short, 'window': None})
        self.assertEqual(feature_block_dims('aivd1w3_dft_longtime'), (1,))

    def test_full_window_equals_independent_sequence_sum(self):
        x, _ = fixture(32)
        d = np.stack((x[:, 1]-x[:, 2], x[:, 3]-x[:, 4], x[:, 5]-x[:, 6]), axis=-1)
        b = np.empty_like(d)
        b[:, 0], b[:, -1] = d[:, 1]-d[:, 0], d[:, -1]-d[:, -2]
        b[:, 1:-1] = .5*(d[:, 2:]-d[:, :-2])
        g = b @ np.linalg.pinv(d, rcond=1e-6)
        w = np.stack((g[..., 2, 1]-g[..., 1, 2], g[..., 0, 2]-g[..., 2, 0],
                      g[..., 1, 0]-g[..., 0, 1]), axis=-1)
        expected = np.linalg.norm(w-w.mean(axis=0), axis=-1).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(encode(x, 'aivd1w3_dft_longtime'), expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(encode(x, 'aivd1w3_dft'), reference_numpy(x), rtol=1e-4, atol=1e-6)

    def test_late_geometry_is_used_only_by_longtime(self):
        x, _ = fixture(32)
        changed = x.copy()
        changed[0, 1, 20, 1] += .2
        np.testing.assert_array_equal(encode(x, 'aivd1w3_dft'), encode(changed, 'aivd1w3_dft'))
        self.assertGreater(float(np.max(np.abs(encode(x, 'aivd1w3_dft_longtime')-
                                               encode(changed, 'aivd1w3_dft_longtime')))), 1e-3)


if __name__ == '__main__':
    unittest.main()
