"""Independent Fourier reference and fixed-center contract checks."""
import unittest

import numpy as np
import torch

from FMT_Utils.FMT_SingleCenter_1_1 import encode, fixed_center_ids, SingleCenterMLP
from FMT_Utils.FMT_P35_NormFrequency_3_1 import primitive_features
from FMT_Utils.FMTNoConvolution_1_1 import assert_no_fmt_convolution, ForbiddenFMTConvolutionError


def reference(x, center_id, k=6):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean((0, 1))
    x /= np.linalg.norm(x, axis=-1).max()
    center = x[center_id]

    def descriptor(sequence):
        f = np.fft.rfft(sequence, axis=0)[:k]
        a, b = f.real, f.imag
        an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        gram = np.stack((an, bn, (a*b).sum(-1)/np.maximum(an*bn, 1e-8)), -1).ravel()
        triple = (np.cross(a[:-1], b[:-1])*a[1:]).sum(-1)/np.maximum(an[:-1]*bn[:-1]*an[1:], 1e-8)
        return np.r_[gram, triple]

    neighbors = np.stack([descriptor(np.diff(line-center, axis=0)) for i, line in enumerate(x) if i != center_id])
    delta = np.diff(center, axis=0)
    tangent = delta/np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12)
    tangent = np.concatenate((tangent, tangent[-1:]))
    spectrum = np.fft.rfft(np.concatenate((center, tangent), -1), axis=0, norm='ortho')[:k]
    direction = np.stack((spectrum.real, spectrum.imag), -1).ravel()
    return np.r_[descriptor(delta), direction, neighbors.mean(0), neighbors.max(0)]


class SingleCenterTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(910917).normal(size=(4, 11, 32, 3)).cumsum(2)

    def test_numpy_reference_variable_lines_and_centers(self):
        counts, centers = np.array([7, 8, 10, 11]), np.array([0, 2, 4, 10])
        actual = encode(self.x, counts, centers).numpy()
        expected = np.stack([reference(x[:n], c) for x, n, c in zip(self.x, counts, centers)])
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    def test_original_p35_formula(self):
        x = torch.tensor(self.x[:, :7], dtype=torch.float64)
        x -= x.mean((1, 2), keepdim=True)
        x /= x.norm(dim=-1).amax((1, 2))[:, None, None, None]
        np.testing.assert_allclose(encode(self.x[:, :7]), primitive_features(x, 6), atol=2e-6, rtol=2e-6)

    def test_neighbor_permutation_padding_batch_and_affine_scale(self):
        x = self.x[:, :7]
        expected = encode(x)
        for changed in (x[:, [0, 5, 3, 2, 6, 1, 4]], x*7.3+np.array([2., -5., 7.])):
            torch.testing.assert_close(encode(changed), expected)
        padded = np.pad(x, ((0, 0), (0, 4), (0, 0), (0, 0)), constant_values=np.nan)
        torch.testing.assert_close(encode(padded, [7]*4), expected)
        torch.testing.assert_close(torch.cat([encode(row[None]) for row in x]), expected)

    def test_points_and_center_selection(self):
        x = np.random.default_rng(10).normal(size=(3, 5, 48, 3))
        self.assertEqual(tuple(encode(x).shape), (3, 141))
        seeds = np.array([[[1., 0, 0], [0., 0, 0], [0., 0, 0]]])
        ids, distance = fixed_center_ids(seeds, [2], [[0., 0, 0]])
        self.assertEqual(ids.tolist(), [1])
        self.assertEqual(distance.tolist(), [0.])
        with self.assertRaises(ValueError):
            encode(x, center_ids=[5, 0, 0])

    def test_network_one_vector_and_no_convolution(self):
        model = SingleCenterMLP()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 78530)
        self.assertEqual(tuple(model(encode(self.x)).shape), (4, 2))
        with self.assertRaises(ValueError):
            model(torch.zeros(4, 11, 141))
        model.forbidden = torch.nn.Conv2d(1, 1, 1)
        with self.assertRaises(ForbiddenFMTConvolutionError):
            assert_no_fmt_convolution(model)


if __name__ == '__main__':
    unittest.main()
