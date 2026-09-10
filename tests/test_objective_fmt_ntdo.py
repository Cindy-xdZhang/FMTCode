import unittest
from unittest.mock import patch

import numpy as np
import torch

from FMT_Utils.objective_fmt_nTDO import build_fourier_inputs, objective_fmt_nTDO


class ObjectiveFMTnTDOTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(12340).normal(size=(4, 7, 32, 3))

    def test_actual_fourier_inputs_are_same_time_arrays(self):
        calls = []
        original = torch.fft.rfft
        def capture(x, *args, **kwargs):
            calls.append(x.detach().numpy().copy())
            return original(x, *args, **kwargs)
        with patch('torch.fft.rfft', side_effect=capture):
            objective_fmt_nTDO(self.x)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], self.x[:, 1:] - self.x[:, :1])
        pairs = [(i, j) for i in range(7) for j in range(i+1, 7)]
        expected = np.stack([np.linalg.norm(self.x[:, j]-self.x[:, i], axis=-1)
                             for i, j in pairs], axis=1)
        np.testing.assert_allclose(calls[1], expected, atol=2e-15)
        self.assertEqual(calls[0].shape, (4, 6, 32, 3))
        self.assertEqual(calls[1].shape, (4, 21, 32))

    def test_vector_branch_matches_frozen_undifferenced_postprocessing(self):
        from FMT_Utils.DFT_FMT_3D import dft_rotation_invariants_3d
        relative = torch.from_numpy(self.x[:, 1:] - self.x[:, :1])
        expected = dft_rotation_invariants_3d(relative.reshape(-1, 32, 3), 6)
        expected = expected.reshape(4, 6, 23).sort(dim=1, descending=True).values.flatten(1)
        actual = objective_fmt_nTDO(self.x, return_blocks=True)
        np.testing.assert_array_equal(actual['relative_fourier'], expected.numpy())
        self.assertEqual(actual['features'].shape, (4, 369))

    def test_static_geometry_retains_dc_not_a_difference(self):
        x = np.zeros((1, 7, 32, 3))
        x[:, 1:] = np.concatenate((np.eye(3), -np.eye(3)))[None, :, None, :]
        blocks = objective_fmt_nTDO(x, return_blocks=True)
        self.assertEqual(blocks['relative_fourier'][0, 0], 32.)
        self.assertEqual(blocks['distance_fourier'][0, 0], 32.)
        real = blocks['distance_fourier'][:, :126].reshape(1, 21, 6)
        np.testing.assert_array_equal(real[:, :, 1:], 0.)

    def test_time_varying_translation_and_constant_rotation(self):
        q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        shift = np.random.default_rng(25).normal(size=(1, 1, 32, 3))
        y = np.einsum('ij,nktj->nkti', q, self.x) + shift
        np.testing.assert_allclose(objective_fmt_nTDO(y), objective_fmt_nTDO(self.x),
                                   rtol=1e-10, atol=1e-10)

    def test_time_varying_observer_inputs_and_distance_output(self):
        t = np.linspace(0, 1.3, 32)
        q = np.zeros((32, 3, 3))
        q[:, 0, 0] = q[:, 1, 1] = np.cos(t)
        q[:, 0, 1] = -np.sin(t); q[:, 1, 0] = np.sin(t); q[:, 2, 2] = 1.
        y = np.einsum('tij,nktj->nkti', q, self.x)
        a, b = build_fourier_inputs(self.x), build_fourier_inputs(y)
        np.testing.assert_allclose(b['relative'].numpy(),
                                   np.einsum('tij,nktj->nkti', q, a['relative'].numpy()), atol=1e-14)
        before = objective_fmt_nTDO(self.x, return_blocks=True)
        after = objective_fmt_nTDO(y, return_blocks=True)
        np.testing.assert_allclose(before['distance_fourier'], after['distance_fourier'], atol=1e-12)
        self.assertGreater(np.max(np.abs(before['relative_fourier'] - after['relative_fourier'])), .1)

    def test_distance_frequency_matches_undifferenced_signal(self):
        t = np.arange(32)/32
        signal = 2. + .2*np.sin(2*np.pi*3*t)
        x = np.zeros((1, 7, 32, 3)); x[:, 1:, :, 0] = signal[None, None, :]
        block = objective_fmt_nTDO(x, return_blocks=True)['distance_fourier']
        expected = np.fft.rfft(signal)[:6]
        np.testing.assert_allclose(block[0, :6], expected.real, atol=1e-13)
        np.testing.assert_allclose(block[0, 126:131], expected.imag[1:], atol=1e-13)

    def test_tensor_dtype_gradients_and_extra_channel(self):
        x = torch.tensor(self.x, dtype=torch.float32, requires_grad=True)
        result = objective_fmt_nTDO(x, return_numpy=False)
        self.assertEqual(result.dtype, torch.float32)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        four = np.concatenate((self.x, np.ones((4, 7, 32, 1))), axis=-1)
        np.testing.assert_array_equal(objective_fmt_nTDO(four), objective_fmt_nTDO(self.x))

    def test_invalid_inputs_and_frequency(self):
        for value in (0, 1, 6.5, True, 18):
            with self.assertRaises(ValueError):
                objective_fmt_nTDO(self.x, num_freq=value)
        for value in (self.x[:, :6], self.x[:, :, :1], self.x[:0], self.x.astype(complex)):
            with self.assertRaises(ValueError):
                objective_fmt_nTDO(value)
        invalid = self.x.copy(); invalid[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            objective_fmt_nTDO(invalid)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
