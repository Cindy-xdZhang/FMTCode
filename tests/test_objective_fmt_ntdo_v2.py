"""Implementation contract tests only; no observer/objectivity experiments."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from FMT_Utils.objective_fmt_nTDO_v2 import build_fourier_inputs, objective_fmt_nTDO_v2


class ScalarFourierContract(unittest.TestCase):
    def test_all_pairs_and_only_scalar_fft_input(self):
        rng = np.random.default_rng(9110)
        x = rng.normal(size=(3, 7, 32, 3))
        pairs = [(i, j) for i in range(7) for j in range(i+1, 7)]
        expected = np.stack([np.linalg.norm(x[:, j]-x[:, i], axis=-1) for i,j in pairs], axis=1)
        inputs = build_fourier_inputs(x)
        self.assertEqual(set(inputs), {'distances', 'pair_indices'})
        np.testing.assert_array_equal(inputs['pair_indices'].numpy().T, pairs)
        np.testing.assert_allclose(inputs['distances'].numpy(), expected, atol=1e-14)
        with patch('torch.fft.rfft', wraps=torch.fft.rfft) as fft:
            actual = objective_fmt_nTDO_v2(x)
        self.assertEqual(fft.call_count, 1)
        np.testing.assert_array_equal(fft.call_args.args[0].numpy(), inputs['distances'].numpy())
        self.assertEqual(fft.call_args.kwargs, {'dim': 2})
        spectrum = np.fft.rfft(expected, axis=2)[:, :, :6]
        oracle = np.concatenate((spectrum.real.reshape(3,-1), spectrum.imag[:,:,1:].reshape(3,-1)), axis=1)
        self.assertEqual(actual.shape, (3,231))
        np.testing.assert_allclose(actual, oracle, atol=1e-12)

    def test_tensor_autograd_and_optional_fourth_coordinate(self):
        x = torch.randn(2,7,32,3,dtype=torch.float64,requires_grad=True)
        result = objective_fmt_nTDO_v2(x, return_blocks=True, return_numpy=False)
        self.assertEqual(set(result), {'features','distance_fourier'})
        self.assertEqual(result['features'].dtype, torch.float64)
        result['features'].square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with_time = torch.cat((x.detach(),torch.arange(32).expand(2,7,32).unsqueeze(-1)),dim=-1)
        np.testing.assert_array_equal(objective_fmt_nTDO_v2(with_time),objective_fmt_nTDO_v2(x))
        self.assertEqual(objective_fmt_nTDO_v2(x,num_freq=3).shape,(2,105))

    def test_invalid_inputs(self):
        x = torch.zeros(2,7,32,3)
        for frequency in (True,1,18,2.5):
            with self.assertRaises(ValueError): objective_fmt_nTDO_v2(x,num_freq=frequency)
        for invalid in (torch.zeros(2,6,32,3),torch.zeros(0,7,32,3),torch.full_like(x,float('nan'))):
            with self.assertRaises(ValueError): objective_fmt_nTDO_v2(invalid)


if __name__ == '__main__': unittest.main()
