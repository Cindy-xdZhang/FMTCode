"""Independent geometric and Fourier checks; executable without pytest."""
import unittest

import numpy as np
import torch

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_Angles_3D import center_angle_inputs, center_angle_fourier
from FMT_Utils.fmt_v5 import fmt_v5
from FMT_Utils.fmt_objective_ntod_v2 import fmt_objective_ntod_v2
from FMT_Utils.objective_fmt_nTDO_v2 import objective_fmt_nTDO_v2


class TestCenterAngles(unittest.TestCase):
    def cross(self):
        offsets = torch.tensor([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],
                                [0,-1,0],[0,0,1],[0,0,-1]], dtype=torch.float64)
        return offsets[None, :, None, :].repeat(2, 1, 32, 1)

    def random(self):
        return torch.randn(5, 7, 32, 3, generator=torch.Generator().manual_seed(914), dtype=torch.float64)

    def test_known_angles_and_constant_spectrum(self):
        inputs = center_angle_inputs(self.cross())
        angles = inputs['angles'].numpy()
        self.assertEqual(angles.shape, (2, 15, 32))
        for k, (i, j) in enumerate(inputs['pair_indices'].T.tolist()):
            expected = np.pi if (i, j) in [(1,2),(3,4),(5,6)] else np.pi/2
            np.testing.assert_allclose(angles[:, k], expected, atol=1e-14)
        spectrum = center_angle_fourier(self.cross()).reshape(2, -1)
        np.testing.assert_allclose(spectrum[:, :90].reshape(2,15,6)[:,:,0], angles[:,:,0]*32)
        np.testing.assert_allclose(spectrum[:, :90].reshape(2,15,6)[:,:,1:], 0, atol=1e-13)
        np.testing.assert_allclose(spectrum[:, 90:], 0, atol=1e-13)

    def test_independent_cosine_law_and_numpy_fft(self):
        x = self.random()
        inputs = center_angle_inputs(x)
        expected = []
        for i, j in inputs['pair_indices'].T.tolist():
            a = np.linalg.norm((x[:,i]-x[:,0]).numpy(), axis=-1)
            b = np.linalg.norm((x[:,j]-x[:,0]).numpy(), axis=-1)
            c = np.linalg.norm((x[:,i]-x[:,j]).numpy(), axis=-1)
            expected.append(np.arccos(np.clip((a*a+b*b-c*c)/(2*a*b), -1, 1)))
        expected = np.stack(expected, axis=1)
        np.testing.assert_allclose(inputs['angles'], expected, atol=2e-13)
        s = np.fft.rfft(expected, axis=2)[:, :, :6]
        coefficients = np.concatenate((s.real.reshape(5,-1),s.imag[:,:,1:].reshape(5,-1)),axis=1)
        np.testing.assert_allclose(center_angle_fourier(x), coefficients, atol=2e-12)

    def test_frozen_parent_prefixes(self):
        for dtype in [torch.float32, torch.float64]:
            x = self.random().to(dtype)
            np.testing.assert_array_equal(fmt_v5(x)[:,:161],pathline_dft_features_3d(x))
            np.testing.assert_array_equal(fmt_objective_ntod_v2(x)[:,:231],objective_fmt_nTDO_v2(x))
            self.assertEqual(fmt_v5(x).shape,(5,326))
            self.assertEqual(fmt_objective_ntod_v2(x).shape,(5,396))
            np.testing.assert_array_equal(
                fmt_v5(x,neighbor_scale=1.0,neighbor_weight=1.0)[:,:161],
                pathline_dft_features_3d(x,neighbor_scale=1.0,neighbor_weight=1.0))

    def test_parallel_angle_is_zero(self):
        x = self.cross(); x[:,2] = 2*x[:,1]
        np.testing.assert_array_equal(center_angle_inputs(x)['angles'][:,0], 0)

    def test_time_dependent_observer(self):
        x = self.random()
        matrices = torch.randn(32,3,3,generator=torch.Generator().manual_seed(3),dtype=x.dtype)
        q, _ = torch.linalg.qr(matrices)
        q[:,:,0] *= torch.linalg.det(q)[:,None]
        shift = torch.randn(32,3,generator=torch.Generator().manual_seed(4),dtype=x.dtype)
        moved = torch.einsum('tlk,nptk->nptl',q,x) + shift[None,None]
        np.testing.assert_allclose(center_angle_fourier(x),center_angle_fourier(moved),atol=1e-12)
        np.testing.assert_allclose(fmt_objective_ntod_v2(x),fmt_objective_ntod_v2(moved),atol=1e-12)
        self.assertGreater(np.max(np.abs(fmt_v5(x)-fmt_v5(moved))), 1e-4)

    def test_invalid_inputs_fail_without_filtering(self):
        x = self.random(); x[:,1,4] = x[:,0,4]
        with self.assertRaisesRegex(ValueError,'coincides'):
            fmt_objective_ntod_v2(x)
        for f in [True, 1, 18, 2.5]:
            with self.assertRaises(ValueError):
                center_angle_fourier(self.cross(), f)


if __name__ == '__main__':
    unittest.main()
