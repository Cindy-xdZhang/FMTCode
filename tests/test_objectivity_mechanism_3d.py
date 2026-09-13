import unittest
import numpy as np
import torch
from FMT_Utils.ObjectivityMechanism_3D import (
    feature_set, gram_series, scalar_spectrum, observe, analytic_checks, pad, WIDTH,
)
from FMT_Utils.FMTAllV2_3D import fmt_all_v2


class MechanismTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(90)
        offsets=np.vstack([np.zeros(3),np.eye(3)[0],-np.eye(3)[0],np.eye(3)[1],
                           -np.eye(3)[1],np.eye(3)[2],-np.eye(3)[2]])
        self.x=np.broadcast_to(offsets[None,:,None,:],(6,7,32,3)).copy()
        self.x+=rng.normal(size=self.x.shape)*.03

    def test_frozen_core_reproduced(self):
        np.testing.assert_allclose(scalar_spectrum(gram_series(self.x)),fmt_all_v2(self.x),rtol=1e-6,atol=1e-6)

    def test_center_mask_is_only_change(self):
        v=feature_set(self.x)
        np.testing.assert_array_equal(v['old_no_center'][:,23:],v['old_full'][:,23:])
        self.assertTrue(np.all(v['old_no_center'][:,:23]==0))
        self.assertTrue(all(pad(x).shape==(6,WIDTH) for x in v.values()))

    def test_matches_cache_generation_parameters(self):
        from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
        cached=pathline_dft_features_3d(torch.as_tensor(self.x,dtype=torch.float32),
                                      neighbor_scale=1.0,neighbor_weight=1.0)
        a=feature_set(self.x,cached)['old_full']
        b=feature_set(self.x)['old_full']
        np.testing.assert_array_equal(a,b)

    def test_cached_and_recomputed_are_distinct_controls(self):
        base=feature_set(self.x)
        changed=base['old_full'][:,:161].copy()+.01
        actual=feature_set(self.x,changed)
        np.testing.assert_array_equal(actual['old_full'][:,:161],changed)
        np.testing.assert_array_equal(actual['old_recomputed_full'],base['old_full'])
        np.testing.assert_array_equal(actual['old_no_center'][:,23:],actual['old_full'][:,23:])

    def test_gram_objectivity(self):
        for y in [observe(self.x),observe(self.x,local=True),observe(self.x,translation=True)]:
            np.testing.assert_allclose(gram_series(y),gram_series(self.x),atol=1e-12)

    def test_no_center_not_objective(self):
        a=feature_set(self.x)['old_no_center'][:,:161]
        b=feature_set(observe(self.x))['old_no_center'][:,:161]
        self.assertGreater(float(np.linalg.norm(a-b)),1)

    def test_full_spectrum_reconstructs(self):
        s=gram_series(self.x);f=scalar_spectrum(s,full=True)
        real=f[:,:21*17].reshape(6,21,17).transpose(0,2,1)
        imag=f[:,21*17:].reshape(6,21,15).transpose(0,2,1)
        imaginary=np.pad(imag,((0,0),(1,1),(0,0)))
        np.testing.assert_allclose(np.fft.irfft(real+1j*imaginary,axis=1),s,atol=1e-7)

    def test_rigid_rotation_counterexample(self):
        self.assertEqual(analytic_checks()['status'],'PASS')


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
