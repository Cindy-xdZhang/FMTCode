"""Scientific checks of discrete rigid removal and the unchanged c156 network."""
import unittest
import numpy as np
import torch
from FMT_Utils.ASAPFrame_1_1 import asap_frame, transformed_observer
from FMT_Utils.PrimitiveRigidCamera_1_1 import analytic_motion, solve_camera
from FMT_Utils.ASAPFMT_Task35_1_1 import classifier, encode


class ASAPTests(unittest.TestCase):
    @staticmethod
    def geometry(scene='deform', count=32):
        t = np.linspace(0, 4, count)
        x = np.array([analytic_motion(s, scene)[0] for s in t]).transpose(1, 0, 2)[None]
        return x, t

    def test_arbitrary_per_sample_observer(self):
        x, t = self.geometry()
        y = asap_frame(x, t)
        other = asap_frame(transformed_observer(x), t)
        np.testing.assert_allclose(y, other, atol=2e-12, rtol=2e-12)
        z = encode(x, t, 'asap_fmt').numpy()
        zs = encode(transformed_observer(x), t, 'asap_fmt').numpy()
        np.testing.assert_allclose(z, zs, atol=2e-5, rtol=2e-5)

    def test_pure_rigid_and_center(self):
        x, t = self.geometry('rigid')
        y, d = asap_frame(x, t, return_details=True)
        np.testing.assert_allclose(y, np.repeat(y[:, :, :1], len(t), axis=2), atol=2e-13)
        np.testing.assert_array_equal(y[:, 0], 0)
        self.assertLess(np.max(d['speed_rms']), 1e-12)
        np.testing.assert_allclose(np.linalg.det(d['rotation']), 1., atol=3e-14)

    def test_distances_residual_and_continuous_limit(self):
        errors=[]
        for count in (17, 65):
            x, t = self.geometry(count=count)
            y, d = asap_frame(x, t, return_details=True)
            continuous=solve_camera(lambda s:analytic_motion(s,'deform'), t)['observed'].transpose(1,0,2)[None]
            errors.append(float(np.max(np.abs(continuous-y))))
            a=np.linalg.norm(x[:, :, None]-x[:, None],axis=-1)
            b=np.linalg.norm(y[:, :, None]-y[:, None],axis=-1)
            np.testing.assert_allclose(a,b,atol=5e-13)
            r=x-x[:,:1]
            before=np.mean(np.sum(np.diff(r[:,1:],axis=2)**2,axis=-1),axis=1)/np.diff(t)**2
            self.assertTrue(np.all(d['speed_rms']**2<=before+1e-12))
        self.assertLess(errors[1],errors[0]/5)

    def test_model_counts_and_cpu_backward(self):
        torch.set_num_threads(2)
        x,t=self.geometry()
        for arm, count in [('raw_c156',993154),('fmt_c156',992386),('asap_fmt',992386)]:
            model=classifier(arm)
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            result=model(encode(x,t,arm))
            result.sum().backward()
            self.assertEqual(tuple(result.shape),(1,2))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_degeneracy_and_times_rejected(self):
        x,t=self.geometry()
        with self.assertRaises(ValueError):asap_frame(x,t[::-1])
        x[...,1:]=0
        with self.assertRaises(ValueError):asap_frame(x,t)


if __name__=='__main__':unittest.main()
