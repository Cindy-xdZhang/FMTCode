import unittest
import numpy as np
from FMT_Utils.FMTAllV2_3D import fmt_all_v2


class FMTAllV2Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7097)
        self.x=rng.normal(size=(8,7,32,3))*.1
        self.x+=np.linspace(0,2,32)[None,None,:,None]

    def test_time_dependent_rigid_observer(self):
        rng=np.random.default_rng(7081);q=[]
        for j in range(32):
            r,_=np.linalg.qr(rng.normal(size=(3,3)))
            r[:,0]*=np.linalg.det(r);q.append(r)
        transformed=np.einsum('tij,nktj->nkti',np.asarray(q),self.x)
        transformed+=rng.normal(size=(1,1,32,3))*10
        np.testing.assert_allclose(fmt_all_v2(transformed),fmt_all_v2(self.x),rtol=2e-6,atol=2e-6)

    def test_arbitrary_centre_transport_does_not_change_features(self):
        shift=np.sin(np.arange(32)**2)[None,None,:,None]*np.array([3.,-2.,7.])
        np.testing.assert_allclose(fmt_all_v2(self.x+shift),fmt_all_v2(self.x),rtol=2e-6,atol=2e-6)

    def test_invariant_geometry_retains_temporal_frequency(self):
        t=np.arange(32)/32
        x=np.zeros((1,7,32,3))
        cross=np.concatenate((np.eye(3),-np.eye(3)))
        x[:,1:]=cross[None,:,None,:]*np.sqrt(1+.2*np.sin(2*np.pi*3*t))[None,None,:,None]
        f=fmt_all_v2(x)
        self.assertEqual(f.shape,(1,231))
        real=f[0,:126].reshape(21,6);imag=f[0,126:].reshape(21,5)
        power=real[:,1:]**2+imag**2
        self.assertEqual(int(np.argmax(power.sum(axis=0)))+1,3)

    def test_chunking_and_input_validation(self):
        np.testing.assert_array_equal(fmt_all_v2(self.x,chunk_size=2),fmt_all_v2(self.x))
        with self.assertRaises(ValueError):fmt_all_v2(np.zeros((1,7,32,3)))
        a=self.x.copy();a[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):fmt_all_v2(a)

    def test_time_varying_rotation_changes_vector_fft_negative_control(self):
        t=np.arange(32)/32
        x=np.zeros((1,7,32,3));x[:,1:]=np.eye(3)[np.arange(6)%3][None,:,None,:]
        c,s=np.cos(2*np.pi*t),np.sin(2*np.pi*t)
        y=x.copy();y[...,0]=c*x[...,0]-s*x[...,1];y[...,1]=s*x[...,0]+c*x[...,1]
        old=np.fft.rfft(x[:,1:]-x[:,:1],axis=2)
        changed=np.fft.rfft(y[:,1:]-y[:,:1],axis=2)
        self.assertGreater(float(np.max(np.abs(old-changed))),1.)
        np.testing.assert_allclose(fmt_all_v2(x),fmt_all_v2(y),atol=1e-12)


if __name__=='__main__':unittest.main()
