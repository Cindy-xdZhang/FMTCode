import unittest
import numpy as np
from FMT_Utils.LargeNeighbor_3D import shell_offsets,large_neighbor_fmt,objectivity_certificate


class LargeNeighborTests(unittest.TestCase):
    def setUp(self):
        t=np.arange(32)/32
        self.x=np.broadcast_to(shell_offsets(.2)[None,:,None,:],(4,25,32,3)).copy()
        self.x[:,:, :,0]*=np.sqrt(1+.2*np.sin(2*np.pi*3*t))
        self.x+=np.array([1,2,3])[None,None,None,:]*t[None,None,:,None]

    def test_exact_shell_geometry(self):
        p=shell_offsets(2.)
        self.assertEqual(p.shape,(25,3));self.assertEqual(len(np.unique(p,axis=0)),25)
        np.testing.assert_allclose(np.linalg.norm(p[1:],axis=1),np.repeat([2,3,4],8))
        for shell in p[1:].reshape(3,8,3):np.testing.assert_allclose(shell.mean(axis=0),0,atol=1e-15)

    def test_time_dependent_rigid_observer(self):
        self.assertEqual(objectivity_certificate(self.x)['status'],'PASS')

    def test_arbitrary_common_center_transport(self):
        shift=np.cos(np.arange(32)**2)[None,None,:,None]*np.array([9,-8,3])
        np.testing.assert_allclose(large_neighbor_fmt(self.x),large_neighbor_fmt(self.x+shift),atol=2e-6)

    def test_retains_nonzero_frequency(self):
        # A pure third-harmonic squared distance has its energy in bin3.
        x=np.broadcast_to(shell_offsets(1)[None,:,None,:],(1,25,32,3)).copy()
        x*=np.sqrt(1+.2*np.sin(2*np.pi*3*np.arange(32)/32))[None,None,:,None]
        f=large_neighbor_fmt(x);self.assertEqual(f.shape,(1,3300))
        energy=np.sum(f[0,:1800].reshape(300,6)[:,1:]**2+f[0,1800:].reshape(300,5)**2,axis=0)
        self.assertEqual(int(np.argmax(energy))+1,3)

    def test_late_geometry_changes_features(self):
        y=self.x.copy();y[:,1:,20:]*=1.1
        self.assertGreater(float(np.linalg.norm(large_neighbor_fmt(y)-large_neighbor_fmt(self.x))),1.)

    def test_outer_shell_information_is_retained(self):
        y=self.x.copy();y[:,17:,10:,1]+=.04
        self.assertGreater(float(np.linalg.norm(large_neighbor_fmt(y)-large_neighbor_fmt(self.x))),1.)

    def test_chunking_and_invalid_input(self):
        np.testing.assert_array_equal(large_neighbor_fmt(self.x,chunk_size=1),large_neighbor_fmt(self.x))
        with self.assertRaises(ValueError):large_neighbor_fmt(np.zeros((1,25,32,3)))
        with self.assertRaises(ValueError):large_neighbor_fmt(self.x[:,:7])


if __name__=='__main__':unittest.main()
