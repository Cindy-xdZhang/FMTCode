"""Independent analytic checks; these do not change the frozen training code."""
import unittest
import numpy as np
from FMT_Utils.Task4B_GeometricCurl_3D import integrate_geometry,geometry_derivatives


class GeometricInvariants(unittest.TestCase):
    def test_translation_and_length_velocity_scaling(self):
        seed=np.array([[.4,.5,.6]]);low=np.zeros((1,3));high=np.ones((1,3));step=np.array([.002])
        curves=[]
        for length,speed,shift in [(1.,1.,np.zeros(3)),(100.,30.,np.array([200.,-3.,5.]))]:
            def velocity(p):
                local=(p-shift)/length
                return speed*np.column_stack((1+local[:,2],local[:,0],.2+local[:,1]))
            q,_,_=integrate_geometry(velocity,seed*length+shift,step*length,low*length+shift,
                high*length+shift,np.array([3.*speed]),mode='time')
            curves.append(q)
        np.testing.assert_allclose(curves[0],curves[1],atol=3e-12,rtol=3e-12)

    def test_solid_body_rotation_curl_and_parallel_angle(self):
        def velocity(p):return np.column_stack((-p[:,1],p[:,0],np.ones(len(p))))
        seed=np.zeros((1,3));step=np.array([.01]);low=np.full((1,3),-1.);high=-low
        q,_,_=integrate_geometry(velocity,seed,step,low,high,np.array([2.]),mode='time')
        _,curl,_,cosine=geometry_derivatives(q)
        np.testing.assert_allclose(curl[0,16],[0,0,.01],atol=5e-8)
        np.testing.assert_allclose(cosine,[1.],atol=1e-12)

    def test_constant_flow_has_zero_curl(self):
        def velocity(p):return np.tile([1.,.2,.3],(len(p),1))
        q,_,_=integrate_geometry(velocity,np.zeros((2,3)),np.full(2,.01),np.full((2,3),-1.),
            np.full((2,3),1.),np.full(2,2.),mode='time')
        _,curl,descriptors,_=geometry_derivatives(q)
        np.testing.assert_allclose(curl,0,atol=1e-12)
        self.assertTrue(np.isfinite(descriptors).all())


if __name__=='__main__':unittest.main()
