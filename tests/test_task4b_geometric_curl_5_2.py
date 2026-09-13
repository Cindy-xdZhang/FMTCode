import unittest
import numpy as np
from FMT_Utils.Task4B_GeometricCurl_3D import integrate_geometry,geometry_derivatives


class GeometricCurlTests(unittest.TestCase):
    def test_shear_time_geometry_retains_curl_unit_geometry_loses_it(self):
        def shear(points):return np.column_stack((1+2*points[:,2],np.zeros(len(points)),np.zeros(len(points))))
        seeds=np.array([[.5,.5,.5]]);h=np.array([.001]);low=np.zeros((1,3));high=np.ones((1,3))
        results={}
        for mode in ('unit','time'):
            q,ql,qh=integrate_geometry(shear,seeds,h,low,high,np.array([3.]),mode=mode)
            _,curl,_,_=geometry_derivatives(q);results[mode]=curl[0,16]
            self.assertTrue((ql>=low).all() and (qh<=high).all())
        np.testing.assert_allclose(results['unit'],[0,0,0],atol=1e-12)
        np.testing.assert_allclose(results['time'],[0,2*.001/3,0],atol=1e-12)

    def test_uniform_speed_scaling_leaves_time_geometry_unchanged(self):
        seeds=np.array([[.5,.5,.5]]);h=np.array([.001]);low=np.zeros((1,3));high=np.ones((1,3))
        curves=[]
        for scale in (1.,100.):
            def velocity(p):return scale*np.column_stack((1+2*p[:,2],p[:,0],np.ones(len(p))))
            q,_,_=integrate_geometry(velocity,seeds,h,low,high,np.array([4.*scale]),mode='time')
            curves.append(q)
        np.testing.assert_allclose(curves[0],curves[1],rtol=1e-13,atol=1e-13)


if __name__=='__main__':unittest.main()
