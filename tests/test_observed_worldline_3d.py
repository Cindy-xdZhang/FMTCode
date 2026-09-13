import unittest
import numpy as np
from FMT_Utils.ObservedFieldWorldline_3D import (
    TranslatingObserverField, ReferenceFrameFromWorldline, ObservedVectorField)
from FMT_Utils.TranslationObserver_3D import integrate_pathlines


class ObservedWorldlineTests(unittest.TestCase):
    def test_camera_integral_reference_time_and_start(self):
        u=TranslatingObserverField([0,1,2],[[0,0,0],[2,0,0],[4,0,0]])
        r=ReferenceFrameFromWorldline(u,1,[7,2,3])
        np.testing.assert_allclose(r.camera_position(2),[10,2,3],atol=1e-11)
        np.testing.assert_allclose(r.camera_position(0),[6,2,3],atol=1e-11)
        np.testing.assert_allclose(r.lab_to_observed([[9,2,3]],2),[[6,2,3]])
        np.testing.assert_allclose(r.observed_to_lab([[6,2,3]],2),[[9,2,3]])

    def test_inverse_query_and_wrong_relative_field_negative_control(self):
        u=TranslatingObserverField([0,1],[[1,0,0]]*2)
        r=ReferenceFrameFromWorldline(u,0)
        v=lambda x,t: np.asarray(x)*.5
        w=ObservedVectorField(v,u,r)
        t=np.linspace(0,1,25);seed=np.array([[1.,.5,0]])
        lab=integrate_pathlines(v,seed,t,.01)
        actual=integrate_pathlines(w,seed,t,.01)
        expected=np.stack([r.lab_to_observed(lab[:,j],time) for j,time in enumerate(t)],axis=1)
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        wrong=integrate_pathlines(lambda x,t:v(x,t)-u(x,t),seed,t,.01)
        self.assertGreater(np.max(np.abs(wrong-expected)),.1)

    def test_observed_trajectory_can_leave_original_box(self):
        def v(x,t):
            ans=np.zeros_like(x);ans[(x[:,0]<0)|(x[:,0]>1)]=np.nan
            return ans
        u=TranslatingObserverField([0,1],[[2,0,0]]*2)
        r=ReferenceFrameFromWorldline(u,0)
        p=integrate_pathlines(ObservedVectorField(v,u,r),np.array([[.5,0,0]]),[0,1],.02)
        np.testing.assert_allclose(p[0,-1],[-1.5,0,0],atol=1e-11)

    def test_equal_time_geometry_not_cross_time_curve_geometry(self):
        t=np.linspace(0,1,11)
        lab=np.stack([np.column_stack([t,t*t,0*t]),np.column_stack([t+.2,t*t+.3,0*t])])
        u=TranslatingObserverField([0,1],[[1,0,0]]*2)
        r=ReferenceFrameFromWorldline(u,0)
        obs=np.stack([r.lab_to_observed(lab[:,j],time) for j,time in enumerate(t)],axis=1)
        np.testing.assert_allclose(obs[1]-obs[0],lab[1]-lab[0],atol=1e-12)
        self.assertFalse(np.allclose(np.diff(obs[0],axis=0),np.diff(lab[0],axis=0)))


if __name__=='__main__':unittest.main()
