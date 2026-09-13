"""Sign, inverse-location, velocity-integral and classifier independence tests."""
import unittest
import numpy as np
from FMT_Utils.TranslationObserver_3D import TranslationObserver,integrate_pathlines,evaluate_observers


class TranslationObserverTests(unittest.TestCase):
    def test_constant_velocity_is_integrated_with_negative_frame_shift(self):
        o=TranslationObserver([2,5],[[2,-1,0]]*2,2)
        np.testing.assert_allclose(o.displacement(4,.25),[1,-.5,0])
        np.testing.assert_allclose(o.transform_paths(np.zeros((1,2,3)),[2,4],.25),[[[0,0,0],[-1,.5,0]]])

    def test_time_varying_velocity_not_interpolated_displacement(self):
        t=np.linspace(0,2,5)
        o=TranslationObserver(t,np.column_stack([2*t,np.zeros((5,2))]),0)
        np.testing.assert_allclose(o.displacement([0,.5,2])[:,0],[0,.25,4],atol=1e-12)
        np.testing.assert_allclose(o.velocity(1,.5),[1,0,0])

    def test_inverse_query_position(self):
        o=TranslationObserver([0,2],[[1,0,0]]*2,0)
        v=lambda x,t:2*x
        observed=o.observed_field(v,1)
        np.testing.assert_allclose(observed(np.array([[0.,0,0]]),1),[[1,0,0]])
        self.assertFalse(np.allclose(observed(np.zeros((1,3)),1),v(np.zeros((1,3)),1)-o.velocity(1)))

    def test_transformed_trajectory_equals_observed_ode(self):
        t=np.linspace(0,1,13)
        o=TranslationObserver([0,.5,1],[[.2,0,0],[.4,.2,0],[.6,.4,0]],0)
        v=lambda x,t:np.column_stack((-x[:,1]+.3,x[:,0],np.zeros(len(x))))
        x0=np.array([[.1,.2,0],[.4,0,0]])
        lab=integrate_pathlines(v,x0,t,.01)
        for a in [0,.25,.5,1]:
            obs=integrate_pathlines(o.observed_field(v,a),x0,t,.01)
            np.testing.assert_allclose(obs,o.transform_paths(lab,t,a),atol=2e-9)

    def test_moving_domain_is_checked_in_original_coordinates(self):
        def v(x,t):
            result=np.zeros_like(x);result[(x[:,0]<0)|(x[:,0]>1)]=np.nan
            return result
        o=TranslationObserver([0,1],[[2,0,0]]*2,0)
        paths=integrate_pathlines(o.observed_field(v,1),np.array([[.5,0,0]]),[0,1],.05)
        self.assertTrue(np.isfinite(paths).all())
        self.assertAlmostEqual(paths[0,-1,0],-1.5)

    def test_four_independent_classifications_with_common_material_ids(self):
        initial=np.zeros((2,7,3));initial[1,:,0]=.1
        v=lambda x,t:np.tile([.4,0,0],(len(x),1))
        o=TranslationObserver([0,1],[[1,0,0]]*2,0)
        calls=[]
        def classifier(p):
            calls.append(p.copy())
            displacement=p[:,0,-1,0]-p[:,0,0,0]
            return displacement>0,displacement[:,None]
        paths,labels,ids,audit=evaluate_observers(v,initial,np.linspace(0,1,4),classifier,o,.02)
        self.assertEqual(len(calls),4)
        np.testing.assert_array_equal(labels[:,0],[1,1,0,0])
        self.assertEqual(audit['changed_labels_vs_original'],[0,0,2,2])
        np.testing.assert_array_equal(ids,[0,1])

    def test_rejects_extrapolation_and_nonincreasing_times(self):
        with self.assertRaises(ValueError):TranslationObserver([1,0],np.zeros((2,3)),0)
        o=TranslationObserver([0,1],np.zeros((2,3)),0)
        with self.assertRaises(ValueError):o.displacement(2)


if __name__=='__main__':unittest.main()
