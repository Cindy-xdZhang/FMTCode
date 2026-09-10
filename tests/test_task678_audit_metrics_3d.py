"""Validate independent replay against frozen metrics across scales and shapes."""
import unittest
import numpy as np
from FMT_Utils.HanFlowMapData_3D import measured_trajectories as frozen
from FMT_Utils.FlowMapAuditMetrics_3D import measured_trajectories as independent


class AuditMetricTests(unittest.TestCase):
    def test_every_metric_matches_across_shapes_and_scales(self):
        rng=np.random.default_rng(91021)
        for n,q,t,offset,radius in [(9,8,13,0.,.2),(5,128,63,1000.,.001),(3,64,125,-100.,10.)]:
            truth=(offset+rng.normal(size=(n,q,t,3))*radius).astype(np.float32)
            prediction=(truth+rng.normal(size=truth.shape)*radius*.03).astype(np.float32)
            r=np.geomspace(radius,radius*2,n)
            a=frozen(prediction,truth,r);b=independent(prediction,truth,r)
            for key,value in a.items():
                if isinstance(value,str):self.assertEqual(value,b[key])
                else:np.testing.assert_allclose(value,b[key],rtol=1e-11,atol=1e-11,err_msg=key)

    def test_zero_error_and_material_translation(self):
        rng=np.random.default_rng(7);truth=rng.normal(size=(4,32,63,3)).astype(np.float32)
        self.assertEqual(independent(truth,truth,np.ones(4))['pair_distance_error'],0.)
        prediction=truth.copy();prediction[:,:,1:,0]+=.25
        a=frozen(prediction,truth,np.ones(4));b=independent(prediction,truth,np.ones(4))
        for key in ('pair_distance_error','centered_shape_error','position_nrmse'):
            self.assertAlmostEqual(a[key],b[key],places=12)


if __name__=='__main__':unittest.main()
