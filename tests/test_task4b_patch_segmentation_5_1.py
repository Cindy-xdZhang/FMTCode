import unittest
import numpy as np
from FMT_Utils.Task4B_PatchSplit_3D import make_manifest,audit_manifest,local_step
from experiments.Task4B_PatchSegmentation_5_1 import integrate_local,aggregate_unique,metrics


class UniformField:
    def velocity(self,points):
        return np.broadcast_to([2.,0.,0.],points.shape).copy()


class PatchSegmentationTests(unittest.TestCase):
    def test_exact_counts_and_intentional_overlap_rejected(self):
        ids=np.zeros((80,60,60),np.int32)
        for i in range(10):
            x=3+(i%5)*10;z=3+(i//5)*30
            ids[z:z+3,5:8,x:x+3]=i+1
        patches,_=make_manifest(ids,seed=19,training_count=90,test_count=10,
            test_fraction=.1,padding=1,gap=1,max_attempts=10000)
        report=audit_manifest(ids,patches,1)
        self.assertEqual(report['patch_counts'],dict(train=90,test=10))
        self.assertEqual(report['instance_counts'],dict(train=9,test=1))
        a=next(p for p in patches if p['split']=='train')
        b=next(p for p in patches if p['split']=='test')
        b['low']=a['low'];b['high']=a['high']
        with self.assertRaises(AssertionError):audit_manifest(ids,patches,1)

    def test_uniform_flow_geometry_and_native_bounds(self):
        seeds=np.array([[.5,.5,.5],[.01,.5,.5]])
        low=np.zeros((2,3));high=np.ones((2,3))
        h=local_step(seeds,low,high,.1,16,.15)
        primitive,valid,ql,qh=integrate_local(UniformField(),seeds,h,low,high,16,.15,1e-8)
        self.assertTrue(valid.all());self.assertEqual(primitive.shape,(2,7,33,3))
        np.testing.assert_allclose(primitive[:,0,:,0],np.broadcast_to(np.arange(-16,17),(2,33)),atol=1e-6)
        self.assertTrue((ql>=low).all() and (qh<=high).all())
        self.assertLess(h[1],h[0])

    def test_overlap_aggregation_and_invalid_penalty(self):
        source=np.array([3,3,8]);truth=np.array([0,0,2])
        probs=np.array([[.7,.3,0,0],[.5,.5,0,0],[0,0,0,0]])
        u,y,p,_=aggregate_unique(source,truth,probs,np.array([0,0,-2]))
        np.testing.assert_array_equal(u,[3,8]);np.testing.assert_array_equal(p,[0,-2])
        result=metrics(y,p)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['class_support'],[1,0,1,0])
        self.assertEqual(result['invalid_prediction_count'],1)


if __name__=='__main__':unittest.main()
