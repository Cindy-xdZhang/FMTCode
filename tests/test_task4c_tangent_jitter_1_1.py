import unittest
import numpy as np
import torch
from FMT_Utils.Task4C_TangentJitter_1_1 import tangent_jitter, set_dropout


def reference(points, noise, sigma, maximum):
    result=points.copy()
    for prefix in np.ndindex(points.shape[:-2]):
        p=points[prefix]
        for i in range(1,len(p)-1):
            t=p[i+1]-p[i-1];norm=np.linalg.norm(t)
            scale=min(np.linalg.norm(p[i]-p[i-1]),np.linalg.norm(p[i+1]-p[i]))
            if norm>0:result[prefix+(i,)]+=t/norm*np.clip(noise[prefix+(i,)]*sigma,-maximum,maximum)*scale
    return result


class TangentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(96611)
        self.p=torch.tensor(rng.normal(size=(2,3,32,3)).cumsum(2),dtype=torch.float64)
        self.n=torch.tensor(rng.normal(size=(2,3,32)),dtype=torch.float64)

    def test_numpy_reference_tangent_bound_and_endpoints(self):
        a=tangent_jitter(self.p,self.n)
        np.testing.assert_allclose(a,reference(self.p.numpy(),self.n.numpy(),.1,.25),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(a[...,[0,-1],:],self.p[...,[0,-1],:],atol=0,rtol=0)
        delta=a[...,1:-1,:]-self.p[...,1:-1,:];t=self.p[...,2:,:]-self.p[...,:-2,:]
        self.assertLess(float(torch.linalg.vector_norm(torch.linalg.cross(delta,t),dim=-1).max()),1e-12)
        distances=torch.linalg.vector_norm(self.p.diff(dim=-2),dim=-1)
        bound=.25*torch.minimum(distances[...,:-1],distances[...,1:])
        self.assertTrue((torch.linalg.vector_norm(delta,dim=-1)<=bound+1e-12).all())

    def test_scale_translation_rotation_equivariance(self):
        q,_=torch.linalg.qr(torch.tensor([[1.,2.,3.],[3.,-1.,4.],[2.,4.,-2.]],dtype=torch.float64))
        a=tangent_jitter(self.p,self.n)
        b=tangent_jitter((self.p@q)*7+23,self.n)
        torch.testing.assert_close(b,(a@q)*7+23,atol=1e-12,rtol=1e-12)

    def test_zero_and_degenerate_are_safe_and_nonmutating(self):
        old=self.p.clone();torch.testing.assert_close(tangent_jitter(self.p,self.n,0),old,atol=0,rtol=0)
        p=torch.zeros_like(self.p);torch.testing.assert_close(tangent_jitter(p,self.n),p,atol=0,rtol=0)
        tangent_jitter(self.p,self.n);torch.testing.assert_close(self.p,old,atol=0,rtol=0)

    def test_dropout_keeps_parameters_and_eval_outputs(self):
        from FMT_Utils.Task4C_OriginalCenter_1_1 import make_model
        torch.manual_seed(96611);m=make_model('p35_h0');m.eval()
        x=torch.randn(4,1,142);x[...,-1]=1
        with torch.no_grad():before=m(x)
        params=[p.clone() for p in m.parameters()];changed=set_dropout(m,.35)
        self.assertEqual(len(changed),2)
        with torch.no_grad():torch.testing.assert_close(m(x),before,atol=0,rtol=0)
        for p,old in zip(m.parameters(),params):torch.testing.assert_close(p,old,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
