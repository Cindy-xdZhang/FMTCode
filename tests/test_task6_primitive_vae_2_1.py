"""Physical integration, split isolation and true geometry VAE checks."""
import copy
import tempfile
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import cross_offsets, integrate
from FMT_Utils.PrimitiveVAE_3D import time_plan, resample_time, fmt_tokens, PrimitiveVAE, vae_loss, geometry_metrics
from FLowUtils.VectorField3d import UnsteadyVectorField3D
from experiments.Task6_PrimitiveVAE_2_1 import strict_window, fit, preflight, build, audit_data
import json


class TestPrimitiveVAE(unittest.TestCase):
    def test_cropped_cylinder_time_and_isolation(self):
        t = np.linspace(7.5, 15, 76)
        plan = time_plan(t, range(76), [0, 15], True)
        self.assertEqual(plan["seed_time_bounds"], [7.5, 12.])
        frames = []
        for role, starts in plan["starts"].items():
            self.assertGreaterEqual(len(starts), 2)
            frames.append(set(j for i in starts for j in range(i,i+13)))
            self.assertTrue(all(7.5 <= t[i] <= 12.000001 for i in starts))
        self.assertFalse(frames[0] & frames[1] or frames[0] & frames[2] or frames[1] & frames[2])
        self.assertEqual(plan["starts"]["train"][0], 0)

    def test_missing_source_frames_never_enter_window(self):
        t = np.arange(200.)
        available = set(range(200)) - set(range(40,49)) - set(range(100,110))
        plan = time_plan(t, available, [0,199])
        for starts in plan["starts"].values():
            self.assertTrue(all(set(range(i,i+13)) <= available for i in starts))

    def test_affine_field_has_full_correct_trajectory(self):
        field = UnsteadyVectorField3D(9,9,9,3,[-4,-4,-4],[4,4,4],0,2)
        z,y,x = np.meshgrid(np.linspace(-4,4,9),np.linspace(-4,4,9),np.linspace(-4,4,9),indexing="ij")
        # v=(y, 1, 0): x(t)=x0+y0*t+t^2/2, y(t)=y0+t.
        field.field = np.stack([np.stack([y,np.ones_like(x),np.zeros_like(x)],-1)]*3).astype(np.float32)
        seeds = .1*cross_offsets()[None]
        paths, flags = integrate(field,seeds,0,1,48)
        self.assertTrue(flags.all())
        target=seeds.copy(); target[...,0]+=seeds[...,1]+.5; target[...,1]+=1
        np.testing.assert_allclose(paths[:,:,-1],target,atol=2e-7)
        sampled=resample_time(paths)
        self.assertEqual(sampled.shape,(1,7,32,3))
        np.testing.assert_array_equal(sampled[:,:,-1], paths[:,:,-1])
        with self.assertRaises(ValueError): integrate(field,seeds,0,3,48)

    def test_masked_voxels_are_not_stationary_fake_flow(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"masked.nc"
            with nc.Dataset(path,"w") as d:
                for a,n in (("t",13),("x",4),("y",4),("z",4)):
                    d.createDimension(a,n); v=d.createVariable(a,"f8",(a,)); v[:]=np.arange(n)
                for name in ("u","v","w"):
                    v=d.createVariable(name,"f4",("t","z","y","x"),fill_value=-999.)
                    v[:]=0.; v[:,:,0,0]=-999.
            field,_=strict_window(path,0,13,96)
            self.assertTrue(np.isnan(field.field[:,:,0,0]).all())
            _,valid=integrate(field,np.array([[0.,0.,1.]]),0,1,32)
            self.assertFalse(valid[0])

    def test_geometry_loss_kl_and_deterministic_mean(self):
        torch.manual_seed(123); torch.set_num_threads(2)
        model=PrimitiveVAE(161,width=32,latent_dim=8,blocks=1)
        x=torch.randn(4,161); y=torch.randn(4,7,32,3)
        p,mu,lv=model(x)
        loss,rec,kl=vae_loss(p,y,mu,lv,1e-5)
        loss.backward()
        self.assertEqual(p.shape,y.shape)
        self.assertGreater(model.log_variance.weight.grad.abs().sum().item(),0)
        np.testing.assert_array_equal(model(x,False)[0].detach(),model(x,False)[0].detach())
        with self.assertRaises(ValueError): vae_loss(p,y,mu,lv,0)

    def test_frozen_token_and_training_reduces_geometry_error(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(7)
        t=np.linspace(0,1,32)
        y=np.broadcast_to(cross_offsets()[None,:,None],(32,7,32,3)).copy()
        y+=rng.uniform(.1,1,(32,1,1,3))*t[None,None,:,None]
        y=y.astype(np.float32); x=fmt_tokens(y)
        self.assertEqual(x.shape,(32,161))
        spec={"vae":dict(width=64,latent_dim=8,blocks=1,batch_size=32,learning_rate=.003,beta=1e-5,probe_every=100)}
        with tempfile.TemporaryDirectory() as folder:
            _,mean,std,record=fit(spec,x,y,x[:4],y[:4],1,torch.device("cpu"),fixed_steps=150,folder=Path(folder))
        self.assertLess(record["curve"][-1]["train_probe_rmse_r"],record["curve"][0]["train_probe_rmse_r"]*.7)
        np.testing.assert_allclose(mean,x.mean(0,dtype=np.float64),rtol=1e-6,atol=1e-6)
        m=geometry_metrics(y+np.array([1.,0,0]),y)
        self.assertAlmostEqual(m["position_rmse_r"],1.)
        self.assertAlmostEqual(m["pair_distance_rmse_r"],0.)

    def test_cache_build_audit_and_corruption_detection(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder); path=folder/"flow.nc"
            with nc.Dataset(path,"w") as d:
                for a,n in (("t",80),("x",5),("y",5),("z",5)):
                    d.createDimension(a,n); v=d.createVariable(a,"f8",(a,)); v[:]=np.arange(n)
                for name in ("u","v","w"):
                    v=d.createVariable(name,"f4",("t","z","y","x")); v[:]=.001
            spec=json.loads(Path("config/mainExp_Task6_PrimitiveVAE_2.1.json").read_text())
            spec.update(datasets=["synthetic"],source_fields={"synthetic":str(path)},
                        cylinder_datasets=[],available_frame_ranges={},output_root=str(folder/"out"))
            spec["sampling"]["retained_samples"]={"train":64,"validation":36,"test":36,"unseen_scale":36}
            config=folder/"config.json"; config.write_text(json.dumps(spec))
            preflight(spec,config); build(spec,config,"synthetic")
            report=audit_data(spec,config,"synthetic")
            self.assertTrue(report["passed"])
            cache=folder/"out"/"data"/"synthetic"/"train.npz"
            with cache.open("ab") as f: f.write(b"corruption")
            with self.assertRaises(AssertionError): audit_data(spec,config,"synthetic")


if __name__=="__main__": unittest.main()
