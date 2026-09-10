import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import cross_offsets
from FMT_Utils.Task6Recovery_3D import initialize_vae
from FMT_Utils.Task6Scarce_3D import subset_order, inputs_numpy, inputs_torch, add_dropout, perturb_geometry, fit_small, primary_score
from FMT_Utils.PrimitiveVAE_3D import geometry_metrics
from experiments.Task6_PrimitiveVAE_2_1 import predict


def geometry(n):
    rng=np.random.default_rng(71)
    x=np.broadcast_to(cross_offsets()[None,:,None],(n,7,32,3)).copy()
    t=np.linspace(0,1,32)[None,None,:,None]
    x+=rng.normal(size=(n,1,1,3))*t
    x+=rng.normal(size=(n,7,1,3))*t*t*.03
    return x.astype(np.float32)


class TestScarce(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_nested_subsets_are_shared_unique_and_reproducible(self):
        a=subset_order(24000,41)
        b=subset_order(24000,41)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(set(a[:4096])),4096)
        self.assertTrue(set(a[:256])<=set(a[:1024])<=set(a[:4096]))
        self.assertFalse(np.array_equal(a,subset_order(24000,42)))

    def test_noise_uses_same_geometry_and_does_not_change_clean_targets(self):
        y=torch.tensor(geometry(8));original=y.clone()
        a=perturb_geometry(y,.005,torch.Generator().manual_seed(9))
        b=perturb_geometry(y,.005,torch.Generator().manual_seed(9))
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(y,original)
        torch.testing.assert_close(a[:,:,0],y[:,:,0])
        self.assertGreater((a[:,:,1:]-y[:,:,1:]).abs().sum().item(),0)
        for k in (10,16):
            np.testing.assert_allclose(inputs_torch(a,'signed_fmt_vae',k).numpy(),
                inputs_numpy(a.numpy(),'signed_fmt_vae',k),rtol=1e-5,atol=1e-7)

    def test_dropout_changes_training_only_and_preserves_frozen_maps(self):
        y=geometry(40);x=inputs_numpy(y,'raw_vae',16)
        model,_=initialize_vae(x,y,'raw_vae',8,16,width=24,blocks=1)
        before={name:value.clone() for name,value in model.named_buffers()}
        add_dropout(model.encoder_residual,.25);add_dropout(model.decoder_residual,.25)
        z=torch.randn(4,24)
        dropout=next(m for m in model.modules() if isinstance(m,torch.nn.Dropout))
        dropout.train();self.assertFalse(torch.equal(dropout(z),dropout(z)))
        dropout.eval();torch.testing.assert_close(dropout(z),z)
        for name,value in model.named_buffers():torch.testing.assert_close(value,before[name])
        model.eval()
        p1=model(torch.tensor(x),sample=False)[0]
        p2=model(torch.tensor(x),sample=False)[0]
        torch.testing.assert_close(p1,p2,rtol=0,atol=0)

    def test_full_frequency_and_raw_share_initial_geometry_reconstruction(self):
        y=geometry(48);q=geometry(12)
        raw,_=initialize_vae(inputs_numpy(y,'raw_vae',16),y,'raw_vae',16,16,width=24,blocks=1)
        fmt,_=initialize_vae(inputs_numpy(y,'signed_fmt_vae',16),y,'signed_fmt_vae',16,16,width=24,blocks=1)
        p1=predict(raw,inputs_numpy(q,'raw_vae',16),torch.device('cpu'))
        p2=predict(fmt,inputs_numpy(q,'signed_fmt_vae',16),torch.device('cpu'))
        np.testing.assert_allclose(p1,p2,atol=3e-6,rtol=1e-5)

    def test_positive_step_selection_small_train_and_independent_metric(self):
        y=geometry(48)
        candidate=dict(id='test',frequencies=16,dropout=.1,weight_decay=.0001,noise_sigma=.001)
        cfg=dict(width=24,blocks=1,latent_dim=8,batch_size=16,updates=4,probe_every=2,
            learning_rate=1e-4,beta=1e-5,gradient_clip=1.)
        with tempfile.TemporaryDirectory() as parent:
            model,result=fit_small(y[:32],y[32:],'signed_fmt_vae',candidate,cfg,7,torch.device('cpu'),Path(parent)/'fit')
            self.assertEqual(result['initialization']['train_samples'],32)
            self.assertEqual(result['updates'],4)
            self.assertGreater(result['selected_step'],0)
            self.assertEqual(result['train_examples_exposed'],64)
            x=inputs_numpy(y[32:],'signed_fmt_vae',16)
            score=primary_score(model,x,y[32:],torch.device('cpu'))
            self.assertAlmostEqual(score,geometry_metrics(predict(model,x,torch.device('cpu')),y[32:])['position_rmse_r'],places=12)
            self.assertFalse(list(Path(parent).rglob('*.pt')))


if __name__=='__main__':unittest.main()
