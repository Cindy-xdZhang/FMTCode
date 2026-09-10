import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from FMT_Utils.FlowMapData_3D import cross_offsets
from FMT_Utils.Task6Recovery_3D import signed_fmt,inverse_signed_fmt,linear_inverse_matrix,initialize_vae
from FMT_Utils.PrimitiveVAE_3D import fmt_tokens,vae_loss
from experiments.Task6_Reconstruction_3_1 import fit


class TestRecovery(unittest.TestCase):
    def test_full_complex_coefficients_roundtrip_with_line_identity(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(4,7,32,3)).cumsum(2)
        x-=x[:,:,:1]
        x+=cross_offsets()[None,:,None]
        np.testing.assert_allclose(inverse_signed_fmt(signed_fmt(x,16),16),x,atol=1e-12)
        for k in (6,10,16):
            matrix,bias=linear_inverse_matrix(k)
            z=signed_fmt(x,k)
            np.testing.assert_allclose((z@matrix+bias).reshape(x.shape),inverse_signed_fmt(z,k),atol=2e-13)

    def test_geometry_phase_and_direction_are_retained(self):
        x=np.broadcast_to(cross_offsets()[None,:,None],(1,7,32,3)).copy()
        x[:,:,:,0]+=np.linspace(0,1,32)
        y=np.broadcast_to(cross_offsets()[None,:,None],(1,7,32,3)).copy()
        y[:,:,:,1]+=np.linspace(0,1,32)
        self.assertGreater(np.linalg.norm(signed_fmt(x)-signed_fmt(y)),.01)
        # The old classification token discards this signed displacement.
        np.testing.assert_allclose(fmt_tokens(x),fmt_tokens(y),atol=1e-6)

    def test_initialized_vae_has_a_real_latent_bottleneck_and_gradients(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(8)
        x=np.broadcast_to(cross_offsets()[None,:,None],(40,7,32,3)).copy()
        x+=rng.normal(size=(40,1,1,3))*np.linspace(0,1,32)[None,None,:,None]
        tokens=signed_fmt(x,16).astype(np.float32)
        model,info=initialize_vae(tokens,x,"signed_fmt_vae",8,16,width=32,blocks=1)
        inputs=torch.tensor(tokens)
        prediction,mu,lv=model(inputs,sample=False)
        np.testing.assert_allclose(prediction.detach().numpy(),x,atol=2e-6)
        self.assertEqual(mu.shape,(40,8))
        np.testing.assert_array_equal(model.decode(mu).detach().numpy(),prediction.detach().numpy())
        self.assertGreater((model.decode(torch.zeros_like(mu))-prediction).abs().sum().item(),0)
        noisy,mu,lv=model(inputs,sample=True)
        loss,_,_=vae_loss(noisy,torch.tensor(x,dtype=torch.float32),mu,lv,1e-5)
        loss.backward()
        self.assertGreater(model.decoder_residual[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.posterior_logvar.grad.abs().sum().item(),0)

    def test_training_selects_a_positive_step_without_test_data(self):
        rng=np.random.default_rng(12)
        x=np.broadcast_to(cross_offsets()[None,:,None],(48,7,32,3)).copy()
        x+=rng.normal(size=(48,1,1,3))*np.linspace(0,1,32)[None,None,:,None]
        x=x.astype(np.float32); inputs=signed_fmt(x,10).astype(np.float32)
        spec=dict(frequencies=10,vae=dict(latent_dim=8,width=32,blocks=1,learning_rate=1e-4,
            batch_size=16,epochs=2,probe_every=3,beta=1e-5,gradient_clip=1.))
        with tempfile.TemporaryDirectory() as folder:
            model,record=fit(spec,inputs[:32],x[:32],inputs[32:],x[32:],"signed_fmt_vae",4,torch.device("cpu"),Path(folder))
            self.assertEqual(record["updates"],4)
            self.assertGreater(record["selected_step"],0)
            self.assertLess(record["selected_validation_rmse_r"],.01)


if __name__=="__main__": unittest.main()
