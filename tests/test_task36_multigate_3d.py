"""Task-conditioned routing, physical labels, training and frozen single-task controls."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FMT_Utils.FMT_3D_pipeline import compute_ivd_reference_3d
from FMT_Utils.Task36Labels_3D import labels_at_seeds
from FMT_Utils.Task36MultiGate_3D import (Task36MultiGate,balanced_loss,classification_metrics,
    choose_threshold,train_joint,VARIANTS)
from FMT_Utils.Task6FMTGeometryMoE_3D import fit_statistics,cached
from tests.test_task6_direct_neural_5_1 import examples
from tests.test_task6_fmt_geometry_moe_1_1 import candidate


def joint_candidate():
    return dict(candidate(),classification_width=32,dropout=0.)


class TestTask36(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_independent_task_gates_can_choose_different_experts(self):
        y=examples(6,72)
        model=Task36MultiGate(fit_statistics(y),joint_candidate(),'joint_task_gates')
        x=cached(model,y,torch.device('cpu'))
        model.condition(x)
        model.eval()
        with torch.no_grad():
            model.classification_router[-1].bias.copy_(torch.tensor([20.,-20.]))
            model.router[-1].bias.copy_(torch.tensor([-20.,20.]))
        z3,z6,g=model.encode_tasks(x)
        za=model.expert_a((x[0]-model.a_mean)/model.a_scale)
        zb=model.geometry((x[1]-model.b_mean)/model.b_scale)
        torch.testing.assert_close(z3,za)
        torch.testing.assert_close(z6,zb)
        self.assertTrue((g['task3']>.99).all() and (g['task6']<.01).all())

    def test_capacity_and_single_gate_controls(self):
        y=examples(5,1)
        models={}
        for variant in VARIANTS:
            torch.manual_seed(11)
            model=Task36MultiGate(fit_statistics(y),joint_candidate(),variant)
            models[variant]=model
            model.eval()
            x=cached(model,y,torch.device('cpu'))
            z3,z6,g=model.encode_tasks(x)
            if variant in ('joint_shared_gate','joint_geometry_only'):
                torch.testing.assert_close(z3,z6)
            if variant=='joint_fixed_routes':
                self.assertTrue((g['task3']==1).all() and (g['task6']==0).all())
        a,b=models['joint_task_gates'],models['joint_raw_task_gates']
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        for x,y in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_task_losses_are_comparable_at_constant_predictions(self):
        labels=torch.tensor([1.,0.,0.,0.])
        self.assertAlmostEqual(float(balanced_loss(torch.zeros(4),labels,.25)),1.,places=6)
        prob=np.array([.9,.3,.2,.1])
        threshold=choose_threshold(labels.numpy(),prob)
        self.assertEqual(classification_metrics(labels.numpy(),prob,threshold)['f1'],1.)
        self.assertIsNone(classification_metrics(np.zeros(4),prob)['average_precision'])

    def test_ivd_uses_the_whole_field_percentile_and_physical_seed(self):
        axis=np.linspace(-1,1,21,dtype=np.float32)
        z,y,x=np.meshgrid(axis,axis,axis,indexing='ij')
        field=UnsteadyVectorField3D(21,21,21,2,[-1,-1,-1],[1,1,1],0.,1.)
        frame=np.stack((x*0,x*x,z*0),axis=-1)
        field.field=np.stack((frame,frame))
        origins=np.array([[0.,0.,0.],[1.,0.,0.],[-1.,0.,0.]])
        labels,values,threshold=labels_at_seeds(field,origins)
        volume,expected,_=compute_ivd_reference_3d(field,0.,origins)
        np.testing.assert_array_equal(values,expected)
        self.assertEqual(threshold,float(np.percentile(volume,95)))
        self.assertEqual(labels.tolist(),[0,1,1])
        # Threshold comes from the volume, not the queried three points.
        _,center_values,center_threshold=labels_at_seeds(field,origins[:1])
        self.assertEqual(threshold,center_threshold)
        self.assertNotEqual(threshold,float(np.percentile(center_values,95)))

    def test_actual_joint_fit_and_different_task_gradients(self):
        y=examples(32,66)
        measure=np.linalg.norm(y[:,0,-1],axis=-1)
        labels=(measure>np.median(measure)).astype(np.int64)
        training=dict(updates=650,batch_size=16,warmup_updates=20,probe_every=325,gradient_clip=1.)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            model,fit=train_joint(y,labels,y,labels,'joint_task_gates',joint_candidate(),training,7,
                                  torch.device('cpu'),Path(temp)/'joint')
            self.assertGreater(fit['train']['task3']['average_precision'],.98)
            self.assertLess(fit['train']['task6_rmse_r'],fit['curve'][0]['train']['task6_rmse_r']*.25)
            self.assertEqual(fit['downstream_token_dimension'],12)
            self.assertEqual(fit['joint_pair_dimensions'],24)
            self.assertIsNotNone(model.classification_router[-1].weight.grad)
            self.assertIsNotNone(model.router[-1].weight.grad)
            self.assertFalse(list(Path(temp).rglob('*.pt')))


if __name__=='__main__':
    unittest.main()
