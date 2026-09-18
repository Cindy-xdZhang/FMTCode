"""Numerical neighborhood / descriptor contracts for original-center FPS16."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from FMT_Utils import Task4C_FPS16_1_1 as method
from FMT_Utils import Task4C_FPS16_Data_1_1 as data
from FMT_Utils.DFT_FMT_3D import dft_rotation_invariants_3d


class FPS16Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97);self.g=torch.randn(4,27,32,3);self.s=torch.randn(4,27,3)
        self.c=torch.tensor([17,19,23,27]);self.a=torch.tensor([5,0,17,26])
        for i,n in enumerate(self.c):self.g[i,n:]=0;self.s[i,n:]=0
        self.config=json.loads(Path('config/Ablation_Task4C_FPS16_1.1.json').read_text())

    def test_fps16_unique_center_excluded_and_fps6_prefix_unchanged(self):
        actual=method.fps_neighbors(self.s,self.c,self.a)
        expected=method.original.select_neighbors(self.s,self.c,self.a,'fps6')
        self.assertTrue(torch.equal(actual[:,:6],expected))
        for i,ids in enumerate(actual):
            self.assertEqual(len(torch.unique(ids)),16);self.assertNotIn(int(self.a[i]),ids.tolist())
            self.assertTrue(torch.all(ids<self.c[i]))
        with self.assertRaisesRegex(ValueError,'Not enough'):method.fps_neighbors(self.s,torch.tensor([16,19,23,27]),self.a)
        with self.assertRaisesRegex(ValueError,'Original center'):method.fps_neighbors(self.s,self.c,torch.tensor([-1,0,17,26]))

    def test_six_controls_exactly_reproduce_frozen_original_center(self):
        for c in self.config['candidates'][2:]:
            a,na=method.encode(self.g,self.s,self.c,self.a,c)
            b,nb=method.original.encode(self.g,self.s,self.c,self.a,c['base_method'])
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(na,nb))

    def test_16_descriptor_statistics_match_each_material_neighbor(self):
        for c in self.config['candidates'][:2]:
            x,neighbors=method.encode(self.g,self.s,self.c,self.a,c)
            old,_=method.original.encode(self.g,self.s,self.c,self.a,c['base_method'])
            torch.testing.assert_close(x[:,:,:95],old[:,:,:95],atol=1e-6,rtol=1e-6)
            g=self.g
            if c['base_method']=='c156':g,_=method.original.normalize(method.original.resample(g,48,'uniform'),self.c)
            scale=100 if c['base_method']=='p35_h0' else 1;weight=.5 if c['base_method']=='p35_h0' else 1
            values=[];batch=torch.arange(4)
            for rank in range(16):
                relative=g[batch,neighbors[:,rank]]-g[batch,self.a]
                values.append(dft_rotation_invariants_3d(relative.diff(dim=1)*scale,6,'gram',True)*weight)
            values=torch.stack(values,1)
            torch.testing.assert_close(x[:,0,95:118],values.mean(1),atol=2e-4,rtol=2e-4)
            torch.testing.assert_close(x[:,0,118:141],values.amax(1),atol=2e-4,rtol=2e-4)
            self.assertEqual(list(x.shape),[4,1,142])
            model=method.make_model(c['base_method']);self.assertEqual(sum(p.numel() for p in model.parameters()),c['parameters'])
            model(x).sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_replacement_gate_rejects_missing_original_center_even_with_17_lines(self):
        builder=data.Builder.__new__(data.Builder);builder.rejections={};builder.actual_trace_calls=0
        builder.plan=[{}];builder.physical={};builder.scene={'grid':None}
        seeds=np.zeros((27,3));seeds[:17,0]=np.arange(17)
        rows=[dict(line_count=16),dict(line_count=17,normalized_seeds=seeds,center=np.array([30.,0,0]),
              centroid=np.zeros(3),radius=1.,neighbor_distance=1.),
              dict(line_count=17,normalized_seeds=seeds,center=np.array([5.,0,0]),centroid=np.zeros(3),radius=1.,neighbor_distance=1.)]
        with patch.object(data.physical,'trace_batch',return_value=(rows,[],{})):
            result=builder.trace([{}],0)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['original_center_id'],5)
        self.assertEqual(builder.rejections,dict(fewer_than_17_clean_lines=1,original_center_removed=1))

    def test_gt_replacement_uses_real_interpolation_and_head_angle(self):
        builder=data.Builder.__new__(data.Builder)
        builder.spec=self.config;builder.index=0;builder.ids={'train':np.array([0])}
        builder.original={'train':dict(labels=np.array([1]),instance=np.array([7]),
            head_component=np.array([-8]),scale_id=np.array([0]))}
        builder.plan=[dict(neighbor_grid_scale=.25)]
        builder.axes=[np.array([-1.,0.,1.]) for _ in range(3)]
        velocity=np.zeros((3,3,3,3));velocity[...,0]=1.
        omega=np.zeros_like(velocity);omega[...,1]=1.
        builder.scene=dict(axes=builder.axes,velocity=velocity,omega=omega,
            lambda2=np.full((3,3,3),-2.),oyf=np.ones((3,3,3)),
            flow=dict(lambda2_threshold=-1.),gt=None,locator=None)
        builder.gt_points={7:np.zeros((1,3))};builder.rejections={}
        with patch('FMT_Utils.Task4C_HairpinBinary_2_1.sample_gt',return_value=(np.array([7]),None)), \
             patch.object(builder,'legal_center',return_value=True):
            sample=builder.propose('train',0,0)
            self.assertEqual(len(sample['seeds']),27)
            self.assertEqual(sample['instance'],7)
            self.assertEqual(sample['head_component'],-8)
            np.testing.assert_array_equal(sample['center'],sample['seeds'][0])
            self.assertGreater(np.linalg.norm(sample['center']),0.)
            builder.scene['omega']=velocity.copy()
            self.assertIsNone(builder.propose('train',0,0))
        self.assertEqual(builder.rejections,{'GT_head_membership':1})


if __name__=='__main__':unittest.main()
