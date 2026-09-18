"""One-factor checks against the frozen Task4-c multi-center implementations."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from FMT_Utils.Task4C_OriginalCenter_1_1 import encode,make_model,original_center_indices
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import fourier_tokens,resample,neighbor_indices
from FMT_Utils.Task4C_LinePooling_4_3 import line_fmt
from FMT_Utils.FMT_V8_Search_2_1 import select_features,pooling_candidates
from experiments import Task4C_OriginalCenter_1_1 as runner


class OriginalCenterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(177)
        self.g=torch.randn(3,27,32,3)
        self.s=torch.randn(3,27,3)
        self.c=torch.tensor([10,19,27]);self.a=torch.tensor([5,11,23])
        for i,n in enumerate(self.c):self.g[i,n:]=0;self.s[i,n:]=0

    def test_matches_only_original_anchor_of_frozen_encoder(self):
        b=torch.arange(3)
        p=next(p for p in pooling_candidates() if p['id']=='p35')
        old=line_fmt(self.g,self.s,self.c)
        reference=select_features(old[:,:,:233],p)[b,self.a]
        actual,_=encode(self.g,self.s,self.c,self.a,'p35_h0')
        self.assertEqual(actual.shape,(3,1,142))
        torch.testing.assert_close(actual[:,0,:141],reference,atol=1e-5,rtol=1e-5)
        n=neighbor_indices(self.s,self.c,'fps6')
        reference=fourier_tokens(resample(self.g,48,'uniform'),self.c,n,'p35')[b,self.a]
        actual,selected=encode(self.g,self.s,self.c,self.a,'c156')
        self.assertTrue(torch.equal(selected,n[b,self.a]))
        torch.testing.assert_close(actual[:,0],reference,atol=1e-5,rtol=1e-5)

    def test_missing_seed_rejected_without_nearest_substitution(self):
        m=dict(center=self.s[torch.arange(3),self.a].numpy().astype(float),
               centroid=np.zeros((3,3)),radius=np.ones(3),counts=self.c.numpy(),neighbor_distance=np.ones(3))
        ids,ratio=original_center_indices(self.s.numpy(),m)
        np.testing.assert_array_equal(ids,self.a.numpy())
        m['center'][1]+=100
        ids,_=original_center_indices(self.s.numpy(),m)
        self.assertEqual(ids[1],-1)
        with self.assertRaisesRegex(ValueError,'Missing original center'):
            encode(self.g,self.s,self.c,torch.tensor(ids),'c156')

    def test_same_network_capacity_and_single_vector_pooling(self):
        for name,total in [('p35_h0',76738),('c156',992386)]:
            model=make_model(name).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),total)
            x,_=encode(self.g,self.s,self.c,self.a,name)
            result=model(x)
            with self.assertRaisesRegex(ValueError,'Exactly one original-center'):
                model(x.repeat(1,2,1))
            network=model.network if name=='p35_h0' else model
            h=network.line(x[:,0,:141])
            torch.testing.assert_close(result,network.head(torch.cat((h,h),-1)),atol=1e-6,rtol=1e-6)
            loss=result.square().mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertFalse(any(isinstance(m,torch.nn.modules.conv._ConvNd) for m in model.modules()))

    def test_paired_control_preserves_descriptors_and_initialization(self):
        spec=json.loads(Path(runner.CONFIG).read_text())
        for single,control in zip(spec['candidates'][:2],spec['candidates'][2:]):
            a,_=runner.encode_candidate(self.g,self.s,self.c,self.a,single)
            b,_=runner.encode_candidate(self.g,self.s,self.c,self.a,control)
            torch.testing.assert_close(a[:,0],b[torch.arange(3),self.a],atol=1e-5,rtol=1e-5)
            torch.manual_seed(96721);m1=runner.make_candidate_model(single)
            torch.manual_seed(96721);m2=runner.make_candidate_model(control)
            for p,q in zip(m1.parameters(),m2.parameters()):self.assertTrue(torch.equal(p,q))

    def test_identical_subset_keeps_original_rows_and_centers_in_all_arms(self):
        spec=json.loads(Path(runner.CONFIG).read_text())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';out=root/'out';(out/'centers').mkdir(parents=True)
            (out/'subsets').mkdir();config=root/'config.json';config.write_text('{}')
            lock=dict(complete=True,identity=dict(config_sha256=runner.sha(config)),flows={})
            for flow in ('channel','tbl'):
                folder=source/'physical'/flow/'train';folder.mkdir(parents=True)
                np.save(folder/'geometry.npy',self.g.numpy());np.save(folder/'seeds.npy',self.s.numpy())
                np.savez(folder/'metadata.npz',counts=self.c.numpy(),labels=[0,1,1],instance=[0,1,2])
                np.savez(out/'centers'/f'{flow}_train.npz',center_ids=[5,-1,23])
                np.savez(out/'subsets'/f'{flow}_train.npz',row_ids=[0,2],center_ids=[5,23])
                lock['flows'][flow]=dict(train=dict(sha256=runner.sha(out/'subsets'/f'{flow}_train.npz')))
            (out/'subset_verification.json').write_text(json.dumps(lock))
            spec.update(source_output=str(source),output=str(out),_config=str(config),_active_seed=96721,
                        subset_expected_counts=dict(train=4))
            for i,candidate in enumerate(spec['candidates']):
                d=runner.Dataset(runner.arm_spec(spec,i),'train',candidate,device='cpu')
                np.testing.assert_array_equal(d.rows,[0,2,0,2]);np.testing.assert_array_equal(d.anchor,[5,23,5,23])
                np.testing.assert_array_equal(d.labels,[0,1,0,1]);d.encode(candidate,batch=2)
                self.assertEqual(list(d.clean.shape),[4,27 if runner.rotating(candidate) else 1,142])


if __name__=='__main__':unittest.main()
