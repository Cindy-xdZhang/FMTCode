import copy
import json
from pathlib import Path
from types import SimpleNamespace, FunctionType
import tempfile
import unittest
import numpy as np
import torch
from FMT_Utils import Task4C_V3Training_3_1 as data
from experiments import Task4C_BallQueryBaselines_2_5 as reference
from experiments import Task4C_V3Training_3_1 as run


class StreamingTests(unittest.TestCase):
    def test_cpu_geometry_exactly_matches_frozen_export_and_order(self):
        rng=np.random.default_rng(11);parts=[]
        for fi in range(2):
            n=40;curves=rng.normal(size=(n,3,32,3)).astype(np.float32);seeds=rng.normal(size=(n,3))
            for k in (6,16):
                neighbors=np.array([[j for j in range(n) if j!=i][:k] for i in range(n)])
                source=SimpleNamespace(curves=curves,seeds=seeds,neighbors=neighbors,k=k)
                source.gather=lambda samples,lengths,s=source:data.Source.gather(s,samples,lengths)
                samples=np.array([2,5,19,23,31]);parts=[dict(source=source,samples=samples)]
                ids=np.array([[0,7],[0,1],[0,13],[0,0],[0,5]])
                actual,counts=data.baseline_geometry(parts,ids)
                m=dict(label=np.arange(n)%2,component=np.arange(n),instance=np.arange(n),half_arc_lengths=np.ones((n,3,2)),half_step_counts=np.ones((n,3,2),np.int32),local_grid_scale=np.ones(n))
                expected,_,_=reference.bundle_function(k)(curves,seeds,m,neighbors,samples[ids[:,1]//3],ids[:,1]%3,[70,100,130],.001)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(counts,np.full(len(ids),k+1))

    def test_host_tokens_match_slice_and_shuffled_tensor_indices(self):
        a=np.random.default_rng(3).random((17,1,142)).astype(np.float32)
        h=data.HostTokens(a,'cpu');t=torch.tensor(a)
        torch.testing.assert_close(h[2:9],t[2:9],atol=0,rtol=0)
        ids=torch.tensor([9,1,8,1]);torch.testing.assert_close(h[ids],t[ids],atol=0,rtol=0)

    def test_streaming_p35_normalizer_is_identical_to_original_numpy_fit(self):
        values=np.random.default_rng(7).normal(size=(211,1,142)).astype(np.float32);values[:,:,-1]=1
        host=SimpleNamespace(clean=data.HostTokens(values,'cpu'),device='cpu')
        original=SimpleNamespace(clean=torch.tensor(values),device='cpu')
        c=dict(base_method='p35_h0')
        a=run.fit_normalizer(host,c);b=run.fit_normalizer(original,c)
        torch.testing.assert_close(a[0],b[0],atol=0,rtol=0);torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
        self.assertEqual(a[2],b[2])

    def test_ten_methods_use_same_source_rows_and_single_seed(self):
        s=run.fmt_spec();self.assertEqual(s['expected_counts'],dict(train=16521207,validation=1835691,test=4930236))
        self.assertEqual(len(s['candidates']),4);self.assertEqual(s['final']['seeds'],[96611])
        for index in range(6):
            b,_=run.baseline_spec(index)
            self.assertEqual(b['training']['seeds'],[96611]);self.assertEqual(len(b['methods']),1)
            self.assertEqual(b['source_audit_sha256'],s['source_audit_sha256'])
            self.assertEqual(b['neighbor_output'],s['neighbor_output'])
            self.assertEqual({r:b['expected_counts'][r] for r in s['expected_counts']},s['expected_counts'])
            for key in ('batch_size','learning_rate','weight_decay','dropout','lr_patience'):
                self.assertEqual(b['training'][key],s['training'][key])
            self.assertEqual(b['training']['epochs'],s['final']['epochs'])
            self.assertEqual(b['training']['patience'],s['final']['patience'])


if __name__=='__main__':unittest.main()
