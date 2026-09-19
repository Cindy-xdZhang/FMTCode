"""Independent checks for weighted physical sampling and quota-constrained Poisson disks."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_FixedDataset_3_2 as b
from experiments.Task4C_FixedDataset_3_2 import validate_spec


def brute_dart(points,labels,radius,limits):
    selected=[];counts=np.zeros(2,int)
    for i,point in enumerate(points):
        label=labels[i]
        if counts[label]>=limits[label]:continue
        if selected and (np.linalg.norm(points[selected]-point,axis=1)<radius).any():continue
        selected.append(i);counts[label]+=1
        if sum(counts)==sum(limits):break
    return np.array(selected,np.int64)


class BalancedDatasetTests(unittest.TestCase):
    def test_quota_dart_matches_independent_all_pair_reference(self):
        rng=np.random.default_rng(19);points=rng.random((2400,3));labels=(points[:,0]<.23).astype(int)
        for radius in (.04,.095,.2):
            limits=np.array([90,180]);actual=b.dart_throw(points,labels,radius,limits)
            np.testing.assert_array_equal(actual,brute_dart(points,labels,radius,limits))
            d=cKDTree(points[actual]).query(points[actual],k=2)[0][:,1]
            self.assertTrue((d>=radius*(1-1e-12)).all())

    def test_opposite_labels_share_the_same_distance_constraint(self):
        points=np.array([[0.,0,0],[.01,0,0],[1.,0,0],[2.,0,0],[2.,0,0],[3.,0,0]])
        labels=np.array([0,1,1,0,1,1]);limits=np.array([1,2])
        np.testing.assert_array_equal(b.dart_throw(points,labels,1.,limits),[0,2,4])

    def test_radius_search_fills_both_quotas_and_reproduces(self):
        points=np.random.default_rng(20).random((14000,3));labels=(points[:,0]<.2).astype(int)
        limits=b.quotas(1500,2/3)
        selected,radius,log=b.poisson_disk(points,labels,limits,.06)
        np.testing.assert_array_equal(np.bincount(labels[selected],minlength=2),limits)
        np.testing.assert_array_equal(b.dart_throw(points,labels,radius,limits),selected)
        self.assertLess(len(b.dart_throw(points,labels,radius*1.001,limits)),sum(limits))
        self.assertGreater(len(log),2)

    def test_weighted_pool_exact_membership_uniform_volume_and_no_duplicates(self):
        axes=[np.linspace(0,1,11),np.linspace(0,1,6),np.linspace(0,1,4)]
        lam=-np.ones((4,6,11));oyf=np.ones_like(lam)
        cells=b.v2.candidate_cells(b.v2.candidate_mask(lam,oyf,0.))
        classify=lambda points:((points[:,0]<.25)&(points[:,1]<.6)).astype(int)
        kwargs=dict(axes=axes,lam=lam,oyf=oyf,threshold=0.,cells=cells,count=12000,seed=98301,chunk=5000,
            positive_fraction=2/3,classify=classify,box_low=np.array([[0.,0,0]]),box_high=np.array([[.25,.6,1.]]))
        points,stats=b.weighted_pool(**kwargs);again,_=b.weighted_pool(**kwargs)
        np.testing.assert_array_equal(points,again)
        np.testing.assert_array_equal(np.bincount(classify(points)),[4000,8000])
        self.assertEqual(len(np.unique(points,axis=0)),len(points))
        positive=points[classify(points)==1]
        np.testing.assert_allclose(positive.mean(0),[.125,.3,.5],atol=.009)
        self.assertEqual(stats['requested'],12000)

    def test_calibration_targets_two_to_one_after_different_survival_rates(self):
        p,survival=b.corrected_fraction([10000,20000],[9000,18000]);self.assertAlmostEqual(p,2/3)
        p,survival=b.corrected_fraction([10000,20000],[9000,16000])
        self.assertAlmostEqual(p*survival[1]/((1-p)*survival[0]),2.)
        self.assertGreater(p,2/3)

    def test_frozen_geometry_folds_counts_and_one_random_setting(self):
        spec=json.loads(Path('config/mainExp_Task4C_FixedDataset_3.2.json').read_text());validate_spec(spec)
        self.assertEqual(spec['sampling']['pool_per_flow']*2,40000000)
        self.assertEqual(spec['sampling']['samples_per_flow']*2,8000000)
        api=b.adapter(spec,pilot=True)
        for name in ('trace_curves','cut_and_clean','assign_instances','sample_gt'):
            self.assertIs(getattr(api,name),getattr(b.v2,name))


if __name__=='__main__':unittest.main()
