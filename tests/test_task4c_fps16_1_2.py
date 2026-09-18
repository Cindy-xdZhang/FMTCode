"""Data-rule contracts for the fixed-center FPS16 rebuild 1.2 (no VTK integration needed)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from FMT_Utils import Task4C_FPS16_Data_1_2 as data
from FMT_Utils.Task4C_Multiscale_4_1 import center_and_neighbors as frozen_center_and_neighbors


def synthetic_field(n=8):
    axes=[np.arange(n+1,dtype=float) for _ in range(3)]
    components=np.zeros((n,n,n),np.int64)
    components[3:5,3:5,3:5]=7          # a 2x2x2-cell head; a 27 stencil at 1.0 spacing cannot stay inside it
    oyf=np.ones((n+1,n+1,n+1))
    return axes,components,oyf


class FPS16DataRules(unittest.TestCase):
    def setUp(self):
        self.config=json.loads(Path('config/Ablation_Task4C_FPS16_1.2.json').read_text())
        self.previous=json.loads(Path('config/Ablation_Task4C_FPS16_1.1.json').read_text())

    def test_center_rule_identical_to_frozen_but_neighbours_only_need_the_domain(self):
        axes,components,oyf=synthetic_field()
        row=dict(head_component=7,cell_ids=np.flatnonzero(components.ravel()==7))
        for scale in ({'neighbor_grid_scale':1.},{'neighbor_grid_scale':.25}):
            for number in range(6):
                old=frozen_center_and_neighbors(row,components,axes,oyf,number,scale,11)
                new=data.head_center_and_neighbors(row,components,axes,oyf,number,scale,11)
                np.testing.assert_array_equal(old['center'],new['center'])
                self.assertEqual(old['source_cell'],new['source_cell']);self.assertEqual(old['neighbor_distance'],new['neighbor_distance'])
                np.testing.assert_array_equal(new['seeds'][0],new['center'])
                self.assertEqual(len(new['seeds']),27)                       # every stencil point is inside the domain
                if scale['neighbor_grid_scale']==1.:self.assertLess(len(old['seeds']),17)  # the frozen same-head filter cannot reach 17
        # The center itself keeps the frozen rule: negative oyf at the center rejects the proposal in both versions.
        negative=-np.ones_like(oyf)
        self.assertIsNone(frozen_center_and_neighbors(row,components,axes,negative,0,{'neighbor_grid_scale':.25},11))
        self.assertIsNone(data.head_center_and_neighbors(row,components,axes,negative,0,{'neighbor_grid_scale':.25},11))
        # A stencil that leaves the domain keeps only in-domain neighbours.
        edge=dict(head_component=1,cell_ids=np.array([0]));components[0,0,0]=1
        sample=data.head_center_and_neighbors(edge,components,axes,oyf,0,{'neighbor_grid_scale':1.},11)
        self.assertLess(len(sample['seeds']),27);np.testing.assert_array_equal(sample['seeds'][0],sample['center'])

    def test_failing_rows_flags_missing_center_or_fewer_than_17_lines(self):
        seeds=np.zeros((4,27,3));seeds[:,:,0]=np.arange(27)[None]
        m=dict(center=np.zeros((4,3)),centroid=np.zeros((4,3)),radius=np.ones(4),counts=np.array([27,17,16,27]),neighbor_distance=np.ones(4))
        m['center'][3]=[50.,0,0]                    # center line absent
        anchor,bad=data.failing_rows(seeds,m,17)
        np.testing.assert_array_equal(anchor,[0,0,0,-1]);np.testing.assert_array_equal(bad,[False,False,True,True])

    def test_stage_schedule(self):
        b=data.Builder.__new__(data.Builder);b.rule=self.config['replacement']
        self.assertEqual([b.stage(a) for a in (0,63,64,999,1000,1999)],['original_head']*2+['any_head']*2+['relaxed']*2)
        self.assertIsNone(b.stage(2000))

    def test_relaxed_head_region_center_is_labelled_by_gt_membership(self):
        axes,components,oyf=synthetic_field()
        b=data.Builder.__new__(data.Builder);b.components=components;b.axes=axes;b.oyf=-np.ones_like(oyf)  # no oyf>0 anywhere
        b.scene=dict(gt=None,locator=None);b.rejections={};b.box_cache={}
        pool=[dict(head_component=7,cell_ids=np.flatnonzero(components.ravel()==7))]
        rng=np.random.default_rng(3)
        with patch.object(data,'sample_gt',return_value=(np.array([5]),None)):
            s=b.relaxed_head_region(rng,1,pool,7,{'neighbor_grid_scale':.5},2,'train',0,-1)
        self.assertEqual((s['label'],s['instance'],s['head_component']),(1,5,7))
        self.assertTrue(np.all(s['center']>=2.) and np.all(s['center']<=6.))      # cell box [3,5] with one spacing margin
        self.assertEqual(len(s['seeds']),27);np.testing.assert_array_equal(s['seeds'][0],s['center'])
        with patch.object(data,'sample_gt',return_value=(np.array([-1]),None)):
            s=b.relaxed_head_region(rng,2,pool,7,{'neighbor_grid_scale':.5},2,'train',1,9)
        self.assertEqual((s['label'],s['instance']),(0,-1))

    def test_trace_gate_requires_17_clean_lines_and_the_original_center(self):
        b=data.Builder.__new__(data.Builder);b.rejections={};b.actual_trace_calls=0;b.rule={'minimum_valid_lines':17}
        b.plan=[{}];b.physical={};b.scene={'grid':None}
        seeds=np.zeros((27,3));seeds[:17]=data.stencil(np.zeros(3),1.)[:17]
        rows=[dict(line_count=16),
              dict(line_count=17,normalized_seeds=seeds,center=np.array([30.,0,0]),centroid=np.zeros(3),radius=1.,neighbor_distance=1.),
              dict(line_count=17,normalized_seeds=seeds,center=np.array([0.,0,0]),centroid=np.zeros(3),radius=1.,neighbor_distance=1.)]
        with patch.object(data.physical,'trace_batch',return_value=(rows,[dict(reason='fewer_than_10_full_length_clean_lines')],{})):
            result=b.trace([{}],0)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['original_center_id'],0)
        np.testing.assert_array_equal(result[0]['stencil_slots'][:17],np.arange(17));self.assertTrue(np.isnan(result[0]['physical_seeds'][17:]).all())
        self.assertEqual(b.rejections,dict(fewer_than_10_full_length_clean_lines=1,fewer_than_17_clean_lines=1,original_center_removed=1))

    def test_assemble_changes_labels_only_for_relaxed_rows(self):
        keys=('labels','instance','scale_id','head_component','counts','center','centroid','radius','neighbor_distance',
              'local_grid_scale','nearest_train_center_distance','nearest_same_head_train_center_distance')
        n=4;old=dict(labels=np.array([1,1,0,0],np.int8),instance=np.array([3,3,-1,-1],np.int32),scale_id=np.zeros(n,np.int32),
            head_component=np.array([5,5,6,6],np.int32),counts=np.full(n,20,np.int16),center=np.tile((np.arange(n)[:,None]+1)*10.,(1,3)),
            centroid=np.zeros((n,3)),radius=np.ones(n),neighbor_distance=np.ones(n),local_grid_scale=np.ones(n),
            nearest_train_center_distance=np.zeros(n),nearest_same_head_train_center_distance=np.zeros(n))
        assert set(old)==set(keys)
        def row(slot,label,instance,relaxed,center):
            center=np.array(center,float);grid=data.stencil(center,1.)
            r=dict(replacement_slot=slot,replacement_attempt=3,relaxed=relaxed,neighbor_filter=data.FILTER_RELAXED if relaxed else data.FILTER_IN_DOMAIN,
                label=label,instance=instance,scale_id=0,head_component=old['head_component'][slot],line_count=17,center=center,
                centroid=np.zeros(3),radius=1.,neighbor_distance=1.,local_grid_scale=1.,nearest_train_center_distance=0.,
                nearest_same_head_train_center_distance=0.,geometry=np.zeros((27,32,3),np.float32),normalized_seeds=np.zeros((27,3),np.float32))
            r['normalized_seeds'][:17]=grid[:17];r['physical_seeds'],r['stencil_slots']=data.physical_seeds_of_row(r);return r
        b=data.Builder.__new__(data.Builder);b.rule={'minimum_valid_lines':17};b.flow='x';b.nbase=100;b.lower_original=np.zeros(100,bool)
        b.axes=[np.arange(0.,400.,50.) for _ in range(3)];field=np.full((8,8,8),-5.);b.lambda2=field;b.oyf=np.ones((8,8,8));b.threshold=-1.
        with tempfile.TemporaryDirectory() as temp:
            src=Path(temp)/'physical/x/validation';src.mkdir(parents=True);b.source=Path(temp)/'physical/x'
            g=np.zeros((n,27,32,3),np.float32);g[:,0,0,0]=np.arange(n)+1;np.save(src/'geometry.npy',g)
            s=np.zeros((n,27,3),np.float32)
            for i in range(n):s[i]=data.stencil(old['center'][i],1.)
            np.save(src/'seeds.npy',s)
            out=Path(temp)/'out';out.mkdir()
            rows=[row(1,0,-1,True,[100.,50,50]),row(2,0,-1,False,[200.,50,50])]
            report=data.assemble(b,'validation',out,np.arange(n),rows,old,lambda p:'h',pilot=True)
            with np.load(out/'metadata.npz') as z:m={k:z[k] for k in z.files}
            with np.load(out/'replacement_index.npz') as z:idx={k:z[k] for k in z.files}
            np.testing.assert_array_equal(m['labels'],[1,0,0,0]);np.testing.assert_array_equal(idx['replaced'],[False,True,True,False])
            np.testing.assert_array_equal(idx['relaxed'],[False,True,False,False]);np.testing.assert_array_equal(idx['neighbor_filter'],[0,2,1,0])
            np.testing.assert_array_equal(idx['original_center_id'],[0,0,0,0]);np.testing.assert_array_equal(idx['source_label'],old['labels'])
            self.assertEqual((report['replaced'],report['relaxed'],report['label_changes']),(2,1,1))
            with np.load(out/data.ATTRIBUTE_FILE) as z:attr={k:z[k] for k in z.files}
            self.assertEqual(attr['seed_points'].shape,(n,27,3));np.testing.assert_array_equal(attr['exact_stencil_coordinates'],[False,True,True,False])
            valid=np.arange(27)[None]<m['counts'][:,None];np.testing.assert_array_equal(m['counts'],[20,17,17,20])
            np.testing.assert_array_equal(np.isfinite(attr['seed_points']).all(-1),valid)
            np.testing.assert_allclose(attr['seed_points'][1,:17],data.stencil(np.array([100.,50,50]),1.)[:17])
            np.testing.assert_allclose(attr['seed_points'][0,:20],data.stencil(np.array([10.,10,10]),1.)[:20],atol=1e-5)
            np.testing.assert_array_equal(attr['head_candidate'],valid);np.testing.assert_array_equal(attr['oyf_positive'],valid)
            self.assertEqual(report['lines'],dict(total=74,head_candidate=74,oyf_positive=74,retained_lines_not_head_candidate=0,new_neighbours_not_head_candidate=0,relaxed_centers_not_head_candidate=0))
            kept=np.load(out/'geometry.npy');self.assertEqual(float(kept[0,0,0,0]),1.);self.assertEqual(float(kept[3,0,0,0]),4.)
            (out/'bad').mkdir()
            with self.assertRaises(AssertionError):      # a non-relaxed replacement may not change the class
                data.assemble(b,'validation',out/'bad',np.arange(n),[row(1,0,-1,False,[100.,50,50])],old,lambda p:'h',pilot=True)

    def test_physical_seeds_match_stencil_and_line_attributes_follow_step1(self):
        center=np.array([3.3,4.1,2.2]);grid=data.stencil(center,.5)
        row=dict(line_count=20,center=center,neighbor_distance=.5,radius=2.,centroid=np.array([1.,1.,1.]),
                 normalized_seeds=((grid[[0,5,3,26,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]-1.)/2.).astype(np.float32))
        points,slots=data.physical_seeds_of_row(row)
        np.testing.assert_allclose(points[:20],grid[slots[:20]]);self.assertEqual(slots[0],0);self.assertTrue(np.isnan(points[20:]).all())
        bad=dict(row,normalized_seeds=np.roll(row['normalized_seeds'],1,axis=0))
        with self.assertRaisesRegex(ValueError,'slot 0'):data.physical_seeds_of_row(bad)
        axes=[np.arange(9,dtype=float) for _ in range(3)];lam=np.full((9,9,9),-2.);lam[:,:,:4]=0.;oyf=np.ones((9,9,9));oyf[:,:4]=-1.
        pts=np.full((1,27,3),np.nan);pts[0,:8]=[[1.5,6,6],[6,6,6],[6,1,6],[1,1,6],[7.9,7.9,7.9],[0.1,0.1,0.1],[6,6,20],[2,6,6]]
        a=data.line_attributes(pts,axes,lam,oyf,-1.)
        np.testing.assert_array_equal(a['inside'][0,:8],[True,True,True,True,True,True,False,True])
        np.testing.assert_array_equal(a['oyf_positive'][0,:8],[True,True,False,False,True,False,False,True])
        np.testing.assert_array_equal(a['head_candidate'][0,:8],[False,True,False,False,True,False,False,False])
        self.assertTrue(np.isnan(a['lambda2'][0,8:]).all() and not a['head_candidate'][0,8:].any())

    def test_config_keeps_frozen_arms_and_full_template(self):
        self.assertEqual(self.config['candidates'],self.previous['candidates'])
        self.assertEqual(self.config['source_files'],self.previous['source_files'])
        self.assertEqual(self.config['expected_counts'],dict(train=196960,validation=3000,test=11320))
        self.assertEqual(self.config['replacement']['minimum_valid_lines'],17)
        self.assertEqual(self.config['replacement']['normal_attempts'],1000)
        self.assertNotIn('subset_source',self.config)


if __name__=='__main__':unittest.main()
