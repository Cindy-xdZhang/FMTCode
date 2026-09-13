import unittest
import numpy as np
from FMT_Utils.DenseObserverPaths_3D import select_complete


class DensePathsTests(unittest.TestCase):
    def test_exact_count_and_domain_exclusions(self):
        initial=np.zeros((6,7,3));initial[:,:,0]=np.arange(6)[:,None]*.2
        def field(points,t):
            v=np.zeros_like(points);v[:,0]=.25
            v[points[:,0]>1]=np.nan
            return v
        chosen,audit=select_complete(field,initial,np.array([0.,1.,2.]),.1,3,batch_size=4)
        self.assertEqual(len(chosen),3)
        self.assertEqual(audit['selected_candidate_indices'],[0,1,2])
        self.assertEqual(audit['candidate_invalid_indices'],[3])
        np.testing.assert_array_equal(chosen,initial[:3])

    def test_insufficient_valid_candidates_fails(self):
        initial=np.zeros((2,7,3))
        with self.assertRaises(ValueError):
            select_complete(lambda p,t:np.zeros_like(p),initial,[0.,1.],.1,3)
