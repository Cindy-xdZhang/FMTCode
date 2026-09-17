"""Regression cases where absolute-value ranking misrepresents positive support."""
import unittest
import numpy as np
from FMT_Utils.Task4C_SupportView_1_2 import summarize_bundle


def fixture(deltas):
    delta=np.asarray(deltas,dtype=float)
    margin=1.0;probability=1/(1+np.exp(-margin));changed=1/(1+np.exp(-(margin-delta)))
    return dict(patch_indices=np.asarray([[0,i,i+2] for i in range(len(delta))]),
        patch_margin_delta=delta,probability=probability,margin=margin,
        patch_probability=changed,patch_probability_delta=probability-changed,
        local_shape_delta=np.zeros((1,32)))


class SupportViewTests(unittest.TestCase):
    def test_negative_extreme_is_not_positive_evidence(self):
        result=summarize_bundle(fixture([-8.,.3,1.2,-2.]),1)
        self.assertEqual(result['strongest_support'],2)
        self.assertEqual(result['positive_order'],[2,1])
        self.assertTrue(result['previous_absolute_max_was_negative'])

    def test_no_positive_window_remains_empty(self):
        result=summarize_bundle(fixture([-2.,0.,-.1]),1)
        self.assertIsNone(result['strongest_support'])
        self.assertEqual(result['positive_order'],[])

    def test_equal_scores_keep_source_order(self):
        self.assertEqual(summarize_bundle(fixture([2.,2.,1.]),1)['positive_order'],[0,1,2])

    def test_wrong_score_sign_fails(self):
        data=fixture([1.,-2.]);data['patch_margin_delta']*=-1
        with self.assertRaises(ValueError):summarize_bundle(data,1)

    def test_wrong_probability_delta_fails(self):
        data=fixture([1.,-2.]);data['patch_probability_delta']*=-1
        with self.assertRaises(ValueError):summarize_bundle(data,1)


if __name__=='__main__':unittest.main()
