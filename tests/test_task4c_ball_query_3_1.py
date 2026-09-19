import unittest
import numpy as np
from FMT_Utils import Task4C_BallQuery_2_5 as old
from FMT_Utils import Task4C_BallQuery_3_1 as new


class ChunkedBallTests(unittest.TestCase):
    def test_exact_dense_sparse_boundary_and_chunk_equivalence(self):
        points=np.random.default_rng(96611).normal(size=(191,3))
        for radius in (.001,.8,10.):
            expected=old.ball_tables(points,radius)
            for chunk in (19,64):
                actual=new.ball_tables(points,radius,chunk=chunk)
                for key in expected: np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
        points=np.column_stack((np.arange(40.),np.zeros((40,2))))
        expected=old.ball_tables(points,6.)
        actual=new.ball_tables(points,6.,chunk=7)
        for key in expected: np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)


if __name__=='__main__': unittest.main()
