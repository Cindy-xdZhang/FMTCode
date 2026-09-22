import unittest
import numpy as np
from FMT_Utils.Task4C_RawCurl_2_1 import trace,clean


class RawCurlTests(unittest.TestCase):
    def test_magnitude_changes_distance(self):
        axes=[np.linspace(-10,10,3)]*3
        for magnitude in (1.,10.,100.):
            field=np.empty((3,3,3,3));field[:]=np.array([1.,2.,3.])*magnitude
            halves,_,reason=trace(axes,field,np.zeros((1,3)),.0002,50)
            np.testing.assert_allclose(halves[1][0][-1],np.array([1.,2.,3.])*magnitude*.01,atol=1e-12)
            self.assertEqual(len(halves[1][0]),51)
            self.assertTrue((reason==5).all())

    def test_boundary_quarter_and_stagnation(self):
        p=np.arange(66)[:,None]*np.array([[.001,.002,.003]])
        for count,expected in [(8,False),(9,True)]:
            result=clean(-p[:count+1],p,[1,5],.0002,[35],{})
            self.assertEqual(bool(result[1][0]),expected)
        self.assertFalse(clean(-p[:10],p,[6,5],.0002,[35],{})[1][0])
        self.assertTrue(clean(-p,p,[5,5],.0002,[35,50,65],{})[1].all())

    def test_boundary_stops_without_clamping(self):
        axes=[np.linspace(-1,1,3)]*3;w=np.ones((3,3,3,3))
        halves,_,r=trace(axes,w,np.array([[.95,.95,.95]]),.02,10)
        self.assertEqual(r[0,1],1)
        self.assertTrue((halves[1][0]<=1).all())
        self.assertEqual(len(halves[1][0]),3)


if __name__=='__main__':unittest.main()
