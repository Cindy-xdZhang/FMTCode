import unittest
import numpy as np
from FMT_Utils.FlowMapData_3D import cross_offsets
from FMT_Utils.Task6Recovery_3D import signed_fmt,inverse_signed_fmt,linear_inverse_matrix


class TestRecovery(unittest.TestCase):
    def test_full_complex_coefficients_roundtrip_with_line_identity(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(4,7,32,3)).cumsum(2)
        x-=x[:,:,:1]
        x+=cross_offsets()[None,:,None]
        np.testing.assert_allclose(inverse_signed_fmt(signed_fmt(x,16),16),x,atol=1e-12)
        for k in (6,10,16):
            matrix,bias=linear_inverse_matrix(k)
            z=signed_fmt(x,k)
            np.testing.assert_allclose((z@matrix+bias).reshape(x.shape),inverse_signed_fmt(z,k),atol=2e-13)

    def test_geometry_phase_and_direction_are_retained(self):
        x=np.broadcast_to(cross_offsets()[None,:,None],(1,7,32,3)).copy()
        x[:,:,:,0]+=np.linspace(0,1,32)
        y=np.broadcast_to(cross_offsets()[None,:,None],(1,7,32,3)).copy()
        y[:,:,:,1]+=np.linspace(0,1,32)
        self.assertGreater(np.linalg.norm(signed_fmt(x)-signed_fmt(y)),.01)


if __name__=="__main__": unittest.main()
