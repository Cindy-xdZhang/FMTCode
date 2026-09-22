import tempfile,unittest
from pathlib import Path
import numpy as np
from FMT_Utils.Task6SquareCylinder_2_5 import read_amira
from FMT_Utils.Task6CPPExtraction_2_5 import filter_merged_corelines


class SquareCylinderTests(unittest.TestCase):
    def test_binary_tag_xyz_components_and_physical_bounds(self):
        array=np.arange(2*3*4*3,dtype='<f4').reshape(2,3,4,3)
        header=b'# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n\ndefine Lattice 4 3 2\nParameters {\nBoundingBox -12 20 -4 4 0 6\nCoordType "uniform"\n}\nLattice { float[3] Data } @1\n\n# Data section follows\n@1\n'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.am';p.write_bytes(header+array.tobytes()+b'\n')
            field,axes,meta=read_amira(p)
            np.testing.assert_array_equal(field,array)
            self.assertEqual(field[1,2,3,2],71)
            self.assertEqual(meta['payload_offset'],len(header))
            np.testing.assert_array_equal([a[[0,-1]].tolist() for a in axes],[[0,6],[-4,4],[-12,20]])
            p.write_bytes(header+array.tobytes()[:-4])
            with self.assertRaises(ValueError):read_amira(p)
    def test_16h_after_merging_includes_bridge_and_equality(self):
        curves=[np.array([[0.,0,0],[15.99,0,0]]),np.array([[0.,0,0],[16.,0,0]])]
        retained,report=filter_merged_corelines(curves,1.)
        self.assertEqual(len(retained),1);self.assertEqual(report['retained_merged_ids'],[1])

if __name__=='__main__':unittest.main()
