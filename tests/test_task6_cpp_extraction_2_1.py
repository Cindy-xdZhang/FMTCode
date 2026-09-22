from pathlib import Path
import numpy as np
from FMT_Utils.Task6CPPExtraction_2_1 import merge_cpp_unique,spacing_h


def test_gap_is_strict_and_scales_with_physical_grid():
    for scale in [1.,8.]:
        a=np.array([[0,0,0],[1,0,0]],np.float32)*scale
        b=np.array([[2,0,0],[3,0,0]],np.float32)*scale
        assert len(merge_cpp_unique([a,b],scale)[0])==2
        assert len(merge_cpp_unique([a,b],np.nextafter(np.float32(scale),np.float32(np.inf)))[0])==1


def test_root_source_is_never_reused():
    # Root 0 absorbs 1, so its old endpoint is no longer free. Legacy code could
    # nevertheless append original source 0 again while processing source 2.
    lines=[np.array([[0,0,0],[1,0,0]],np.float32),
           np.array([[1.1,0,0],[2,0,0]],np.float32),
           np.array([[1.05,0,0],[1.5,0,0]],np.float32)]
    merged,sources,bridges=merge_cpp_unique(lines,.2)
    assert len(merged)==2 and len(bridges)==1
    assert [v['source'] for s in sources for v in s]==[0,1,2]


def test_verified_eight_curve_reference_is_preserved():
    from experiments.Filter_Task6_CoreLength_1_4 import read_cores
    root=Path(__file__).resolve().parents[1]/'outputs/Verify_Task6_CPPParity_1.1/resampled_sat_cpp_vtk'
    if not root.exists():return
    with np.load(root/'coordinates.npz') as z:axes=[z[a] for a in 'zyx']
    _,h=spacing_h(axes)
    assert abs(.2/h-3.975)<1e-5
    merged,_,_=merge_cpp_unique(read_cores(root/'points10.vtp'),4*h)
    reference=read_cores(root/'cpp_result.vtp')
    assert len(merged)==len(reference)==8
    for actual,expected in zip(merged,reference):np.testing.assert_array_equal(actual,expected)
