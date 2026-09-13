import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from FLowUtils.flowDatasetUtils.JHTDB_VTK import write_frame,read_frame,check_geometry,geometry,add_velocity,save,read_grid


def test_non_cubic_vtk_roundtrip(tmp_path):
    axes=[np.linspace(3,3.3,2),np.linspace(-.9,-.6,3),np.linspace(.2,.5,4)]
    z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
    values=np.stack([x+10,y+20,z+30],axis=-1).astype(np.float32)
    path=tmp_path/'frame.vtk'
    write_frame(path,values,axes,1.062)
    restored,grid=read_frame(path)
    np.testing.assert_array_equal(restored,values)
    check_geometry(grid,axes)
    assert vtk_to_numpy(grid.GetFieldData().GetArray('TimeValue'))[0]==1.062
    with path.open('rb') as f:
        assert b'BINARY\nDATASET STRUCTURED_GRID' in f.read(150)


def test_combined_time_arrays(tmp_path):
    axes=[np.array([0.,.1])]*3
    grid=geometry(axes)
    for t in [1.,1.002,1.004]:
        add_velocity(grid,np.full((2,2,2,3),t,dtype=np.float32),f'velocity_{t:.7f}')
    path=tmp_path/'all.vtk'
    save(grid,path)
    result=read_grid(path)
    assert result.GetPointData().GetNumberOfArrays()==3
    for t in [1.,1.002,1.004]:
        np.testing.assert_array_equal(vtk_to_numpy(result.GetPointData().GetArray(f'velocity_{t:.7f}')),np.full((8,3),t,dtype=np.float32))
