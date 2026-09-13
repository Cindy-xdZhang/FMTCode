"""Legacy binary STRUCTURED_GRID velocity files, including named time arrays."""
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def dimensions(grid):
    result=[0,0,0]
    grid.GetDimensions(result)
    return tuple(result)


def geometry(axes):
    x,y,z=[np.asarray(a,dtype=np.float64) for a in axes]
    zz,yy,xx=np.meshgrid(z,y,x,indexing='ij')
    points=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
    grid=vtk.vtkStructuredGrid()
    grid.SetDimensions(len(x),len(y),len(z))
    vp=vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(points,deep=True))
    grid.SetPoints(vp)
    return grid


def add_velocity(grid, values, name='velocity'):
    values=np.asarray(values,dtype=np.float32)
    dims=dimensions(grid)
    if values.shape != (dims[2],dims[1],dims[0],3) or not np.isfinite(values).all():
        raise ValueError('Velocity must be finite (Z,Y,X,3)')
    array=numpy_to_vtk(values.reshape(-1,3),deep=True)
    array.SetName(name)
    grid.GetPointData().AddArray(array)
    if grid.GetPointData().GetVectors() is None:
        grid.GetPointData().SetActiveVectors(name)


def save(grid,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial.vtk')
    writer=vtk.vtkStructuredGridWriter()
    writer.SetFileName(str(temp))
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    if writer.Write()!=1 or writer.GetErrorCode():
        raise IOError(f'Failed to write {path}')
    temp.replace(path)


def write_frame(path,values,axes,time):
    grid=geometry(axes)
    add_velocity(grid,values)
    t=numpy_to_vtk(np.array([time],dtype=np.float64),deep=True)
    t.SetName('TimeValue')
    grid.GetFieldData().AddArray(t)
    save(grid,path)


def read_grid(path):
    reader=vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllFieldsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    grid=reader.GetOutput()
    if reader.GetErrorCode() or not grid.IsA('vtkStructuredGrid') or not grid.GetNumberOfPoints():
        raise ValueError(f'Invalid STRUCTURED_GRID file: {path}')
    return grid


def read_frame(path,name='velocity'):
    grid=read_grid(path)
    dims=dimensions(grid)
    array=grid.GetPointData().GetArray(name)
    if array is None or array.GetNumberOfComponents()!=3:
        raise ValueError(f'Missing 3-component point array {name}')
    values=vtk_to_numpy(array).reshape(dims[2],dims[1],dims[0],3).copy()
    return values,grid


def check_geometry(grid,axes):
    x,y,z=axes
    assert dimensions(grid)==(len(x),len(y),len(z))
    xyz=vtk_to_numpy(grid.GetPoints().GetData()).reshape(len(z),len(y),len(x),3)
    np.testing.assert_array_equal(xyz[...,0],np.broadcast_to(x,(len(z),len(y),len(x))))
    np.testing.assert_array_equal(xyz[...,1],np.broadcast_to(np.asarray(y)[None,:,None],(len(z),len(y),len(x))))
    np.testing.assert_array_equal(xyz[...,2],np.broadcast_to(np.asarray(z)[:,None,None],(len(z),len(y),len(x))))
