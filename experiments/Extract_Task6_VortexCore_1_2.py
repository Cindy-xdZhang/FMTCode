"""Extract VTK corelines and apply the fixed IVD candidate condition."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import vtk
from scipy.ndimage import binary_dilation
from scipy.interpolate import RegularGridInterpolator
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
from FMT_Utils.Task6VortexCore_3D import instantaneous_vorticity_deviation,rectilinear_field,extract_vortex_core,polylines


def write_lines(path,lines,point_arrays=None):
    poly=vtk.vtkPolyData();points=vtk.vtkPoints();cells=vtk.vtkCellArray()
    from vtk.util.numpy_support import numpy_to_vtk
    flat=np.concatenate(lines) if lines else np.empty((0,3))
    points.SetData(numpy_to_vtk(np.ascontiguousarray(flat),deep=True));poly.SetPoints(points)
    offset=0
    for line in lines:
        cells.InsertNextCell(len(line))
        for i in range(len(line)):cells.InsertCellPoint(offset+i)
        offset+=len(line)
    poly.SetLines(cells)
    if point_arrays:
        for name,values in point_arrays.items():
            array=numpy_to_vtk(np.concatenate(values) if values else np.empty(0),deep=True);array.SetName(name);poly.GetPointData().AddArray(array)
    writer=vtk.vtkXMLPolyDataWriter();writer.SetFileName(str(path));writer.SetInputData(poly);writer.SetDataModeToBinary()
    if writer.Write()!=1:raise IOError(path)


def join_native_segments(poly,h):
    clean=vtk.vtkCleanPolyData();clean.SetInputData(poly);clean.ToleranceIsAbsoluteOn();clean.SetAbsoluteTolerance(h*1e-7)
    clean.ConvertLinesToPointsOff();clean.Update()
    strip=vtk.vtkStripper();strip.SetInputConnection(clean.GetOutputPort());strip.JoinContiguousSegmentsOn();strip.SetMaximumLength(1000000);strip.Update()
    return polylines(strip.GetOutput())


def resample(line,spacing):
    # Retain every native VTK vertex. Extending either endpoint must not move
    # the evaluation points on an unchanged interior segment.
    points=[]
    for a,b in zip(line[:-1],line[1:]):
        distance=np.linalg.norm(b-a)
        if distance<=1e-12:continue
        count=max(1,int(np.ceil(distance/spacing)))
        points.append(a[None,:]+np.arange(count)[:,None]/count*(b-a)[None,:])
    return np.r_[np.concatenate(points),line[-1:]] if points else line[:1]


def valid_runs(line,mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return [line[a:b] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=2]


def length(line):return float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum())


def main():
    p=argparse.ArgumentParser();p.add_argument('--flow',required=True);p.add_argument('--index',type=int,default=75)
    p.add_argument('--methods',default='0,1');p.add_argument('--faster',action='store_true');args=p.parse_args()
    suffix='_fast1' if args.faster else ''
    folder=ROOT/'outputs/Verify_Task6_VortexCore_1.2'/args.flow/f'frame_{args.index:03d}'
    with np.load(folder/'coordinates.npz') as z:axes=[z[a] for a in 'zyx']
    velocity=np.load(folder/'relative_velocity.npy');h=min(float(np.mean(np.diff(a))) for a in axes)
    ivd=np.load(folder/'ivd.npy');audit=json.loads((folder/'ivd.json').read_text())
    assert audit['field']=='original_velocity_v'
    print(json.dumps({'event':'ivd','flow':args.flow,**audit}),flush=True)
    mask=ivd>audit['threshold'];shape=np.array(mask.shape)-1
    cellmask=np.zeros(tuple(shape),bool)
    for dz in (0,1):
        for dy in (0,1):
            for dx in (0,1):cellmask|=mask[dz:dz+shape[0],dy:dy+shape[1],dx:dx+shape[2]]
    # Four full cell layers protect derivatives used by both native VTK methods.
    # The global IVD and its maximum were computed before this optimization.
    expanded=binary_dilation(cellmask,iterations=4)
    ids=np.flatnonzero(expanded).astype(np.int64)
    grid=rectilinear_field(axes,velocity)
    extract=vtk.vtkExtractCells();extract.SetInputData(grid)
    cellids=vtk.vtkIdList();cellids.SetNumberOfIds(len(ids))
    for n,value in enumerate(ids):cellids.SetId(n,int(value))
    extract.SetCellList(cellids);extract.Update()
    print(json.dumps({'event':'vtk_support','candidate_cells':int(cellmask.sum()),'cells_with_four_layer_halo':len(ids)}),flush=True)
    interp=RegularGridInterpolator(axes,ivd,bounds_error=False,fill_value=0)
    reports=[]
    for mode in [int(v) for v in args.methods.split(',')]:
        started=time.time();poly=extract_vortex_core(extract.GetOutput(),higher_order=bool(mode),faster_approximation=args.faster)
        raw=join_native_segments(poly,h);accepted=[];context=[];context_mask=[]
        for line in raw:
            dense=resample(line,h/3)
            if len(dense)<2:continue
            valid=interp(dense[:,::-1])>audit['threshold']
            context.append(dense);context_mask.append(valid)
            accepted.extend(valid_runs(dense,valid))
        write_lines(folder/f'vtk_higher{mode}{suffix}_support_raw.vtp',raw)
        write_lines(folder/f'vtk_higher{mode}{suffix}_ivd.vtp',accepted)
        np.savez_compressed(folder/f'vtk_higher{mode}{suffix}_context.npz',
                            points=np.concatenate(context) if context else np.empty((0,3)),
                            candidate=np.concatenate(context_mask) if context else np.empty(0,bool),
                            offsets=np.r_[0,np.cumsum([len(c) for c in context])])
        report={'HigherOrderMethod':bool(mode),'FasterApproximation':args.faster,'raw_support_lines':len(raw),'ivd_lines':len(accepted),
                'ivd_total_length':sum(map(length,accepted)),'ivd_longest_length':max(map(length,accepted),default=0),
                'raw_total_length':sum(map(length,raw)),'seconds':time.time()-started,'status':'winding_not_yet_checked'}
        reports.append(report);print(json.dumps(report),flush=True)
    (folder/f'vtk_extraction{suffix}.json').write_text(json.dumps({'candidate_support':'all candidate cells plus four cell layers; global field and IVD retained','methods':reports},indent=2))


if __name__=='__main__':main()
