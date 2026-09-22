"""Fixed C++-style VTK extraction with a dimensionless endpoint gap."""
from pathlib import Path
import json
import time
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from FMT_Utils.Task6VortexCore_3D import polylines
from experiments.Extract_Task6_VortexCore_1_2 import write_lines, length


def spacing_h(axes_zyx):
    spacing=np.array([(a[-1]-a[0])/(len(a)-1) for a in axes_zyx[::-1]],np.float64)
    if not np.isfinite(spacing).all() or (spacing<=0).any():raise ValueError('Invalid physical grid')
    return spacing,float(spacing.min())


def merge_cpp_unique(lines,maximum_gap):
    """C++ sequential endpoint ordering; consume every source exactly once.

    The reference function omitted marking the root i as consumed, allowing an
    earlier root to appear in a later result. This version records and prevents
    that reuse while retaining its four endpoint cases and one-pass order.
    """
    if not np.isfinite(maximum_gap) or maximum_gap<=0:raise ValueError('Invalid merge gap')
    lines=[np.asarray(c,np.float32).copy() for c in lines]
    if any(c.ndim!=2 or c.shape[1]!=3 or len(c)<2 or not np.isfinite(c).all() for c in lines):
        raise ValueError('Invalid polylines')
    used=np.zeros(len(lines),bool);result=[];sources=[];bridges=[]
    threshold=np.float32(maximum_gap)
    for i in range(len(lines)):
        if used[i]:continue
        used[i]=True
        line=lines[i].copy();ids=[dict(source=i,reversed=False)]
        for j in range(len(lines)):
            if used[j]:continue
            start1,end1=line[0],line[-1];start2,end2=lines[j][0],lines[j][-1]
            pairs=[(end1,start2),(start1,end2),(start1,start2),(end1,end2)]
            selected=None
            for k,(a,b) in enumerate(pairs):
                gap=np.sqrt(np.sum((a-b)*(a-b),dtype=np.float32))
                if gap<threshold:selected=(k,a.copy(),b.copy(),float(gap));break
            if selected is None:continue
            k,a,b,gap=selected
            if k==0:
                line=np.concatenate([line,lines[j]]);ids.append(dict(source=j,reversed=False))
            elif k==1:
                line=np.concatenate([lines[j],line]);ids.insert(0,dict(source=j,reversed=False))
            elif k==2:
                line=np.concatenate([line[::-1],lines[j]])
                ids=[dict(source=v['source'],reversed=not v['reversed']) for v in ids[::-1]]+[dict(source=j,reversed=False)]
            else:
                line=np.concatenate([line,lines[j][::-1]]);ids.append(dict(source=j,reversed=True))
            used[j]=True
            bridges.append(dict(output=len(result),added_source=j,case=k,gap=gap,start=a.tolist(),end=b.tolist()))
        result.append(line);sources.append(ids)
    assert sorted(v['source'] for group in sources for v in group)==list(range(len(lines)))
    return result,sources,bridges


def extract_fixed(field,axes_zyx,folder,gap_h=4.,minimum_vertices=10):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    spacing,h=spacing_h(axes_zyx)
    if field.shape!=tuple(map(len,axes_zyx))+(3,) or not np.isfinite(field).all():raise ValueError('Invalid velocity field')
    assert vtk.vtkSMPTools.SetBackend('Sequential')
    image=vtk.vtkImageData();image.SetDimensions(*field.shape[:3][::-1])
    image.SetOrigin(*[float(a[0]) for a in axes_zyx[::-1]]);image.SetSpacing(*spacing)
    array=numpy_to_vtk(np.ascontiguousarray(field.reshape(-1,3),dtype=np.float32),deep=True)
    array.SetName('Velocity');image.GetPointData().AddArray(array);image.GetPointData().SetActiveVectors('Velocity')
    tetra=vtk.vtkDataSetTriangleFilter();tetra.SetInputData(image);tetra.SetTetrahedraOnly(False);tetra.Update()
    core=vtk.vtkVortexCore();core.SetHigherOrderMethod(False);core.SetFasterApproximation(False)
    core.SetInputData(tetra.GetOutput());core.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,'Velocity')
    started=time.monotonic();core.Update()
    if core.GetErrorCode():raise RuntimeError('VTK vortex-core extraction failed')
    strip=vtk.vtkStripper();strip.SetInputData(core.GetOutput());strip.SetJoinContiguousSegments(False);strip.SetMaximumLength(1000);strip.Update()
    raw=[c.astype(np.float32) for c in polylines(strip.GetOutput())]
    selected_ids=[i for i,c in enumerate(raw) if len(c)>=minimum_vertices]
    selected=[raw[i] for i in selected_ids]
    merged,sources,bridges=merge_cpp_unique(selected,gap_h*h)
    for name,lines in [('vtk_stripped',raw),('vtk_points10',selected),('vtk_merged',merged)]:write_lines(folder/(name+'.vtp'),lines)
    report=dict(version='Task6CPPExtraction_2.1',vtk_version=vtk.vtkVersion.GetVTKVersion(),
                grid='vtkImageData',velocity_dtype='float32',higher_order=False,faster_approximation=False,
                tetrahedra_only=False,stripper_join=False,stripper_maximum=1000,minimum_vertices=minimum_vertices,
                h=h,spacing_xyz=spacing.tolist(),maximum_gap_h=gap_h,maximum_gap=gap_h*h,
                represented_gap_float32=float(np.float32(gap_h*h)),comparison='<',
                native_lines=core.GetOutput().GetNumberOfLines(),stripped_lines=len(raw),selected_lines=len(selected),
                selected_raw_ids=selected_ids,merged_lines=len(merged),sources=sources,bridges=bridges,
                every_source_used_exactly_once=True,total_length=sum(map(length,merged)),
                lengths=list(map(length,merged)),seconds=time.monotonic()-started)
    (folder/'extraction.json').write_text(json.dumps(report,indent=2)+'\n')
    return merged,report
