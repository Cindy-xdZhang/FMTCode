"""True 3D voxel rendering and interactive mesh export for Task4-b 4.1.

No orthographic axis projections, interpolation, model fitting or point
subsampling. All overview samples are rendered. Detail uses the largest
sampled positive VortexId, selected by voxel count without model scores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

NAMES = ['Ordinary streamwise', 'Ordinary spanwise', 'Hairpin head', 'Hairpin leg']
COLORS = ['#0072B2', '#E69F00', '#CC79A7', '#009E73']


def rgb(color):
    return tuple(int(color[i:i+2],16)/255 for i in (1,3,5))


def points_mesh(points, labels, ids):
    vertices=vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(np.ascontiguousarray(points,np.float64),deep=True))
    mesh=vtk.vtkPolyData()
    mesh.SetPoints(vertices)
    for name,array in [('class_id',labels),('VortexIds',ids)]:
        value=numpy_to_vtk(np.ascontiguousarray(array),deep=True)
        value.SetName(name)
        mesh.GetPointData().AddArray(value)
    mesh.GetPointData().SetActiveScalars('class_id')
    return mesh


def actor(points, label, spacing, opacity):
    poly=points_mesh(points,np.full(len(points),label,np.int8),np.zeros(len(points),np.int32))
    cube=vtk.vtkCubeSource()
    cube.SetXLength(float(spacing[0])*.96)
    cube.SetYLength(float(spacing[1])*.96)
    cube.SetZLength(float(spacing[2])*.96)
    mapper=vtk.vtkGlyph3DMapper()
    mapper.SetInputData(poly)
    mapper.SetSourceConnection(cube.GetOutputPort())
    mapper.ScalingOff(); mapper.OrientOff(); mapper.ScalarVisibilityOff()
    result=vtk.vtkActor(); result.SetMapper(mapper)
    prop=result.GetProperty()
    prop.SetColor(*rgb(COLORS[label])); prop.SetOpacity(opacity)
    prop.SetInterpolationToFlat(); prop.SetAmbient(.24); prop.SetDiffuse(.76)
    return result


def text(renderer,value,x,y,size=23,color=(.16,.19,.24)):
    a=vtk.vtkTextActor(); a.SetInput(value)
    a.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    a.SetPosition(x,y)
    p=a.GetTextProperty(); p.SetFontFamilyToArial(); p.SetFontSize(size); p.SetColor(*color)
    renderer.AddViewProp(a)
    return a


def render(points,labels,spacing,low,high,path,title,subtitle,ordinary_opacity=1.):
    window=vtk.vtkRenderWindow(); window.SetOffScreenRendering(1)
    window.SetSize(1800,1150); window.SetAlphaBitPlanes(1); window.SetMultiSamples(0)
    renderer=vtk.vtkRenderer(); window.AddRenderer(renderer)
    renderer.SetBackground(.975,.982,.99)
    renderer.UseDepthPeelingOn(); renderer.SetMaximumNumberOfPeels(100); renderer.SetOcclusionRatio(.02)
    for k in range(4):
        if np.any(labels==k):
            renderer.AddActor(actor(points[labels==k],k,spacing,ordinary_opacity if k<2 else 1.))
    bounds=tuple(v for pair in zip(low,high) for v in pair)
    outline=vtk.vtkOutlineSource(); outline.SetBounds(*bounds)
    om=vtk.vtkPolyDataMapper(); om.SetInputConnection(outline.GetOutputPort())
    oa=vtk.vtkActor(); oa.SetMapper(om); oa.GetProperty().SetColor(.72,.76,.80); oa.GetProperty().SetLineWidth(1.)
    renderer.AddActor(oa)
    camera=renderer.GetActiveCamera()
    center=(low+high)/2
    direction=np.array([.7,-1.25,.83]); direction/=np.linalg.norm(direction)
    camera.SetPosition(*(center+np.linalg.norm(high-low)*2*direction))
    camera.SetFocalPoint(*center); camera.SetViewUp(0,0,1)
    camera.ParallelProjectionOff(); camera.SetViewAngle(27)
    renderer.ResetCamera(*bounds); camera.Zoom(.92)
    axes=vtk.vtkCubeAxesActor(); axes.SetBounds(*bounds); axes.SetCamera(camera)
    axes.SetXTitle('x / streamwise'); axes.SetYTitle('y / spanwise'); axes.SetZTitle('z / vertical')
    axes.SetFlyModeToOuterEdges(); axes.SetGridLineLocation(vtk.vtkCubeAxesActor.VTK_GRID_LINES_FURTHEST)
    axes.XAxisMinorTickVisibilityOff(); axes.YAxisMinorTickVisibilityOff(); axes.ZAxisMinorTickVisibilityOff()
    axes.XAxisLabelVisibilityOff(); axes.YAxisLabelVisibilityOff(); axes.ZAxisLabelVisibilityOff()
    for prop in [axes.GetXAxesLinesProperty(),axes.GetYAxesLinesProperty(),axes.GetZAxesLinesProperty()]:
        prop.SetColor(.60,.65,.70)
    for i in range(3):
        axes.GetTitleTextProperty(i).SetColor(.22,.25,.29)
        axes.GetLabelTextProperty(i).SetColor(.35,.39,.44)
        axes.GetTitleTextProperty(i).SetFontSize(18)
        axes.GetLabelTextProperty(i).SetFontSize(15)
    renderer.AddActor(axes)
    text(renderer,title,.027,.951,30)
    for k in range(4):
        if np.any(labels==k):
            text(renderer,NAMES[k],.027+.245*k,.901,22,rgb(COLORS[k]))
    ranges='   '.join(f'{axis}: {lo:.3g} to {hi:.3g}' for axis,lo,hi in zip('xyz',low,high))
    text(renderer,subtitle+'\n'+ranges,.027,.019,19)
    renderer.ResetCameraClippingRange()
    window.Render()
    capture=vtk.vtkWindowToImageFilter(); capture.SetInput(window)
    capture.SetInputBufferTypeToRGB(); capture.ReadFrontBufferOff(); capture.Update()
    writer=vtk.vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
    meta=dict(file=path.name,sample_count=len(points),class_support=np.bincount(labels,minlength=4).tolist(),
        perspective=not bool(camera.GetParallelProjection()),camera_position=list(camera.GetPosition()),
        focal_point=list(camera.GetFocalPoint()),view_up=list(camera.GetViewUp()),bounds=list(bounds),
        spacing=spacing.tolist(),ordinary_opacity=ordinary_opacity,depth_peeling_used=bool(renderer.GetLastRenderingUsedDepthPeeling()))
    window.Finalize()
    return meta


def export_interactive(points,labels,ids,spacing,name,path):
    import plotly.graph_objects as go
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    triangles=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    figure=go.Figure()
    for k in range(4):
        centers=points[labels==k]
        vertices=(centers[:,None,:]+signs[None,:,:]*spacing[None,None,:]*.48).reshape(-1,3)
        faces=(triangles[None,:,:]+8*np.arange(len(centers))[:,None,None]).reshape(-1,3)
        figure.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
            color=COLORS[k],name=f'{NAMES[k]} ({len(centers):,})',flatshading=True,opacity=.36 if k<2 else 1.,
            showlegend=True,hoverinfo='name',lighting=dict(ambient=.5,diffuse=.8),visible=True))
    figure.update_layout(title=f'{name} | FMT four-class predictions | all {len(points):,} fitted samples',
        scene=dict(aspectmode='data',xaxis_title='x / streamwise',yaxis_title='y / spanwise',zaxis_title='z / vertical',
            camera=dict(eye=dict(x=1.4,y=-2,z=1.4),projection=dict(type='perspective'))),
        legend=dict(orientation='h',y=.99),margin=dict(l=5,r=5,t=100,b=10),height=850,
        updatemenus=[dict(type='buttons',direction='right',x=0,y=1.1,buttons=[
            dict(label='All four classes',method='restyle',args=[{'visible':[True,True,True,True]}]),
            dict(label='Hairpin only',method='restyle',args=[{'visible':['legendonly','legendonly',True,True]}]),
            dict(label='Opaque all',method='restyle',args=[{'opacity':[1,1,1,1]}]),
            dict(label='Transparent ordinary',method='restyle',args=[{'opacity':[.36,.36,1,1]}])])])
    figure.write_html(str(path),include_plotlyjs=True,full_html=True,config={'displaylogo':False,'responsive':True})


def main(root):
    root=Path(root); output=root/'figures_3d_r2'; output.mkdir(exist_ok=True)
    report=json.loads((root/'four_class_report.json').read_text())
    prediction=root/'predictions/fmt_only_seed7068.npz'
    if hashlib.sha256(prediction.read_bytes()).hexdigest()!=report['original_prediction_sha256']:
        raise RuntimeError('Prediction identity mismatch')
    with np.load(root/'cache.npz') as cache, np.load(prediction) as p:
        labels=p['predicted_labels']; target=p['targets']
        if not np.array_equal(labels,target):
            raise RuntimeError('These views are for the audited zero-error fit')
        data={k:cache[k] for k in ['volume_codes','voxel_indices_xyz','vortex_ids','seeds_xyz','labels']}
    manifest=dict(experiment=report['experiment'],purpose='3D inspection of fitted sample labels; not dense inference',
        prediction_sha256=report['original_prediction_sha256'],all_predictions_equal_proxy=True,
        classes=report['classes'],colors=COLORS,smoothing=False,subsampling=False,views={},
        hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'))
    for volume,name in enumerate(['channel','tbl']):
        mask=data['volume_codes']==volume
        points=data['seeds_xyz'][mask].astype(float); y=labels[mask]; ids=data['vortex_ids'][mask]
        with np.load(root/f'{name}_label_volume.npz') as grid:
            low=grid['domain_min_xyz']; high=grid['domain_max_xyz']; spacing=(high-low)/grid['resolution_xyz']
        # Reconstruct float64 voxel centers from integer indices and saved grid.
        points=low+(data['voxel_indices_xyz'][mask]+.5)*spacing
        mesh=points_mesh(points,y,ids)
        writer=vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(output/f'{name}_predictions_points.vtp')); writer.SetInputData(mesh); writer.Write()
        overview=render(points,y,spacing,low,high,output/f'{name}_four_classes_3d.png',f'{name.upper()} | FMT four-class prediction',
            f'All {len(points):,} fitted samples | ordinary voxels translucent | physical xyz aspect',ordinary_opacity=.36)
        h=ids>0
        hlow=points[h].min(0)-spacing; hhigh=points[h].max(0)+spacing
        hairpin=render(points[h],y[h],spacing,hlow,hhigh,output/f'{name}_hairpins_3d.png',f'{name.upper()} | Hairpin-labelled samples',
            f'All {int(h.sum()):,} valid hairpin voxels | head / leg | physical xyz aspect')
        unique,counts=np.unique(ids[h],return_counts=True)
        largest=int(unique[np.argmax(counts)]); detail=ids==largest
        dlow=points[detail].min(0)-2*spacing; dhigh=points[detail].max(0)+2*spacing
        detail_meta=render(points[detail],y[detail],spacing,dlow,dhigh,output/f'{name}_hairpin_detail_3d.png',f'{name.upper()} | Hairpin instance {largest}',
            f'Largest sampled instance by voxel count | all {int(detail.sum()):,} voxels | no smoothing')
        detail_meta['vortex_id']=largest
        export_interactive(points,y,ids,spacing,name.upper(),output/f'{name}_interactive_3d.html')
        manifest['views'][name]=dict(overview=overview,hairpins=hairpin,detail=detail_meta)
        print(name,json.dumps(manifest['views'][name]),flush=True)
    (output/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/Verify_Task4B_VelocityCurlMemorization_4.1')
    main(parser.parse_args().output)
