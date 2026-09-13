"""Version 4.4: user-specified physical IVD thresholds; ground truth only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from experiments.Visualize_Task4B_VelocityCurl_3D import COLORS, NAMES, rgb, text, render


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):
            h.update(chunk)
    return h.hexdigest()


def render_volume(labels,low,high,spacing,path,name,threshold,fraction,spec):
    # Padding supplies transparent background; nearest interpolation keeps the
    # categorical voxel boxes, with no smoothed or interpolated class labels.
    values=np.pad((labels+1).astype(np.uint8),1)
    image=vtk.vtkImageData()
    image.SetDimensions(*values.shape[::-1]); image.SetSpacing(*spacing)
    image.SetOrigin(*(low-.5*spacing))
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR))
    mapper=vtk.vtkFixedPointVolumeRayCastMapper(); mapper.SetInputData(image)
    mapper.SetBlendModeToComposite()
    mapper.SetSampleDistance(float(spacing.min())*.5)
    mapper.SetAutoAdjustSampleDistances(False)
    mapper.SetImageSampleDistance(1.)
    color=vtk.vtkColorTransferFunction(); color.AddRGBPoint(0,1,1,1)
    opacity=vtk.vtkPiecewiseFunction(); opacity.AddPoint(0,0)
    for k in range(4):
        color.AddRGBPoint(k+1,*rgb(COLORS[k]))
        opacity.AddPoint(k+1,spec['ordinary_opacity_per_minimum_voxel_distance'] if k<2 else spec['hairpin_opacity_per_minimum_voxel_distance'])
    prop=vtk.vtkVolumeProperty(); prop.SetColor(color); prop.SetScalarOpacity(opacity)
    prop.SetScalarOpacityUnitDistance(float(spacing.min()))
    prop.SetInterpolationTypeToNearest(); prop.ShadeOff()
    volume=vtk.vtkVolume(); volume.SetMapper(mapper); volume.SetProperty(prop)
    renderer=vtk.vtkRenderer(); renderer.SetBackground(.975,.982,.99); renderer.AddVolume(volume)
    bounds=tuple(v for pair in zip(low,high) for v in pair)
    outline=vtk.vtkOutlineSource(); outline.SetBounds(*bounds)
    om=vtk.vtkPolyDataMapper(); om.SetInputConnection(outline.GetOutputPort())
    oa=vtk.vtkActor(); oa.SetMapper(om); oa.GetProperty().SetColor(.62,.67,.72); renderer.AddActor(oa)
    # True spatial axis triad, placed at the minimum-coordinate corner.
    axes=vtk.vtkAxesActor(); axes.SetPosition(*low)
    length=float(min((high-low)[:2]))*.22
    axes.SetTotalLength(length,length,length)
    axes.SetXAxisLabelText('x'); axes.SetYAxisLabelText('y'); axes.SetZAxisLabelText('z')
    axes.AxisLabelsOff()
    renderer.AddActor(axes)
    camera=renderer.GetActiveCamera(); center=(low+high)/2
    direction=np.array([.7,-1.25,.83]); direction/=np.linalg.norm(direction)
    camera.SetPosition(*(center+2*np.linalg.norm(high-low)*direction)); camera.SetFocalPoint(*center)
    camera.SetViewUp(0,0,1); camera.ParallelProjectionOff(); camera.SetViewAngle(27)
    window=vtk.vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(*spec['image_size'])
    window.SetMultiSamples(0); window.AddRenderer(renderer)
    renderer.ResetCamera(*bounds); camera.Zoom(.92)
    text(renderer,f'{name.upper()}  |  Proxy ground truth: IVD > {threshold:g}',.027,.951,30)
    for k in range(4):
        text(renderer,NAMES[k],.027+.245*k,.901,22,rgb(COLORS[k]))
    count=int(np.sum(labels>=0))
    text(renderer,f'Full grid | {count:,} labelled vortex voxels | threshold {threshold:.7g} | vortex area {100*fraction:.3f}%\n'
        'Physical xyz aspect | ordinary classes translucent | no model predictions',.027,.019,19)
    renderer.ResetCameraClippingRange(); window.Render()
    capture=vtk.vtkWindowToImageFilter(); capture.SetInput(window); capture.SetInputBufferTypeToRGB(); capture.ReadFrontBufferOff(); capture.Update()
    writer=vtk.vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
    result=dict(file=path.name,perspective=not bool(camera.GetParallelProjection()),bounds=list(bounds),
        camera_position=list(camera.GetPosition()),sample_count=count,volume_renderer=mapper.GetClassName(),
        scalar_interpolation='nearest',ordinary_opacity=spec['ordinary_opacity_per_minimum_voxel_distance'],
        hairpin_opacity=spec['hairpin_opacity_per_minimum_voxel_distance'])
    window.Finalize()
    return result


def main(config, render_only=False):
    vtk.vtkMultiThreader.SetGlobalMaximumNumberOfThreads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    config=Path(config); spec=json.loads(config.read_text())
    assert spec['threshold_policy']=='user_specified_per_flow_physical_ivd'
    assert spec['vortex_comparison']=='strictly_greater'
    root=Path(spec['source']); out=Path(spec['output']); out.mkdir(parents=True,exist_ok=True)
    if render_only:
        summary=json.loads((out/'summary.json').read_text())
        display=out/'figures_r2'; display.mkdir(exist_ok=True)
        reports={}
        for name in ['channel','tbl']:
            source=out/f'{name}_groundtruth.npz'
            assert sha(source)==summary['flows'][name]['groundtruth_sha256']
            with np.load(source) as d:
                low=d['domain_min_xyz']; high=d['domain_max_xyz']
                reports[name]=render_volume(d['labels'],low,high,(high-low)/d['resolution_xyz'],
                    display/f'{name}_groundtruth_3d.png',name,float(d['threshold']),
                    summary['flows'][name]['new_vortex_fraction'],spec['render'])
        (display/'render_summary.json').write_text(json.dumps(dict(script_sha256=sha(__file__),
            job_id=os.environ.get('SLURM_JOB_ID'),label_arrays_unchanged=True,flows=reports),indent=2)+'\n')
        return
    if (out/'summary.json').exists():
        raise RuntimeError('Frozen output exists; choose a new version for further changes')
    (out/'config_snapshot.json').write_bytes(config.read_bytes())
    original=json.loads((root/'build_summary.json').read_text())
    summary=dict(version=spec['version'],operation=spec['operation'],training_run=False,predictions_read=False,
        config_sha256=sha(config),source_build_sha256=sha(root/'build_summary.json'),
        script_sha256=sha(__file__),renderer_sha256=sha(Path(__file__).with_name('Visualize_Task4B_VelocityCurl_3D.py')),
        hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),flows={})
    for name in ['channel','tbl']:
        source=root/f'{name}_label_volume.npz'
        if sha(source)!=original['flows'][name]['label_volume_sha256']:
            raise RuntimeError(f'{name} original label-volume hash mismatch')
        with np.load(source) as d:
            ivd=d['ivd']; ids=d['vortex_ids']; cosine=d['abs_cosine']; old_labels=d['labels']
            low=d['domain_min_xyz']; high=d['domain_max_xyz']; resolution=d['resolution_xyz']
            gt_ivd=d['original_gt_cell_ivd']
        assert np.isfinite(gt_ivd).all() and np.isfinite(ivd[ids>0]).all()
        previous_threshold=min(float(gt_ivd.min()),float(ivd[ids>0].min()))
        threshold=float(spec['thresholds'][name])
        assert np.isfinite(threshold) and threshold>previous_threshold
        vortex=ivd>threshold
        parallel=cosine*cosine>=.5-4*np.finfo(float).eps
        labels=np.where(ids>0,np.where(parallel,3,2),np.where(parallel,0,1)).astype(np.int8)
        labels[~vortex|~np.isfinite(cosine)]=-1
        # An independent implementation of the intended change: retain old
        # orientation labels and only raise the scalar vortex threshold.
        expected=old_labels.copy(); expected[ivd<=threshold]=-1
        assert np.array_equal(labels,expected)
        # Hairpin membership does not override the user-specified vortex cutoff.
        assert np.all(labels[ivd<=threshold]==-1)
        spacing=(high-low)/resolution
        np.savez_compressed(out/f'{name}_groundtruth.npz',labels=labels,vortex_ids=ids,domain_min_xyz=low,
            domain_max_xyz=high,resolution_xyz=resolution,threshold=threshold)
        report=dict(threshold_policy=spec['threshold_policy'],threshold_source=spec['threshold_source'],
            original_gt_minimum_ivd=float(gt_ivd.min()),grid_gt_minimum_ivd=float(ivd[ids>0].min()),
            old_threshold=previous_threshold,new_threshold=threshold,old_vortex_fraction=float((ivd>previous_threshold).mean()),
            new_vortex_fraction=float(vortex.mean()),removed_voxel_count=int(np.sum((ivd>previous_threshold)&~vortex)),
            excluded_grid_hairpin_count=int(np.sum((ids>0)&~vortex)),
            excluded_original_hairpin_cell_count=int(np.sum(gt_ivd<=threshold)),
            full_grid_count=int(ivd.size),class_support=np.bincount(labels[labels>=0],minlength=4).tolist(),
            grid_hairpin_coverage=float(vortex[ids>0].mean()),original_hairpin_cell_coverage=float(np.mean(gt_ivd>threshold)),
            source_label_sha256=sha(source),groundtruth_sha256=sha(out/f'{name}_groundtruth.npz'),
            independent_rethreshold_check=True)
        print(name,json.dumps(report),flush=True)
        report['overview']=render_volume(labels,low,high,spacing,out/f'{name}_groundtruth_3d.png',name,threshold,float(vortex.mean()),spec['render'])
        z,y,x=np.nonzero((ids>0)&(labels>=0)); points=low+(np.column_stack((x,y,z))+.5)*spacing
        hairpin_labels=labels[z,y,x]
        report['hairpins']=render(points,hairpin_labels,spacing,points.min(0)-spacing,points.max(0)+spacing,
            out/f'{name}_groundtruth_hairpins_3d.png',f'{name.upper()} | Proxy GT hairpins: IVD > {threshold:g}',
            f'{len(points):,} retained hairpin voxels on the complete GT grid | no streamline-validity filter | no model prediction')
        summary['flows'][name]=report
        print(name,'rendered',flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/Other_Task4B_ProxyGTThreshold_4.4.json')
    p.add_argument('--render-only',action='store_true',help='Redraw frozen labels without changing label files or their summary')
    args=p.parse_args()
    main(args.config,args.render_only)
