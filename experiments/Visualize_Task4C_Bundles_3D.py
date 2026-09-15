"""Inspect physical vortex-line bundles, binary predictions and annotated surfaces.

Export selected real examples on Ibex, then build a self-contained local viewer.
This reads completed predictions, never model weights or unfinished checkpoints.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np

CONFIG = 'config/Other_Task4C_BundleVisualization_1.1.json'
COLORS = {'hairpin': '#218990', 'non_hairpin': '#c08650', 'ground_truth': '#a6a0b4'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def identity(config):
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        config_sha256=sha(config), source_sha256=sha(__file__),
        host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'))


def diverse_centers(centers, count, seed):
    """Label-blind nested display order, independent of classifier scores."""
    centers = np.asarray(centers, dtype=np.float64)
    span = np.ptp(centers, axis=0); span[span == 0] = 1
    x = (centers-centers.min(0))/span
    rng = np.random.default_rng(seed); first = int(rng.integers(len(x)))
    selected = []; minimum = np.full(len(x), np.inf); index = first
    for _ in range(min(count, len(x))):
        selected.append(index)
        minimum = np.minimum(minimum, np.sum((x-x[index])**2, axis=1))
        minimum[selected] = -1
        index = int(np.argmax(minimum))
    return np.asarray(selected, dtype=np.int64)


def f1_score(labels, probability, threshold=.5):
    y = np.asarray(labels).astype(bool); p = np.asarray(probability) >= threshold
    tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
    return 2*tp/max(2*tp+fp+fn, 1)


def export_examples(config, output):
    spec = json.loads(Path(config).read_text(encoding='utf-8'))
    assert sha(spec['data_config']) == spec['data_config_sha256']
    data_spec = json.loads(Path(spec['data_config']).read_text())
    root = Path(spec['physical_output']); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = {}; selected = {}; artifacts = {}
    for fi, flow in enumerate(data_spec['flows']):
        folder = root/'physical'/flow['name']
        prepared = json.loads((folder/'preparation.json').read_text())
        for si, split in enumerate(spec['splits']):
            key = flow['name']+'/'+split; part = folder/split
            files = prepared['splits'][split]['files']
            for name in ('metadata.npz', 'geometry.npy'): assert sha(part/name) == files[name]
            with np.load(part/'metadata.npz') as z: meta = {k:z[k] for k in z.files}
            geometry = np.load(part/'geometry.npy', mmap_mode='r')
            assert geometry.shape == (len(meta['labels']), 27, 32, 3)
            ids = diverse_centers(meta['center'], spec['maximum_bundles_per_flow_split'], spec['selection_seed']+100*fi+si)
            counts = meta['counts'][ids].astype(np.int64)
            assert np.all((counts >= 10) & (counts <= 27))
            g = np.asarray(geometry[ids], dtype=np.float64)
            physical = g * meta['radius'][ids, None, None, None] + meta['centroid'][ids, None, None, :]
            valid = np.arange(27)[None] < counts[:, None]
            physical[~valid] = 0
            recovered = (physical-meta['centroid'][ids, None, None, :])/meta['radius'][ids, None, None, None]
            assert np.allclose(recovered[valid], g[valid], atol=1e-10)
            geometry32 = physical.astype(np.float32)
            error = float(np.abs(physical[valid]-geometry32[valid]).max())
            assert error <= max(float(np.abs(physical[valid]).max())*1e-7, 1e-7)
            pack = dict(geometry=geometry32, row_ids=ids, counts=counts,
                        **{k:meta[k][ids] for k in ('labels','center','head_component','instance','scale_id','radius','neighbor_distance')})
            source[key] = meta; selected[key] = ids; artifacts[key] = pack
    models = []; pending = []
    for model in spec['models']:
        path = Path(model['result'])
        if not path.is_file(): pending.append(dict(model, reason='no_completed_result')); continue
        result = json.loads(path.read_text())
        pred_path = path.parent/'predictions.npz'
        encoding_path = path.parents[2]/'encoding.json'
        encoding = json.loads(encoding_path.read_text())
        for flow in data_spec['flows']:
            for split in spec['splits']:
                key = flow['name']+'/'+split
                assert encoding['splits'][key]['files']['metadata.npz'] == sha(root/'physical'/key/'metadata.npz')
        assert sha(pred_path) == result['predictions_sha256']
        assert result['seed'] == model['seed'] and result['threshold'] == spec['fixed_threshold'] == .5
        with np.load(pred_path) as predictions:
            for split in spec['splits']:
                expected = np.concatenate([source[f['name']+'/'+split]['labels'] for f in data_spec['flows']])
                probability = predictions[split+'_probability']
                assert np.array_equal(predictions[split+'_labels'], expected)
                assert probability.shape == expected.shape and np.isfinite(probability).all()
                assert np.all((probability >= 0) & (probability <= 1))
                field = 'training' if split == 'train' else split
                assert abs(f1_score(expected, probability)-result[field]['pooled']['f1']) < 1e-12
                offset = 0
                for fi, flow in enumerate(data_spec['flows']):
                    key = flow['name']+'/'+split; meta = source[key]; n = len(meta['labels']); sl = slice(offset, offset+n)
                    assert np.all(predictions[split+'_flow_index'][sl] == fi)
                    for name in ('head_component','scale_id'):
                        assert np.array_equal(predictions[split+'_'+name][sl], meta[name])
                    assert abs(f1_score(meta['labels'], probability[sl])-result[field]['per_flow'][flow['name']]['f1']) < 1e-12
                    artifacts[key]['p_'+model['id']] = probability[sl][selected[key]]
                    offset += n
        models.append(dict(model, version=result['version'], parameters=result['parameters'],
            result_sha256=sha(path), predictions_sha256=sha(pred_path), input_encoding_sha256=sha(encoding_path), source_identity=result['identity'],
            metrics={s:result['training' if s == 'train' else s] for s in spec['splits']}))
    if not models: raise ValueError('No completed predictions to visualize')
    entries = {}
    for key, pack in artifacts.items():
        filename = key.replace('/', '_')+'.npz'
        np.savez_compressed(output/filename, **pack)
        part = root/'physical'/key
        entries[key] = dict(file=filename, sha256=sha(output/filename), selected=len(pack['row_ids']),
            population=len(source[key]['labels']), metadata_sha256=sha(part/'metadata.npz'), geometry_sha256=sha(part/'geometry.npy'),
            selection_uses_labels_or_scores=False, physical_coordinates_restored=True)
    manifest = dict(version=spec['version'], identity=identity(config), config=spec, flows=data_spec['flows'],
        models=models, pending=pending, splits=entries, threshold=.5, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_scope='Real selected bundles and completed per-bundle predictions; metrics use full per-flow split', weights_read=False)
    write(output/'manifest.json', manifest)
    print(json.dumps(dict(status='PASS', models=[m['id'] for m in models], pending=[m['id'] for m in pending],
        selected_bundles=sum(e['selected'] for e in entries.values()), manifest_sha256=sha(output/'manifest.json'))), flush=True)


def array_json(value, dtype):
    value = np.ascontiguousarray(value, dtype=dtype)
    return dict(dtype=str(value.dtype), shape=list(value.shape), data=base64.b64encode(value.tobytes()).decode('ascii'))


def read_vtk(path):
    import vtk
    reader = vtk.vtkDataSetReader(); reader.SetFileName(str(path))
    reader.ReadAllScalarsOn(); reader.ReadAllVectorsOn(); reader.ReadAllFieldsOn(); reader.Update()
    if reader.GetErrorCode(): raise ValueError(f'Cannot read {path}')
    return reader.GetOutput()


def gt_surface(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    grid = read_vtk(path)
    assert grid.GetCellData().GetArray('VortexIds') is not None
    surface = vtk.vtkDataSetSurfaceFilter(); surface.SetInputData(grid)
    triangles = vtk.vtkTriangleFilter(); triangles.SetInputConnection(surface.GetOutputPort()); triangles.Update()
    mesh = vtk.vtkPolyData(); mesh.DeepCopy(triangles.GetOutput())
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1,3)
    assert mesh.GetNumberOfCells() == len(faces)
    ids = vtk_to_numpy(mesh.GetCellData().GetArray('VortexIds')).astype(np.int32)
    assert np.all(ids >= 0)
    # Export only geometry and instance identifiers, not unrelated flow fields.
    keep = vtk.vtkIntArray(); keep.DeepCopy(mesh.GetCellData().GetArray('VortexIds')); keep.SetName('VortexIds')
    mesh.GetPointData().Initialize(); mesh.GetCellData().Initialize(); mesh.GetCellData().AddArray(keep)
    return mesh


def save_vtp(mesh, path):
    import vtk
    writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(path)); writer.SetInputData(mesh)
    writer.SetDataModeToBinary(); assert writer.Write() == 1


def bundle_mesh(pack, models):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    counts = pack['counts']; points = []; owners = []
    for i, count in enumerate(counts):
        points.append(pack['geometry'][i, :count].reshape(-1,3)); owners.extend([i]*int(count))
    xyz = np.concatenate(points); owners = np.asarray(owners, dtype=np.int64); number = len(owners)
    mesh = vtk.vtkPolyData(); vertices = vtk.vtkPoints(); vertices.SetData(numpy_to_vtk(xyz, deep=True)); mesh.SetPoints(vertices)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(number+1, dtype=np.int64)*32, deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(xyz), dtype=np.int64), deep=True)); mesh.SetLines(cells)
    for name, values in [('bundle_row', pack['row_ids'][owners]), ('gt_label', pack['labels'][owners]),
                         ('head_component', pack['head_component'][owners]), ('scale_id', pack['scale_id'][owners])]:
        a = numpy_to_vtk(np.ascontiguousarray(values), deep=True); a.SetName(name); mesh.GetCellData().AddArray(a)
    for model in models:
        p = pack['p_'+model['id']][owners]
        for name, values in [('p_'+model['id'], p), ('class_'+model['id'], (p >= .5).astype(np.int8))]:
            a = numpy_to_vtk(np.ascontiguousarray(values), deep=True); a.SetName(name); mesh.GetCellData().AddArray(a)
    return mesh


def preview(mesh, gt, bounds, model, flow, path, f1, bundles):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    def rgb(color): return tuple(int(color[i:i+2],16)/255 for i in (1,3,5))
    renderer = vtk.vtkRenderer(); renderer.SetBackground(.973,.977,.97)
    window = vtk.vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1800,1100)
    window.SetMultiSamples(0); window.SetAlphaBitPlanes(1); window.AddRenderer(renderer)
    renderer.UseDepthPeelingOn(); renderer.SetMaximumNumberOfPeels(100); renderer.SetOcclusionRatio(.01)
    target = vtk.vtkPolyData(); target.DeepCopy(mesh)
    labels = vtk_to_numpy(target.GetCellData().GetArray('class_'+model['id']))
    colors = np.array([[int(COLORS['non_hairpin'][i:i+2],16) for i in (1,3,5)],
                       [int(COLORS['hairpin'][i:i+2],16) for i in (1,3,5)]], dtype=np.uint8)
    array = numpy_to_vtk(colors[labels], deep=True); array.SetName('display_color'); target.GetCellData().SetScalars(array)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(target); mapper.SetScalarModeToUseCellData(); mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetLineWidth(1.7); renderer.AddActor(actor)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(gt); mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(*rgb(COLORS['ground_truth']))
    actor.GetProperty().SetOpacity(.18); renderer.AddActor(actor)
    outline = vtk.vtkOutlineSource(); outline.SetBounds(bounds)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(outline.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(.74,.77,.74); renderer.AddActor(actor)
    low=np.array(bounds[::2]); high=np.array(bounds[1::2]); center=(low+high)/2
    camera=renderer.GetActiveCamera(); camera.SetPosition(*(center+np.linalg.norm(high-low)*np.array([.45,-1.25,.85])))
    camera.SetFocalPoint(*center); camera.SetViewUp(0,0,1); camera.SetViewAngle(30); renderer.ResetCamera(*bounds); camera.Zoom(1.18)
    axes=vtk.vtkCubeAxesActor(); axes.SetBounds(bounds); axes.SetCamera(camera)
    axes.SetXTitle('x: streamwise'); axes.SetYTitle('y: spanwise'); axes.SetZTitle('z: vertical')
    axes.SetFlyModeToOuterEdges(); axes.XAxisMinorTickVisibilityOff(); axes.YAxisMinorTickVisibilityOff(); axes.ZAxisMinorTickVisibilityOff()
    axes.ZAxisLabelVisibilityOff()
    for i in range(3): axes.GetTitleTextProperty(i).SetColor(.25,.29,.3); axes.GetLabelTextProperty(i).SetColor(.4,.44,.44)
    renderer.AddActor(axes)
    def text(value,x,y,size,color=(.2,.24,.25)):
        t=vtk.vtkTextActor(); t.SetInput(value); t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport();t.SetPosition(x,y)
        t.GetTextProperty().SetFontSize(size); t.GetTextProperty().SetColor(*color);t.GetTextProperty().SetFontFamilyToArial();renderer.AddViewProp(t)
    text(f'{flow.upper()} | Task4-c | {model["label"].replace("·", "-")}',.025,.947,28)
    text('Predicted Hairpin',.025,.9,22,rgb(COLORS['hairpin']))
    text('Predicted Non-hairpin',.28,.9,22,rgb(COLORS['non_hairpin']))
    text('Ground truth surface (18%)',.6,.9,22,rgb(COLORS['ground_truth']))
    text(f'Test examples: {bundles} displayed | Full flow-test F1: {f1:.4f} | Seed {model["seed"]}',.025,.022,20)
    renderer.ResetCameraClippingRange();window.Render()
    capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.SetInputBufferTypeToRGB();capture.ReadFrontBufferOff();capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(path));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
    record=dict(file=path.name, model=model['id'], bounds=list(bounds), displayed_bundles=bundles,
        depth_peeling=bool(renderer.GetLastRenderingUsedDepthPeeling()), camera_position=list(camera.GetPosition()))
    window.Finalize();return record


def build_viewer(package, input_root, output, make_previews):
    from vtk.util.numpy_support import vtk_to_numpy
    from plotly.offline import get_plotlyjs
    package=Path(package); output=Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    models=manifest['models']; payload=dict(models=models, pending=manifest['pending'], colors=COLORS,
        threshold=.5, display_bundles=manifest['config']['display_bundles'], flows={}, evidence_note=manifest.get('evidence_note'))
    records=[]; surfaces={}; vtk_files=[]
    for flow in manifest['flows']:
        name=flow['name']; native=Path(input_root)/flow['flow']; gt_path=Path(input_root)/flow['gt']
        assert sha(native)==flow['flow_sha256']; assert sha(gt_path)==flow['gt_sha256']
        domain=read_vtk(native).GetBounds(); gt=gt_surface(gt_path); surfaces[name]=gt
        points=vtk_to_numpy(gt.GetPoints().GetData()); faces=vtk_to_numpy(gt.GetPolys().GetConnectivityArray()).reshape(-1,3)
        ids=vtk_to_numpy(gt.GetCellData().GetArray('VortexIds'))
        gt_file=output/f'{name}_ground_truth.vtp';save_vtp(gt,gt_file);vtk_files.append(gt_file.name)
        entry=dict(domain=list(domain), gt=dict(points=array_json(points,'<f4'),faces=array_json(faces,'<i4'),
            ids=array_json(ids,'<i4'),instances=np.unique(ids).tolist(),bounds=list(gt.GetBounds())),splits={})
        for split in manifest['config']['splits']:
            key=name+'/'+split; source=manifest['splits'][key];assert sha(package/source['file'])==source['sha256']
            with np.load(package/source['file']) as z: pack={k:z[k] for k in z.files}
            entry['splits'][split]=dict(population=source['population'],selected=source['selected'],
                geometry=array_json(pack['geometry'],'<f4'),
                **{k:pack[k].tolist() for k in ('counts','row_ids','labels','center','head_component','instance','scale_id','radius','neighbor_distance')},
                probabilities={m['id']:pack['p_'+m['id']].tolist() for m in models})
            for optional in ('owner_instance','mandatory_head'):
                if optional in pack: entry['splits'][split][optional]=pack[optional].tolist()
            mesh=bundle_mesh(pack,models);vtk_file=output/f'{name}_{split}_bundles.vtp';save_vtp(mesh,vtk_file);vtk_files.append(vtk_file.name)
            if split=='test' and make_previews:
                take=manifest['config']['display_bundles']; subset={k:v[:take] for k,v in pack.items()}
                selected_mesh=bundle_mesh(subset,models)
                low=np.minimum(np.array(gt.GetBounds()[::2]),np.array(selected_mesh.GetBounds()[::2]))
                high=np.maximum(np.array(gt.GetBounds()[1::2]),np.array(selected_mesh.GetBounds()[1::2]));low[2]=domain[4]
                bounds=np.column_stack((low,high)).ravel().tolist()
                records.append(preview(selected_mesh,gt,bounds,models[0],name,output/f'{name}_preview.png',
                    models[0]['metrics']['test']['per_flow'][name]['f1'],len(subset['counts'])))
        payload['flows'][name]=entry
    template=Path(__file__).parent/'templates'/'task4c_bundles.html'
    html=template.read_text(encoding='utf-8').replace('/*__PLOTLY__*/',get_plotlyjs()).replace('/*__PAYLOAD__*/',
        json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    path=output/'index.html';path.write_text(html,encoding='utf-8')
    write(output/'viewer_manifest.json',dict(version=manifest['version'],source_manifest_sha256=sha(package/'manifest.json'),
        html_sha256=sha(path),template_sha256=sha(template),build_source_sha256=sha(__file__),models=[m['id'] for m in models], pending=[m['id'] for m in manifest['pending']],
        ground_truth_exact_boundary=True,gt_smoothing=False,gt_decimation=False,instance_zero_included=True,
        source_gt_sha256={f['name']:f['gt_sha256'] for f in manifest['flows']},previews=records,vtk_files=vtk_files))
    print(json.dumps(dict(status='PASS',viewer=str(path.resolve()),models=[m['id'] for m in models],html_mb=path.stat().st_size/1e6)),flush=True)


def runtime(config, output, state, code=None):
    from experiments.Task4C_HairpinBinary_2_1 import append_locked
    row=dict(version='Other_Task4C_BundleVisualization_1.1',state=state,exit_code=code,time=datetime.now(timezone.utc).isoformat(),**identity(config))
    append_locked(str(Path(output).parent/'runtime_events.jsonl'),json.dumps(row)+'\n')
    append_locked('docs/ibex_run_registry.md','\n- Task4-c bundle visualization runtime '+json.dumps(row)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('export','build'))
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--output',required=True)
    parser.add_argument('--package');parser.add_argument('--input-root');parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    if args.phase=='export':
        code=1
        if os.environ.get('SLURM_JOB_ID'):runtime(args.config,args.output,'STARTED')
        try: export_examples(args.config,args.output);code=0
        finally:
            if os.environ.get('SLURM_JOB_ID'):runtime(args.config,args.output,'ENDED',code)
    else:
        if not args.package or not args.input_root:parser.error('build requires --package and --input-root')
        build_viewer(args.package,args.input_root,args.output,args.preview)


if __name__=='__main__':main()
