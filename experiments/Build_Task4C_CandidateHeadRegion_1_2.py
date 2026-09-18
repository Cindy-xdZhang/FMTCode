"""Add the step-1 candidate-head region to the GT-head classification viewer (1.2).

Candidate head region = {x : lambda2(x) < threshold and oyf(x) > 0} evaluated with the
same trilinear interpolation the sampling code uses for seed points.  Its boundary is the
zero level set of f(x) = max(lambda2(x) - threshold, -oyf(x)), extracted by marching cubes
on the native grid and mapped back to physical coordinates (marching cubes interpolates
linearly along grid edges, so the vertices are exactly the trilinear boundary crossings on
those edges).  Nothing here touches labels, predictions, or sampled data; the viewer only
gains an optional surface and per-center lambda2 / oyf values.
"""
from pathlib import Path
import argparse
import json
import shutil
import time

import numpy as np

DEFAULT_INPUT_ROOT = 'C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D'
DEFAULT_PACKAGE = 'outputs/mainExp_Task4C_GTHeadCoverage_1.1/viewer_package'
DEFAULT_VIEWER = 'outputs/Other_Task4C_GTHeadCoverage_1.1/viewer'
CONTROLS = Path('experiments/templates/task4c_candidate_head_1_2.js')
COLOR = '#7fb069'


def load_fields(path):
    """Native lambda2 / oyf point arrays as (nz, ny, nx) plus separable axes."""
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset
    from vtk.util.numpy_support import vtk_to_numpy
    grid = read_dataset(path)
    dims = [0, 0, 0]; grid.GetDimensions(dims); nx, ny, nz = dims
    xyz = vtk_to_numpy(grid.GetPoints().GetData()).reshape(nz, ny, nx, 3)
    axes = [xyz[0, 0, :, 0].copy(), xyz[0, :, 0, 1].copy(), xyz[:, 0, 0, 2].copy()]
    for d, ref in enumerate((axes[0][None, None, :], axes[1][None, :, None], axes[2][:, None, None])):
        assert np.all(np.diff(axes[d]) > 0) and np.allclose(xyz[..., d], ref, atol=1e-7, rtol=0)
    pd = grid.GetPointData()
    lam = vtk_to_numpy(pd.GetArray('lambda2')).reshape(nz, ny, nx).astype(np.float64)
    oyf = vtk_to_numpy(pd.GetArray('oyf')).reshape(nz, ny, nx).astype(np.float64)
    assert np.isfinite(lam).all() and np.isfinite(oyf).all()
    return axes, lam, oyf


def index_to_physical(verts_kji, axes):
    nx, ny, nz = (len(a) for a in axes)
    return np.stack([np.interp(verts_kji[:, 2], np.arange(nx), axes[0]),
                     np.interp(verts_kji[:, 1], np.arange(ny), axes[1]),
                     np.interp(verts_kji[:, 0], np.arange(nz), axes[2])], 1)


def vertex_component_size(verts_kji, mask, sizes, labels):
    """Size of the candidate component each surface vertex belongs to.

    Every marching-cubes vertex lies on a grid edge joining one inside and one
    outside grid point; the inside endpoint fixes the connected component.
    """
    lo = np.floor(verts_kji).astype(np.int64); hi = np.ceil(verts_kji).astype(np.int64)
    result = np.zeros(len(verts_kji), dtype=np.int64); found = np.zeros(len(verts_kji), dtype=bool)
    for corner in range(8):
        pick = np.array([(corner >> b) & 1 for b in range(3)], dtype=bool)
        point = np.where(pick[None, :], hi, lo)
        inside = mask[point[:, 0], point[:, 1], point[:, 2]] & ~found
        result[inside] = sizes[labels[point[inside, 0], point[inside, 1], point[inside, 2]]]
        found |= inside
    assert found.all(), 'A surface vertex has no inside grid endpoint'
    return result


def marching(f, level=0.0):
    from skimage import measure
    if not (f.min() < level < f.max()):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    verts, faces, _, _ = measure.marching_cubes(f, level=level, allow_degenerate=False)
    return verts, faces.astype(np.int64)


def decimate(verts, faces, reduction):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    mesh = vtk.vtkPolyData(); points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(np.ascontiguousarray(verts), deep=True)); mesh.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(faces)+1, dtype=np.int64)*3, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces.reshape(-1), dtype=np.int64), deep=True))
    mesh.SetPolys(cells)
    filt = vtk.vtkQuadricDecimation(); filt.SetInputData(mesh); filt.SetTargetReduction(reduction); filt.VolumePreservationOn(); filt.Update()
    out = filt.GetOutput(); out_faces = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    assert out.GetNumberOfCells() == len(out_faces)
    return vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64), out_faces.astype(np.int64)


def write_chunk(path, key, verts, faces, sizes, extra):
    from experiments.Visualize_Task4C_Bundles_3D import array_json
    payload = dict(verts=array_json(verts.reshape(-1), '<f4'), faces=array_json(faces.reshape(-1), '<i4'), **extra)
    if sizes is not None:
        payload['sizes'] = array_json(sizes, '<i4')
    path.write_text('window.task4cCandidateChunk(' + json.dumps(key) + ',' + json.dumps(payload, separators=(',', ':')) + ');\n', encoding='utf-8')


def export_flow(name, threshold, axes, lam, oyf, folder, block, reduction):
    from skimage import measure
    from FMT_Utils.Task4C_GTHeadCoverage_1_1 import sha
    nz, ny, nx = lam.shape
    f = np.maximum(lam - threshold, -oyf)                     # f < 0  <=>  step-1 candidate
    mask = f < 0
    assert np.array_equal(mask, (lam < threshold) & (oyf > 0))
    labels, count = measure.label(mask, connectivity=3, return_num=True)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    started = time.time(); chunks = []; total_faces = 0
    for k0 in range(0, nz-1, block):
        for j0 in range(0, ny-1, block):
            for i0 in range(0, nx-1, block):
                k1, j1, i1 = min(k0+block, nz-1), min(j0+block, ny-1), min(i0+block, nx-1)
                verts, faces = marching(f[k0:k1+1, j0:j1+1, i0:i1+1])
                if not len(faces):
                    continue
                verts = verts + np.array([k0, j0, i0], dtype=np.float64)
                comp = vertex_component_size(verts, mask, sizes, labels)
                face_sizes = comp[faces].min(1)
                physical = index_to_physical(verts, axes)
                key = f'{name}_{len(chunks):04d}'; path = folder / (key + '.js')
                bounds = [float(v) for pair in zip(physical.min(0), physical.max(0)) for v in pair]
                write_chunk(path, key, physical, faces, face_sizes, dict(bounds=bounds))
                chunks.append(dict(key=key, path='candidate/' + path.name, bounds=bounds, faces=int(len(faces)), vertices=int(len(verts)), sha256=sha(path)))
                total_faces += len(faces)
    full_verts, full_faces = marching(f)
    # Block-wise and whole-grid extraction share every edge crossing; only the
    # removal of zero-area triangles can differ by a handful of faces.
    assert abs(len(full_faces) - total_faces) <= max(16, total_faces // 10000), (len(full_faces), total_faces)
    physical = index_to_physical(full_verts, axes)
    dec_verts, dec_faces = decimate(physical, full_faces, reduction)
    key = name + '_overview'; path = folder / (key + '.js')
    write_chunk(path, key, dec_verts, dec_faces, None, dict(bounds=[float(v) for pair in zip(dec_verts.min(0), dec_verts.max(0)) for v in pair], decimated=True))
    overview = dict(key=key, path='candidate/' + path.name, faces=int(len(dec_faces)), vertices=int(len(dec_verts)),
                    target_reduction=reduction, actual_reduction=1 - len(dec_faces) / len(full_faces), sha256=sha(path))
    component_sizes = sizes[1:]
    stats = dict(lambda2_threshold=threshold, dimensions_xyz=[nx, ny, nz], grid_points=int(lam.size),
                 lambda2_below_threshold_fraction=float((lam < threshold).mean()), oyf_positive_fraction=float((oyf > 0).mean()),
                 candidate_point_fraction=float(mask.mean()), candidate_points=int(mask.sum()),
                 connected_components_26=int(count),
                 component_size_quantiles_50_90_99_100=np.quantile(component_sizes, [.5, .9, .99, 1]).tolist(),
                 points_in_components_at_least_10=int(component_sizes[component_sizes >= 10].sum()),
                 surface_vertices=int(len(full_verts)), surface_faces=int(len(full_faces)), chunk_faces=int(total_faces),
                 surface_definition='zero level set of max(lambda2-threshold, -oyf) on the native grid; marching cubes, linear edge interpolation',
                 seconds=time.time() - started)
    return dict(threshold=threshold, chunks=chunks, overview=overview, stats=stats, color=COLOR), (labels, sizes)


def center_values(payload_flow, axes, lam, oyf, threshold):
    from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
    summary = {}
    for role, s in payload_flow['splits'].items():
        centers = np.asarray(s['center'], dtype=np.float64)
        l, inside_l, _ = interpolate_scalar(centers, axes, lam); o, inside_o, _ = interpolate_scalar(centers, axes, oyf)
        assert inside_l.all() and inside_o.all(), 'A bundle center lies outside the native domain'
        step1 = (l < threshold) & (o > 0)
        s['center_lambda2'] = [float(f'{v:.6g}') for v in l]
        s['center_oyf'] = [float(f'{v:.6g}') for v in o]
        s['center_step1'] = step1.astype(int).tolist()
        head = (np.asarray(s['labels']) == 1) & np.asarray(s['center_is_head'], dtype=bool)
        added = np.asarray(s.get('mandatory_head', [False]*len(step1)), dtype=bool)
        summary[role] = dict(rows=int(len(step1)), step1_rows=int(step1.sum()),
                             gt_head_rows=int(head.sum()), gt_head_step1_rows=int((head & step1).sum()),
                             added_head_rows=int((added & head).sum()), added_head_step1_rows=int((added & head & step1).sum()),
                             non_hairpin_rows=int((np.asarray(s['labels']) == 0).sum()), non_hairpin_step1_rows=int(((np.asarray(s['labels']) == 0) & step1).sum()))
    return summary


def decorate(viewer, package, input_root, block, reduction, refresh_workbench):
    from FMT_Utils.Task4C_GTHeadCoverage_1_1 import sha
    viewer = Path(viewer); path = viewer / 'index.html'
    backup = viewer / 'index_1.1.html'
    if not backup.exists():
        shutil.copyfile(path, backup)
    html = backup.read_text(encoding='utf8')            # always decorate the 1.1 page, never a 1.2 page twice
    assert 'installCandidateHead' not in html
    start = html.index('const DATA=') + len('const DATA=')
    payload, length = json.JSONDecoder().raw_decode(html[start:])
    manifest = json.loads((Path(package) / 'manifest.json').read_text())
    folder = viewer / 'candidate'; folder.mkdir(exist_ok=True)
    for old in folder.glob('*.js'):
        old.unlink()
    index = dict(version='candidate head 1.2', color=COLOR, flows={}); centers = {}; sources = {}
    for flow in manifest['flows']:
        name = flow['name']; native = Path(input_root) / flow['flow']
        assert sha(native) == flow['flow_sha256'], f'{native} differs from the frozen flow snapshot'
        sources[name] = dict(file=flow['flow'], sha256=flow['flow_sha256'])
        axes, lam, oyf = load_fields(native)
        entry, _ = export_flow(name, flow['lambda2_threshold'], axes, lam, oyf, folder, block, reduction)
        centers[name] = center_values(payload['flows'][name], axes, lam, oyf, flow['lambda2_threshold'])
        entry['center_summary'] = centers[name]
        index['flows'][name] = entry
        print(json.dumps(dict(flow=name, faces=entry['stats']['surface_faces'], chunks=len(entry['chunks']), overview_faces=entry['overview']['faces'], centers=centers[name])), flush=True)
    (folder / 'index.js').write_text('window.TASK4C_CANDIDATE=' + json.dumps(index, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf-8')
    payload['viewer_version'] = 'GT头部覆盖 1.1 · candidate head 1.2'
    html = html[:start] + json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/') + html[start+length:]
    needle = 'installGTCoverage();refreshRegions();render(true).then('
    assert html.count(needle) == 1
    html = html.replace(needle, 'installGTCoverage();installCandidateHead();refreshRegions();render(true).then(')
    detail = "'<br>本次新增的GT头部线束':'')"
    assert html.count(detail) == 1
    html = html.replace(detail, detail + '+candidateDetail(s,id)')
    controls_sha = sha(CONTROLS); index_sha = sha(folder / 'index.js')
    html = html.replace('</head>', '<script src="candidate/index.js?v=' + index_sha[:16] + '"></script><script src="candidate_head.js?v=' + controls_sha[:16] + '"></script></head>', 1)
    path.write_text(html, encoding='utf8')
    shutil.copyfile(CONTROLS, viewer / 'candidate_head.js')
    record = dict(version='Other_Task4C_CandidateHeadRegion_1.2', html_sha256=sha(path), base_1_1_html_sha256=sha(backup),
                  controls_sha256=sha(viewer / 'candidate_head.js'), index_js_sha256=index_sha, sources=sources,
                  block_cells=block, overview_target_reduction=reduction, flows=index['flows'],
                  criterion='center candidate: lambda2 < threshold and oyf > 0 (trilinear at the point); Channel -13.395, TBL -0.0272',
                  science_unchanged='labels, predictions, sampled bundles and F1 are untouched; only display data added')
    (viewer / 'candidate_head_manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    coverage = json.loads((viewer / 'head_coverage_manifest.json').read_text())
    coverage['html_sha256'] = record['html_sha256']; coverage['candidate_head_manifest_sha256'] = sha(viewer / 'candidate_head_manifest.json')
    (viewer / 'head_coverage_manifest.json').write_text(json.dumps(coverage, indent=2) + '\n', encoding='utf8')
    build = json.loads((viewer / 'viewer_manifest.json').read_text())
    build['html_sha256'] = record['html_sha256']; build['candidate_head_manifest_sha256'] = coverage['candidate_head_manifest_sha256']
    build['head_coverage_manifest_sha256'] = sha(viewer / 'head_coverage_manifest.json')
    (viewer / 'viewer_manifest.json').write_text(json.dumps(build, indent=2) + '\n', encoding='utf8')
    if refresh_workbench:
        from experiments.Build_FMT_AnalysisWorkbench_1_5 import build as build_workbench
        build_workbench('outputs/Other_FMT_AnalysisWorkbench_1.4', viewer, 'outputs/Other_FMT_AnalysisWorkbench_1.5', update_entry=False)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', default=DEFAULT_INPUT_ROOT); p.add_argument('--package', default=DEFAULT_PACKAGE)
    p.add_argument('--viewer', default=DEFAULT_VIEWER); p.add_argument('--block', type=int, default=64)
    p.add_argument('--overview-reduction', type=float, default=0.85)
    p.add_argument('--no-workbench', action='store_true', help='do not refresh the workbench 1.5 build manifest')
    a = p.parse_args()
    record = decorate(a.viewer, a.package, a.input_root, a.block, a.overview_reduction, not a.no_workbench)
    print(json.dumps(dict(status='PASS', html_sha256=record['html_sha256'], flows={k: v['stats']['surface_faces'] for k, v in record['flows'].items()})), flush=True)
