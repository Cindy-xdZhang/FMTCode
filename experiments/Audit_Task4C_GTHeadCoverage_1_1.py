"""Audit every annotated instance against the exact rows in a result viewer."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def audit_catalog(input_root, output):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy import ndimage
    from experiments.Task4C_PhysicalLength_4_14 import load_spec, build_catalog
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow, cell_centers
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset, sample_gt
    from FMT_Utils.Task4C_InstanceCoverage_9_15 import vector_at, head_mask
    from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar

    spec = load_spec('config/mainExp_Task4C_PhysicalLength_4.14.json')
    result = {'version': 'Verify_Task4C_GTHeadCoverage_1.1', 'flows': {}}
    for flow in spec['flows']:
        name = flow['name']; root = Path(input_root)
        for key in ('flow', 'gt'):
            assert hashlib.sha256((root / flow[key]).read_bytes()).hexdigest() == flow[key + '_sha256']
        axes, _, heads, oyf, grid, info = load_flow(root / flow['flow'], flow['lambda2_threshold'])
        gt = read_dataset(root / flow['gt'])
        components, catalog, details = build_catalog(axes, heads, gt, spec)
        flat = np.flatnonzero(heads)
        native_ids, _ = sample_gt(gt, cell_centers(flat, heads.shape, axes))
        component_ids = components.ravel()[flat]
        labels = vtk_to_numpy(gt.GetCellData().GetArray('VortexIds')).astype(np.int64)
        centers = vtk.vtkCellCenters(); centers.SetInputData(gt); centers.Update()
        points = vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).copy()
        native = read_dataset(root / flow['flow'])
        shape = tuple(len(a) for a in axes[::-1])
        velocity = vtk_to_numpy(native.GetPointData().GetArray('velocity')).reshape(*shape, 3)
        omega = vtk_to_numpy(grid.GetPointData().GetArray('vorticity')).reshape(*shape, 3)
        lam = vtk_to_numpy(native.GetPointData().GetArray('lambda2')).reshape(shape)
        angle, cosine = head_mask(vector_at(points, axes, velocity), vector_at(points, axes, omega))
        lv, inside, _ = interpolate_scalar(points, axes, lam)
        ov, _, _ = interpolate_scalar(points, axes, oyf)
        possible = angle & inside & (lv < flow['lambda2_threshold']) & (ov > 0)
        rejected = {r['head']: r['reason'] for r in details['excluded_heads']}
        eligible = {r['head_component']: r for r in catalog}
        rows = []
        for iid in np.unique(labels):
            overlap = np.unique(component_ids[native_ids == iid])
            eligible_ids = [int(k) for k in overlap if k in eligible and eligible[k]['label'] == 1 and eligible[k]['instance'] == iid]
            rows.append({'instance': int(iid), 'gt_head_cells': int(np.sum((labels == iid) & angle)),
                'pointwise_candidate_head_cells': int(np.sum((labels == iid) & possible)),
                'strict_candidate_cells_inside_gt': int(np.sum(native_ids == iid)),
                'eligible_positive_components': eligible_ids,
                'component_rejections': [{'head': int(k), 'reason': rejected.get(k, 'owned_by_another_gt_instance'),
                    'owner': eligible[k]['instance'] if k in eligible else None} for k in overlap if int(k) not in eligible_ids]})
        result['flows'][name] = {'instances': rows, 'source': info}
        out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
        print(name, json.dumps({'instances': len(rows), 'without_original_positive_component': [r['instance'] for r in rows if not r['eligible_positive_components']],
            'without_pointwise_head_pool': [r['instance'] for r in rows if r['pointwise_candidate_head_cells'] == 0]}), flush=True)
        del native, grid, components, heads, oyf, velocity, omega, lam, gt


def audit(viewer, input_root, output):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    html = Path(viewer).read_text(encoding='utf8')
    marker = 'const DATA='
    data, _ = json.JSONDecoder().raw_decode(html[html.index(marker) + len(marker):])
    config = json.loads(Path('config/mainExp_Task4C_Multiscale_4.1.json').read_text())
    report = {'version': 'Verify_Task4C_GTHeadCoverage_1.1',
        'viewer': str(viewer), 'viewer_sha256': hashlib.sha256(html.encode('utf8')).hexdigest(), 'flows': {}}
    for flow in config['flows']:
        name = flow['name']
        path = Path(input_root) / flow['gt']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == flow['gt_sha256']
        reader = vtk.vtkDataSetReader(); reader.SetFileName(str(path))
        reader.ReadAllScalarsOn(); reader.ReadAllFieldsOn(); reader.Update()
        gt = reader.GetOutput()
        values = vtk_to_numpy(gt.GetCellData().GetArray('VortexIds')).astype(np.int64)
        identifiers = np.unique(values)
        assert np.array_equal(identifiers, data['flows'][name]['gt']['instances'])
        locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(gt); locator.BuildLocator()
        rows = {int(i): {'instance': int(i)} for i in identifiers}
        summary = {}
        for split, source in data['flows'][name]['splits'].items():
            centers = np.asarray(source['center'])
            cell_ids = np.fromiter((locator.FindCell(p) for p in centers), dtype=np.int64, count=len(centers))
            center_instance = np.full(len(centers), -1, dtype=np.int64)
            inside = cell_ids >= 0
            center_instance[inside] = values[cell_ids[inside]]
            labels = np.asarray(source['labels'])
            head = np.asarray(source['center_is_head'], dtype=bool)
            assigned = np.asarray(source['instance'])
            assert not np.any(head & ~inside)
            assert source['selected'] == source['population'] == len(centers)
            for iid in identifiers:
                on_instance = center_instance == iid
                visible_head = on_instance & head & (labels == 1)
                rows[int(iid)][split] = {
                    'source_positive_bundles': int(np.sum((assigned == iid) & (labels == 1))),
                    'centers_inside_gt': int(on_instance.sum()),
                    'head_centers_any_label': int(np.sum(on_instance & head)),
                    'visible_head_bundles': int(visible_head.sum()),
                    'head_rows_hidden_by_negative_label': int(np.sum(on_instance & head & (labels == 0))),
                    'visible_head_row_ids': np.asarray(source['row_ids'])[visible_head].tolist(),
                    'predicted_hairpin_among_heads': {m: int(np.sum(np.asarray(p)[visible_head] >= .5))
                        for m, p in source['probabilities'].items()},
                }
            summary[split] = {'bundles': len(centers), 'head_bundles': int(np.sum(head & (labels == 1))),
                'gt_instances': len(identifiers),
                'covered_head_instances': sum(rows[int(i)][split]['visible_head_bundles'] > 0 for i in identifiers),
                'missing_head_instances': [int(i) for i in identifiers if rows[int(i)][split]['visible_head_bundles'] == 0],
                'missing_source_positive_instances': [int(i) for i in identifiers if rows[int(i)][split]['source_positive_bundles'] == 0],
                'head_negative_rows_hidden': int(np.sum(head & (labels == 0)))}
        report['flows'][name] = {'summary': summary, 'instances': list(rows.values())}
        print(name, json.dumps(summary), flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', default='outputs/Other_Task4C_BundleVisualization_1.3/viewer/index.html')
    parser.add_argument('--input-root', default='C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D')
    parser.add_argument('--output', default='outputs/Verify_Task4C_GTHeadCoverage_1.1/coverage.json')
    parser.add_argument('--catalog', action='store_true')
    args = parser.parse_args()
    if args.catalog:
        audit_catalog(args.input_root, args.output)
    else:
        audit(args.viewer, args.input_root, args.output)
