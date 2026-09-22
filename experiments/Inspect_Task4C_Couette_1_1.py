"""Read-only Couette input audit before freezing physical parameters."""
import hashlib
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


ROOT = Path('C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D/couette_flow')
OUTPUT = Path('outputs/Verify_Task4C_CouetteInputs_1.1')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    r = vtk.vtkGenericDataObjectReader()
    r.SetFileName(str(path)); r.ReadAllScalarsOn(); r.ReadAllVectorsOn(); r.Update()
    assert r.GetErrorCode() == 0
    return r.GetOutput()


def main():
    flow = read(ROOT/'couette.vtk'); gt = read(ROOT/'couette_GTs.vtk')
    dims = [0, 0, 0]; flow.GetDimensions(dims)
    xyz = vtk_to_numpy(flow.GetPoints().GetData()).reshape(*dims[::-1], 3)
    axes = [xyz[0, 0, :, 0], xyz[0, :, 0, 1], xyz[:, 0, 0, 2]]
    for d, reference in enumerate((axes[0][None, None, :], axes[1][None, :, None], axes[2][:, None, None])):
        assert np.all(np.diff(axes[d]) > 0)
        np.testing.assert_allclose(xyz[..., d], np.broadcast_to(reference, xyz.shape[:-1]), atol=1e-7, rtol=0)
    fields = {}
    for name in ('velocity', 'lambda2', 'oyf'):
        a = vtk_to_numpy(flow.GetPointData().GetArray(name)); assert np.isfinite(a).all()
        fields[name] = dict(shape=list(a.shape), dtype=str(a.dtype), minimum=a.min(axis=0).tolist(), maximum=a.max(axis=0).tolist())
        if a.ndim == 1:
            fields[name].update(negative=int((a < 0).sum()), zero=int((a == 0).sum()), positive=int((a > 0).sum()), quantiles=np.quantile(a, [0, .01, .05, .5, .95, .99, 1]).tolist())
    ids = vtk_to_numpy(gt.GetCellData().GetArray('VortexIds'))
    region_ids = vtk_to_numpy(gt.GetCellData().GetArray('RegionIds'))
    point_ids = vtk_to_numpy(gt.GetPointData().GetArray('PointIds')).astype(np.int64)
    points = vtk_to_numpy(gt.GetPoints().GetData())
    assert ((point_ids >= 0) & (point_ids < flow.GetNumberOfPoints())).all()
    np.testing.assert_array_equal(points, xyz.reshape(-1, 3)[point_ids])
    for name in ('velocity', 'lambda2', 'oyf'):
        np.testing.assert_array_equal(vtk_to_numpy(gt.GetPointData().GetArray(name)), vtk_to_numpy(flow.GetPointData().GetArray(name))[point_ids])
    instances = []
    for iid in np.unique(ids):
        selected = np.flatnonzero(ids == iid)
        threshold = vtk.vtkThreshold(); threshold.SetInputData(gt)
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, 'VortexIds')
        threshold.SetLowerThreshold(float(iid)); threshold.SetUpperThreshold(float(iid))
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN); threshold.Update()
        instances.append(dict(id=int(iid), cells=len(selected), bounds=list(threshold.GetOutput().GetBounds()), regions=np.unique(region_ids[selected]).astype(int).tolist()))
    average_h = [(float(a[-1])-float(a[0]))/(len(a)-1) for a in axes]
    parent = list(range(len(instances)))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, first in enumerate(instances):
        for j, second in enumerate(instances[:i]):
            a = np.asarray(first['bounds']).reshape(3, 2)
            b = np.asarray(second['bounds']).reshape(3, 2)
            if np.all(a[:, 0] <= b[:, 1]) and np.all(b[:, 0] <= a[:, 1]):
                parent[find(i)] = find(j)
    groups = {}
    for i, entry in enumerate(instances):
        groups.setdefault(find(i), []).append(entry['id'])
    report = dict(version='Verify_Task4C_CouetteInputs_1.1', complete=True,
        inputs={name: dict(path=str(ROOT/name), bytes=(ROOT/name).stat().st_size, sha256=sha(ROOT/name)) for name in ('couette.vtk', 'couette_GTs.vtk')},
        dimensions_xyz=dims, bounds_xyz=[[float(a[0]), float(a[-1])] for a in axes],
        native_spacing=[dict(minimum=float(np.diff(a).min()), maximum=float(np.diff(a).max()), span_over_n_minus_one=average_h[d]) for d, a in enumerate(axes)],
        proposed_ball_h=min(average_h), ball_h_definition='minimum across axes of span/(node_count-1), following the user-defined Channel/TBL convention',
        fields=fields, instances=instances, overlapping_box_groups=list(groups.values()), gt_point_geometry_and_fields_exact_match=True,
        gt_cells=gt.GetNumberOfCells(), gt_points=gt.GetNumberOfPoints(), region_count=len(np.unique(region_ids)),
        vorticity_source='derive curl from native velocity using the existing finite-difference routine; source flow has no vorticity array',
        pending_parameters=['lambda2_threshold', 'three_half_lengths_and_ds'], dataset_built=False, training_submitted=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'input_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
