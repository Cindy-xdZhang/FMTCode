"""Instance-balanced, head-covered vortex bundles for Task4-c v9.15."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset, gt_instance_bounds, sample_gt
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow, cell_centers, bundle_voxels
from experiments.Task4C_PhysicalLength_4_14 import trace_batch


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def vector_at(points, axes, values):
    return np.column_stack([interpolate_scalar(points, axes, values[..., i])[0] for i in range(3)])


def head_mask(velocity, omega):
    """Closer to perpendicular than parallel/antiparallel; exactly 45 is parallel."""
    speed = np.linalg.norm(velocity, axis=-1)
    strength = np.linalg.norm(omega, axis=-1)
    valid = (speed > 0) & (strength > 0) & np.isfinite(velocity).all(-1) & np.isfinite(omega).all(-1)
    cosine = np.ones_like(speed)
    cosine[valid] = np.clip(np.abs((velocity[valid]*omega[valid]).sum(-1))/(speed[valid]*strength[valid]), 0, 1)
    return valid & (cosine*cosine < .5-4*np.finfo(np.float64).eps), cosine


def instance_split(identifiers, low, high, seed):
    """Keep overlapping instance boxes together; exact nearest 10% instance count."""
    parent = list(range(len(identifiers)))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(identifiers)):
        for j in range(i):
            if np.all(low[i] <= high[j]) and np.all(low[j] <= high[i]):
                parent[find(i)] = find(j)
    groups = {}
    for i, identifier in enumerate(identifiers):
        groups.setdefault(find(i), []).append(int(identifier))
    groups = list(groups.values())
    order = np.random.default_rng(seed).permutation(len(groups))
    target = max(1, round(len(identifiers)*.1))
    possible = {0: []}
    for g in order:
        for size, selected in list(possible.items()):
            new = size+len(groups[g])
            if new <= target and new not in possible:
                possible[new] = selected+[int(g)]
    if target not in possible:
        raise ValueError('Cannot achieve the requested 90/10 instance count with overlapping boxes kept together')
    test = sorted(i for g in possible[target] for i in groups[g])
    roles = {int(i): ('test' if i in test else 'train') for i in identifiers}
    return roles, groups


def load_scene(spec, flow_index, input_root=None):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    flow = spec['flows'][flow_index]
    root = Path(input_root or spec['input_root'])
    for key in ('flow', 'gt'):
        digest = hashlib.sha256()
        with (root/flow[key]).open('rb') as handle:
            for block in iter(lambda: handle.read(4*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != flow[key+'_sha256']:
            raise ValueError(f'Frozen input changed: {root/flow[key]}')
    axes, _, _, oyf, grid, info = load_flow(root/flow['flow'], flow['lambda2_threshold'])
    source = read_dataset(root/flow['flow'])
    shape = tuple(len(a) for a in axes[::-1])
    lam = vtk_to_numpy(source.GetPointData().GetArray('lambda2')).reshape(shape).copy()
    velocity = vtk_to_numpy(source.GetPointData().GetArray('velocity')).reshape(*shape, 3).copy()
    omega = vtk_to_numpy(grid.GetPointData().GetArray('vorticity')).reshape(*shape, 3)
    gt = read_dataset(root/flow['gt'])
    identifiers, low, high, cell_labels = gt_instance_bounds(gt)
    locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(gt); locator.BuildLocator()
    centers = vtk.vtkCellCenters(); centers.SetInputData(gt); centers.Update()
    gt_points = vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).copy()
    vl, inside, _ = interpolate_scalar(gt_points, axes, lam)
    vo, _, _ = interpolate_scalar(gt_points, axes, oyf)
    candidate = inside & (vl < flow['lambda2_threshold']) & (vo > 0)
    heads, cosine = head_mask(vector_at(gt_points, axes, velocity), vector_at(gt_points, axes, omega))
    spacing = np.array([np.median(np.diff(a)) for a in axes])
    pad = np.maximum((high-low)*spec['sampling']['bbox_relative_padding'],
                     spacing*spec['sampling']['bbox_minimum_grid_padding'])
    box_low = np.maximum(low-pad, [a[0] for a in axes])
    box_high = np.minimum(high+pad, [a[-1] for a in axes])
    roles, split_groups = instance_split(identifiers, box_low, box_high, spec['sampling']['split_seed']+flow_index)
    split_locators = {}
    for role in ('train', 'test'):
        chosen = np.flatnonzero(np.isin(cell_labels, [i for i, r in roles.items() if r == role]))
        selection = vtk.vtkIdList(); selection.SetNumberOfIds(len(chosen))
        for j, value in enumerate(chosen):
            selection.SetId(j, int(value))
        extract = vtk.vtkExtractCells(); extract.SetInputData(gt); extract.SetCellList(selection); extract.Update()
        subset = vtk.vtkUnstructuredGrid(); subset.DeepCopy(extract.GetOutput())
        loc = vtk.vtkStaticCellLocator(); loc.SetDataSet(subset); loc.BuildLocator()
        split_locators[role] = loc
    # Candidate *centers* use trilinear lambda2, rather than the old all-eight rule.
    cell_lam = np.zeros(np.array(shape)-1, np.float64)
    cell_oyf = np.zeros_like(cell_lam)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                sl = np.s_[dz:dz+shape[0]-1, dy:dy+shape[1]-1, dx:dx+shape[2]-1]
                cell_lam += lam[sl]/8; cell_oyf += oyf[sl]/8
    native = cell_centers(np.flatnonzero((cell_lam < flow['lambda2_threshold']) & (cell_oyf > 0)), cell_lam.shape, axes)
    near = np.zeros(len(native), bool)
    for a, b in zip(box_low, box_high):
        near |= ((native >= a) & (native <= b)).all(1)
    native = native[near]
    native_labels, _ = sample_gt(gt, native, locator)
    native = native[native_labels < 0]
    # Overlapping boxes do not duplicate a negative center: nearest actual GT wins.
    nearest = []
    for point in native:
        closest = [0., 0., 0.]; cell = vtk.reference(0); sub = vtk.reference(0); dist = vtk.reference(0.)
        locator.FindClosestPoint(point, closest, cell, sub, dist)
        nearest.append(int(cell_labels[int(cell)]))
    nearest = np.array(nearest)
    instances = []
    for j, identifier in enumerate(identifiers):
        iid = int(identifier)
        positive = gt_points[(cell_labels == iid) & candidate]
        hp = gt_points[(cell_labels == iid) & candidate & heads]
        negative = native[(nearest == iid) & ((native >= box_low[j]) & (native <= box_high[j])).all(1)]
        instances.append(dict(instance=iid, role=roles[iid], gt_bounds=np.stack((low[j], high[j])),
            box_bounds=np.stack((box_low[j], box_high[j])), positive=positive, head=hp, negative=negative,
            annotated_head_voxels=int(((cell_labels == iid) & heads).sum())))
    return dict(flow=flow, flow_index=flow_index, axes=axes, lambda2=lam, oyf=oyf,
        velocity=velocity, omega=omega, grid=grid, gt=gt, locator=locator, cell_labels=cell_labels,
        roles=roles, split_groups=split_groups, instances=instances, info=info, split_locators=split_locators)


def scene_report(scene):
    rows = []
    for item in scene['instances']:
        rows.append({key: item[key] for key in ('instance', 'role', 'annotated_head_voxels')} | dict(
            gt_bounds=item['gt_bounds'].tolist(), box_bounds=item['box_bounds'].tolist(),
            positive_candidate_centers=len(item['positive']), head_candidate_centers=len(item['head']),
            negative_candidate_centers=len(item['negative'])))
    return dict(flow=scene['flow']['name'], source=scene['info'], instances=rows, overlap_groups=scene['split_groups'],
        training_instances=[r['instance'] for r in rows if r['role'] == 'train'],
        test_instances=[r['instance'] for r in rows if r['role'] == 'test'],
        missing_candidate_pools=[r['instance'] for r in rows if min(r['positive_candidate_centers'],
            r['head_candidate_centers'], r['negative_candidate_centers']) == 0])


def local_label(ids):
    """User-confirmed labels of the retained seed set, never an entire head."""
    ids = np.asarray(ids)
    if len(ids) == 0:
        return None, -1
    values, counts = np.unique(ids[ids >= 0], return_counts=True)
    if len(values) == 0:
        return 0, -1
    best = int(np.argmax(counts))
    if counts[best]*2 >= len(ids) and np.sum(counts == counts[best]) == 1:
        return 1, int(values[best])
    return None, -1


def nearest_instance(scene, point):
    import vtk
    closest = [0., 0., 0.]; cell = vtk.reference(0); sub = vtk.reference(0); dist = vtk.reference(0.)
    scene['locator'].FindClosestPoint(point, closest, cell, sub, dist)
    return int(scene['cell_labels'][int(cell)])


def sample_bundle(scene, item, spec, number, scale_id, mandatory=False, label=1):
    pool = item['head' if mandatory else ('positive' if label else 'negative')]
    if not len(pool):
        return None
    rng = np.random.default_rng(np.random.SeedSequence([spec['sampling']['seed'], scene['flow_index'],
        item['instance'], number, label, int(mandatory)]))
    # Cycle through cells before revisiting; each revisit has a new continuous jitter.
    position = pool[number % len(pool)]
    axes = scene['axes']
    idx = [np.clip(np.searchsorted(a, position[j])-1, 0, len(a)-2) for j, a in enumerate(axes)]
    spacing = np.array([a[i+1]-a[i] for a, i in zip(axes, idx)])
    center = position+rng.uniform(-.2, .2, 3)*spacing
    ids, _ = sample_gt(scene['gt'], center[None], scene['locator'])
    if label and ids[0] != item['instance']:
        return None
    if not label and (ids[0] >= 0 or nearest_instance(scene, center) != item['instance']):
        return None
    if np.any(center < item['box_bounds'][0]) or np.any(center > item['box_bounds'][1]):
        return None
    center_head, cosine = head_mask(vector_at(center[None], axes, scene['velocity']), vector_at(center[None], axes, scene['omega']))
    if mandatory and not center_head[0]:
        return None
    scales = scale_plan(spec, scene['flow']['name'])
    scale = scales[scale_id]
    distance = float(np.prod(spacing)**(1/3))*scale['neighbor_grid_scale']
    offsets = np.array([[x, y, z] for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)], float)
    offsets[[0, 13]] = offsets[[13, 0]]
    seeds = center+offsets*distance
    lam, inside, _ = interpolate_scalar(seeds, axes, scene['lambda2'])
    oyf, _, _ = interpolate_scalar(seeds, axes, scene['oyf'])
    good = inside & (lam < scene['flow']['lambda2_threshold']) & (oyf > 0)
    good &= ((seeds >= item['box_bounds'][0]) & (seeds <= item['box_bounds'][1])).all(1)
    if not good[0] or good.sum() < spec['bundles']['minimum_valid_lines']:
        return None
    return dict(seeds=seeds[good], center=center, label=label, instance=item['instance'] if label else -1,
        owner_instance=item['instance'], head_component=-1, scale_id=scale_id, center_number=number,
        mandatory_head=mandatory, center_is_head=bool(center_head[0]), center_abs_cosine=float(cosine[0]),
        neighbor_distance=distance, local_grid_scale=float(np.prod(spacing)**(1/3)),
        seed_rms_distance=float(np.linalg.norm(seeds[good]-center, axis=1).mean()))


def scale_plan(spec, flow):
    return [dict(step, neighbor_grid_scale=r) for r in spec['sampling']['neighbor_grid_scales']
            for step in spec['integration'][flow]]


def accept_bundle(scene, item, row):
    import vtk
    n = row['line_count']
    seeds = row['normalized_seeds'][:n]*row['radius']+row['centroid']
    ids, _ = sample_gt(scene['gt'], seeds, scene['locator'])
    label, instance = local_label(ids)
    if label != row['label'] or (label and instance != item['instance']):
        return False, 'mixed_or_other_instance_seed_label'
    if row['mandatory_head'] and not np.allclose(seeds[0], row['center'], atol=2e-6, rtol=1e-6):
        return False, 'head_center_line_was_removed'
    if any(scene['roles'][int(i)] != item['role'] for i in ids if i >= 0):
        return False, 'seed_in_opposite_instance'
    # Reject intersections of actual model-input segments with opposite-split GT.
    # Boxes are only an acceleration step; overlapping boxes alone do not reject.
    g = row['geometry'][:n]*row['radius']+row['centroid']
    starts = g[:, :-1].reshape(-1, 3); delta = np.diff(g, axis=1).reshape(-1, 3)
    possible = np.zeros(len(starts), bool)
    for opposite in scene['instances']:
        if opposite['role'] == item['role']:
            continue
        low, high = opposite['gt_bounds']
        # If a segment enters the box, exact GT intersection is checked below.
        if np.any(row['bounds'][1] < low) or np.any(row['bounds'][0] > high):
            continue
        nonzero = np.abs(delta) > 1e-30
        a = np.divide(low-starts, delta, out=np.full_like(delta, -np.inf), where=nonzero)
        b = np.divide(high-starts, delta, out=np.full_like(delta, np.inf), where=nonzero)
        parallel_outside = (~nonzero & ((starts < low) | (starts > high))).any(1)
        enter = np.maximum(np.minimum(a, b).max(1), 0)
        leave = np.minimum(np.maximum(a, b).min(1), 1)
        possible |= (enter <= leave) & ~parallel_outside
    loc = scene['split_locators']['test' if item['role'] == 'train' else 'train']
    if possible.any() and any(loc.FindCell(point) >= 0 for point in g[:, 0]):
        return False, 'curve_starts_inside_opposite_instance'
    t = vtk.reference(0.); sub = vtk.reference(0); cell = vtk.reference(0)
    point = [0., 0., 0.]; coordinates = [0., 0., 0.]
    for j in np.flatnonzero(possible):
        if loc.IntersectWithLine(starts[j], starts[j]+delta[j], 1e-10, t, point, coordinates, sub, cell):
            return False, 'curve_intersects_opposite_instance'
    row['seed_gt_ids'] = np.pad(ids, (0, 27-n), constant_values=-2)
    row['instance'] = instance
    return True, None


def generate_instance(scene, item, spec, target_per_class=None):
    target = int(target_per_class or spec['sampling']['bundles_per_class_per_instance'])
    minimum_heads = spec['sampling']['mandatory_head_bundles']
    rows = []; rejected = {}; started = time.perf_counter()
    stages = [(1, True, minimum_heads), (1, False, target-minimum_heads), (0, False, target)]
    if target < minimum_heads:
        raise ValueError('Pilot still requires all mandatory head bundles')
    scales = scale_plan(spec, scene['flow']['name'])
    for label, mandatory, quota in stages:
        kept = 0; iteration = 0
        while kept < quota and iteration < spec['sampling']['maximum_batches_per_stage']:
            scale_id = (iteration % len(scales)) if not mandatory else ((iteration % len(spec['sampling']['neighbor_grid_scales']))*3+1)
            # Three integration lengths can share a center; radius groups use distinct centers.
            base = (iteration//len(scales))*32+(scale_id//3)*8
            if mandatory:
                base = iteration*8+10_000_000
            samples = []
            for offset in range(8):
                number = base+offset+(20_000_000 if label == 0 else 0)
                value = sample_bundle(scene, item, spec, number, scale_id, mandatory, label)
                if value is not None:
                    samples.append(value)
            iteration += 1
            if not samples:
                rejected['no_valid_seed_neighborhood'] = rejected.get('no_valid_seed_neighborhood', 0)+8
                continue
            generated, failures, _ = trace_batch(scene['grid'], samples, scales[scale_id], spec)
            for failure in failures:
                key = failure['reason']; rejected[key] = rejected.get(key, 0)+1
            for row in generated:
                accepted, reason = accept_bundle(scene, item, row)
                if accepted and kept < quota:
                    rows.append(row); kept += 1
                elif not accepted:
                    rejected[reason] = rejected.get(reason, 0)+1
        if kept != quota:
            raise RuntimeError(f"{scene['flow']['name']} instance {item['instance']} label={label} head={mandatory}: "
                f"{kept}/{quota}; no quota or cleaning relaxation permitted; rejected={rejected}")
    centers = np.stack([r['center'] for r in rows])
    assert len(np.unique(centers[[r['mandatory_head'] for r in rows]], axis=0)) == minimum_heads
    signatures = [(r['label'], r['mandatory_head'], r['center_number'], r['scale_id']) for r in rows]
    assert len(set(signatures)) == len(rows)
    for row in rows:
        n = row['line_count']; g = row['geometry'][:n]
        assert 10 <= n <= 27 and np.isfinite(g).all() and np.all(np.ptp(g, axis=1) > 0)
        assert np.allclose(g.mean((0, 1)), 0, atol=2e-6)
        assert abs(np.linalg.norm(g, axis=-1).max()-1) < 2e-6
        assert np.all(row['geometry'][n:] == 0)
        length = row['ds']*row['maxiteration']; arcs = row['half_arc_lengths'][:n]
        assert np.all(arcs >= .95*length) and np.all(arcs <= 1.002*length)
        assert np.all(row['half_step_counts'][:n] <= row['maxiteration'])
        if row['mandatory_head']:
            assert row['label'] == 1 and row['center_is_head'] and row['center_abs_cosine']**2 < .5
    return rows, dict(instance=item['instance'], role=item['role'], bundles=len(rows),
        positive=target, negative=target, mandatory_head_bundles=minimum_heads,
        distinct_centers=len(np.unique(centers, axis=0)), rejected=rejected,
        seconds=time.perf_counter()-started)


def save_instance(path, rows, report):
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    np.save(path/'geometry.npy', np.stack([r['geometry'] for r in rows]))
    np.save(path/'seeds.npy', np.stack([r['normalized_seeds'] for r in rows]))
    keys = ('center', 'centroid', 'radius', 'bounds', 'instance', 'owner_instance', 'scale_id',
            'center_number', 'mandatory_head', 'center_is_head', 'center_abs_cosine', 'neighbor_distance',
            'local_grid_scale', 'ds', 'maxiteration', 'requested_half_length', 'half_arc_lengths',
            'half_step_counts', 'seed_gt_ids')
    metadata = {key: np.asarray([r[key] for r in rows]) for key in keys}
    metadata['labels'] = np.array([r['label'] for r in rows], np.int64)
    metadata['counts'] = np.array([r['line_count'] for r in rows], np.int64)
    metadata['head_component'] = np.full(len(rows), -1, np.int64)
    np.savez_compressed(path/'metadata.npz', **metadata)
    write(path/'coverage.json', report)


def sparse_voxels(geometry, counts, resolution):
    """Losslessly store the frozen float16 voxel representation at occupied voxels."""
    voxels = bundle_voxels(geometry, counts, resolution, 4).half()
    flat = voxels.flatten(2).transpose(1, 2)
    # A tiny mass can round to float16 zero while its mean tangent is nonzero.
    owners, index = flat.ne(0).any(-1).nonzero(as_tuple=True)
    lengths = torch.bincount(owners, minlength=len(geometry)).cpu().numpy()
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    return offsets, index.cpu().numpy().astype(np.int32), flat[owners, index].cpu().numpy()


def dense_sparse_batch(parts, indices, resolution, device):
    """Each occupied index occurs once per bundle; no floating atomic accumulation."""
    locations = []; values = []
    for batch_index, (part, row) in enumerate(indices):
        p = parts[part]; start, stop = p['offsets'][row:row+2]
        locations.append(np.asarray(p['indices'][start:stop], np.int64)+batch_index*resolution**3)
        values.append(np.asarray(p['values'][start:stop]))
    result = torch.zeros((len(indices)*resolution**3, 4), device=device, dtype=torch.float32)
    if locations:
        result[torch.as_tensor(np.concatenate(locations), device=device)] = torch.as_tensor(
            np.concatenate(values), device=device, dtype=torch.float32)
    return result.reshape(len(indices), resolution, resolution, resolution, 4).permute(0, 4, 1, 2, 3).contiguous()
