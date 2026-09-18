"""Task4-c fixed dataset v2: seed-point samples with three vortex lines each, no neighbours (2.1).

Rules (user, 2026-09-18 evening; docs/Task4C_fixed_dataset_rules_v2.md):
* A sample is a 3-D seed point.  Its three curves are the bidirectional vortex lines of
  half length L_1 < L_2 < L_3 (Channel 0.07/0.10/0.13, TBL 3.5/5/6.5) integrated with the
  frozen fixed RK45 step (Channel 0.001, TBL 0.025).  No neighbour lines are generated;
  methods query neighbouring samples themselves.
* step1: candidate region = {lambda2 < threshold and oyf > 0} (trilinear at the point).
* step2: M uniform-by-volume points inside the candidate region -> Poisson-disk dart
  throwing (largest radius that still yields N points) -> N samples -> one integration to
  the longest length, shorter lengths are prefixes -> all three curves must pass cleaning.
* step3: label = GT cell membership of the seed.  Assignment (grouping only): GT cell ->
  GT box -> candidate component overlapping an instance -> nearest instance.
* step4: instances (overlapping GT boxes tied) are balanced into five folds per flow;
  fold 0 is the first test fold, the others train.
"""
from pathlib import Path
import json
import time
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_GTHeadCoverage_1_1 as coverage
from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt, gt_instance_bounds
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
from FMT_Utils.Task4C_Bundles_1_1 import resample_line

KIND_GT_CELL, KIND_GT_BOX, KIND_COMPONENT, KIND_NEAREST = 0, 1, 2, 3
SPLIT_TRAIN, SPLIT_TEST = 0, 1
METADATA_KEYS = ('label', 'instance', 'assignment_kind', 'gt_owner', 'fold', 'split', 'component', 'component_size',
                 'lambda2', 'oyf', 'local_grid_scale', 'half_arc_lengths', 'half_step_counts', 'half_termination', 'boundary_relaxed',
                 'curve_bounds', 'requested_half_length', 'half_steps', 'ds', 'poisson_radius', 'pool_index', 'lambda2_threshold')
OUT_OF_DOMAIN = 1          # vtkStreamTracer ReasonForTermination: the half line left the flow domain
FILES = ('seeds.npy', 'curves.npy', 'metadata.npz')


# ---------------------------------------------------------------- step1: candidate region
def candidate_mask(lam, oyf, threshold):
    return (lam < threshold) & (oyf > 0)


def candidate_components(mask):
    from skimage import measure
    labels, count = measure.label(mask, connectivity=3, return_num=True)
    sizes = np.bincount(labels.ravel(), minlength=count+1); sizes[0] = 0
    return labels, sizes, int(count)


def candidate_cells(mask):
    """Flat indices (nz-1, ny-1, nx-1) of grid cells with at least one candidate vertex."""
    nz, ny, nx = mask.shape
    any_vertex = np.zeros((nz-1, ny-1, nx-1), bool)
    for k in (0, 1):
        for j in (0, 1):
            for i in (0, 1):
                any_vertex |= mask[k:k+nz-1, j:j+ny-1, i:i+nx-1]
    return np.flatnonzero(any_vertex.ravel())


def cell_corners(cells, axes):
    """Lower corner and edge lengths of flat cells indexed in (nz-1, ny-1, nx-1) order."""
    shape = tuple(len(a)-1 for a in axes[::-1])
    k, j, i = np.unravel_index(cells, shape)
    low = np.column_stack([axes[0][i], axes[1][j], axes[2][k]])
    size = np.column_stack([np.diff(axes[0])[i], np.diff(axes[1])[j], np.diff(axes[2])[k]])
    return low, size


def initial_pool(axes, lam, oyf, threshold, cells, count, seed, chunk):
    """``count`` points uniform by physical volume inside the trilinear candidate region."""
    low, size = cell_corners(cells, axes); volume = size.prod(1); probability = volume/volume.sum()
    rng = np.random.default_rng([seed]); parts = []; total = 0; drawn = 0
    while total < count:
        chosen = rng.choice(len(cells), size=chunk, p=probability); u = rng.random((chunk, 3))
        points = low[chosen]+u*size[chosen]
        l, inside, _ = interpolate_scalar(points, axes, lam); o, _, _ = interpolate_scalar(points, axes, oyf)
        keep = inside & (l < threshold) & (o > 0)
        parts.append(points[keep]); total += int(keep.sum()); drawn += chunk
    pool = np.concatenate(parts)[:count]
    return pool, dict(requested=int(count), drawn=int(drawn), accepted_before_truncation=int(total), acceptance=total/drawn,
                      candidate_cells=int(len(cells)), candidate_cell_volume=float(volume.sum()))


# ---------------------------------------------------------------- step2a: Poisson-disk downsampling
@njit(cache=True)
def _dart_throw(points, radius, low, dims, capacity, target):
    n = points.shape[0]; cells = dims[0]*dims[1]*dims[2]
    count = np.zeros(cells, np.int32); items = np.full((cells, capacity), -1, np.int32)
    accepted = np.empty(target, np.int64); found = 0; r2 = radius*radius; overflow = False
    for idx in range(n):
        x = points[idx, 0]; y = points[idx, 1]; z = points[idx, 2]
        cx = int((x-low[0])/radius); cy = int((y-low[1])/radius); cz = int((z-low[2])/radius)
        ok = True
        for a in range(max(cx-1, 0), min(cx+2, dims[0])):
            for b in range(max(cy-1, 0), min(cy+2, dims[1])):
                for c in range(max(cz-1, 0), min(cz+2, dims[2])):
                    cell = (a*dims[1]+b)*dims[2]+c
                    for t in range(count[cell]):
                        j = items[cell, t]
                        dx = points[j, 0]-x; dy = points[j, 1]-y; dz = points[j, 2]-z
                        if dx*dx+dy*dy+dz*dz < r2:
                            ok = False; break
                    if not ok: break
                if not ok: break
            if not ok: break
        if ok:
            cell = (cx*dims[1]+cy)*dims[2]+cz
            if count[cell] >= capacity:
                overflow = True; break
            items[cell, count[cell]] = idx; count[cell] += 1
            accepted[found] = idx; found += 1
            if found >= target: break
    return accepted[:found], overflow


def dart_throw(points, radius, target, capacity=8, max_cells=300_000_000):
    """Greedy Poisson-disk acceptance in the given order; returns accepted indices (at most ``target``)."""
    low = points.min(0)-1e-12; dims = (np.floor((points.max(0)-low)/radius).astype(np.int64)+1)
    if int(np.prod(dims)) > max_cells: raise ValueError(f'Poisson grid too fine: {dims}')
    accepted, overflow = _dart_throw(np.ascontiguousarray(points, np.float64), float(radius), low.astype(np.float64), dims, int(capacity), int(target))
    if overflow: raise ValueError('More accepted points in one grid cell than capacity; impossible for a valid Poisson-disk set')
    return accepted


def poisson_disk(points, target, radius_guess, tolerance=1e-4, capacity=8):
    """Largest radius (bisection) for which dart throwing in the given order accepts >= target points."""
    assert len(points) >= target > 0
    log = []
    def feasible(r):
        accepted = dart_throw(points, r, target, capacity); ok = len(accepted) >= target
        log.append(dict(radius=float(r), accepted=int(len(accepted)), feasible=bool(ok))); return ok, accepted
    r = float(radius_guess); ok, best = feasible(r)
    if ok:
        low, low_accepted = r, best
        while True:
            r *= 2; ok, acc = feasible(r)
            if not ok: high = r; break
            low, low_accepted = r, acc
    else:
        high = r
        while True:
            r /= 2; ok, acc = feasible(r)
            if ok: low, low_accepted = r, acc; break
    while (high-low) > tolerance*low:
        mid = .5*(low+high); ok, acc = feasible(mid)
        if ok: low, low_accepted = mid, acc
        else: high = mid
    return low_accepted[:target], low, log


# ---------------------------------------------------------------- step2b: integration and cleaning
def half_steps(half_lengths, ds):
    steps = [int(round(L/ds)) for L in half_lengths]
    for L, n in zip(half_lengths, steps): assert abs(n*ds-L) < 1e-9, (L, ds)
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
    return steps


def half_arc_band(terminated_out_of_domain, integration):
    """Half lines that left the fluid domain may be short: [0.25L, 1.25L]; otherwise [0.95L, 1.002L] (user rule 2026-09-18)."""
    if terminated_out_of_domain:
        return integration['out_of_domain_min_half_arc_fraction'], integration['out_of_domain_max_half_arc_fraction']
    return integration['minimum_actual_half_arc_fraction'], integration['maximum_actual_half_arc_fraction']


def cut_and_clean(back, front, termination, ds, steps, integration):
    """The k-th curve is the first ``steps[k]`` RK45 steps of each half; cleaning per curve.

    ``termination`` holds the VTK reason of the backward and forward half.  A half that ended
    because it left the domain only needs [0.25L, 1.25L] of arc; the curve is then flagged
    ``relaxed``.  Everything else keeps the v1 band [0.95L, 1.002L].
    """
    points = integration['points_per_curve']; K = len(steps)
    curves = np.zeros((K, points, 3), np.float32); valid = np.zeros(K, bool); relaxed = np.zeros(K, bool)
    arcs = np.full((K, 2), np.nan); counts = np.zeros((K, 2), np.int32); bounds = np.full((K, 2, 3), np.nan)
    reasons = []
    for k, n in enumerate(steps):
        L = ds*n; halves = (back[:n+1], front[:n+1])
        counts[k] = [len(h)-1 for h in halves]
        arcs[k] = [np.linalg.norm(np.diff(h, axis=0), axis=1).sum() if len(h) > 1 else 0. for h in halves]
        curve = np.concatenate((halves[0][:0:-1], halves[1]))
        if not np.isfinite(curve).all() or len(curve) <= 2: reasons.append('short_or_nonfinite'); continue
        ok = True
        for d in range(2):
            out = int(termination[d]) == OUT_OF_DOMAIN and counts[k, d] < n     # left the domain before finishing this length
            lo, hi = half_arc_band(out, integration)
            if not (lo*L <= arcs[k, d] <= hi*L): ok = False
            relaxed[k] |= out
        if not ok: reasons.append('insufficient_physical_length'); relaxed[k] = False; continue
        curve = curve[np.r_[True, np.linalg.norm(np.diff(curve, axis=0), axis=1) > 0]]
        if len(curve) <= 2 or np.any(np.ptp(curve, axis=0) == 0): reasons.append('axis_degenerate'); relaxed[k] = False; continue
        sampled = resample_line(curve, points)
        if sampled is None or np.any(np.ptp(sampled, axis=0) == 0): reasons.append('axis_degenerate'); relaxed[k] = False; continue
        curves[k] = sampled; valid[k] = True; bounds[k] = (curve.min(0), curve.max(0))
    return curves, valid, relaxed, arcs, counts, bounds, reasons


def trace_halves(grid, seeds, ds, n_max, integration):
    """Frozen GTHeadCoverage 1.1 tracer settings: fixed-step RK45 along ``vorticity`` to n_max steps per direction."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(np.ascontiguousarray(seeds, np.float64), deep=True))
    source = vtk.vtkPolyData(); source.SetPoints(points); limit = ds*n_max
    halves = []; reasons = {}; termination_of = np.full((len(seeds), 2), -1, np.int64)
    for direction in range(2):
        tracer = vtk.vtkStreamTracer(); tracer.SetInputData(grid); tracer.SetSourceData(source)
        tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'vorticity')
        tracer.SetIntegratorTypeToRungeKutta45(); tracer.SetInterpolatorTypeToCellLocator()
        tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.LENGTH_UNIT)
        tracer.SetInitialIntegrationStep(ds); tracer.SetMinimumIntegrationStep(ds); tracer.SetMaximumIntegrationStep(ds)
        tracer.SetMaximumPropagation(limit+ds*integration['vtk_internal_propagation_guard_steps'])
        tracer.SetMaximumNumberOfSteps(n_max); tracer.SetMaximumError(ds*integration['maximum_error_over_ds'])
        tracer.SetTerminalSpeed(1e-12); tracer.SetComputeVorticity(False)
        if direction: tracer.SetIntegrationDirectionToForward()
        else: tracer.SetIntegrationDirectionToBackward()
        tracer.Update(); output = tracer.GetOutput(); curves = {}
        if output.GetNumberOfCells():
            coords = vtk_to_numpy(output.GetPoints().GetData()); ids = vtk_to_numpy(output.GetCellData().GetArray('SeedIds'))
            termination = vtk_to_numpy(output.GetCellData().GetArray('ReasonForTermination'))
            for j in range(output.GetNumberOfCells()):
                cell = output.GetCell(j)
                curves[int(ids[j])] = coords[[cell.GetPointId(k) for k in range(min(cell.GetNumberOfPoints(), n_max+1))]].copy()
                termination_of[int(ids[j]), direction] = int(termination[j])
                key = str(int(termination[j])); reasons[key] = reasons.get(key, 0)+1
        halves.append(curves)
    return halves, reasons, termination_of


def trace_curves(grid, seeds, ds, steps, integration):
    halves, reasons, termination = trace_halves(grid, seeds, ds, steps[-1], integration)
    n = len(seeds); K = len(steps); P = integration['points_per_curve']
    curves = np.zeros((n, K, P, 3), np.float32); valid = np.zeros((n, K), bool); relaxed = np.zeros((n, K), bool)
    arcs = np.full((n, K, 2), np.nan); counts = np.zeros((n, K, 2), np.int32); bounds = np.full((n, K, 2, 3), np.nan)
    rejections = {}
    for i in range(n):
        back = halves[0].get(i, seeds[i:i+1]); front = halves[1].get(i, seeds[i:i+1])
        curves[i], valid[i], relaxed[i], arcs[i], counts[i], bounds[i], why = cut_and_clean(back, front, termination[i], ds, steps, integration)
        for w in why: rejections[w] = rejections.get(w, 0)+1
    return dict(curves=curves, valid=valid, relaxed=relaxed, arcs=arcs, counts=counts, bounds=bounds, termination=termination,
                stats=dict(termination_reasons=reasons, curve_rejections=rejections))


# ---------------------------------------------------------------- step3: label and assignment
def component_instances(mask, labels, axes, gt, locator):
    """Instance with the most overlapping candidate grid points per component (-1 when none)."""
    k, j, i = np.nonzero(mask)
    points = np.column_stack([axes[0][i], axes[1][j], axes[2][k]])
    owner, _ = sample_gt(gt, points, locator); comp = labels[k, j, i]
    mapping = np.full(int(labels.max())+1, -1, np.int64); overlap = np.zeros(int(labels.max())+1, np.int64)
    inside = owner >= 0
    if inside.any():
        pairs, counts = np.unique(np.column_stack([comp[inside], owner[inside]]), axis=0, return_counts=True)
        order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))  # per component: most overlap first, then smallest instance id
        pairs, counts = pairs[order], counts[order]
        first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
        mapping[pairs[first, 0]] = pairs[first, 1]; overlap[pairs[first, 0]] = counts[first]
    return mapping, overlap, points


def assign_instances(seeds, owner, instances, low, high, gt_points, gt_point_instance, seed_component, component_instance):
    """Assignment kinds 0..3 in rule order; labels are decided separately by ``owner``."""
    instances = np.asarray(instances, np.int64); n = len(seeds)
    instance = np.where(owner >= 0, owner, -1).astype(np.int64); kind = np.full(n, KIND_GT_CELL, np.int64)
    trees = {}
    def nearest_of(i, candidates):
        best = None
        for c in candidates:
            if c not in trees: trees[c] = cKDTree(gt_points[gt_point_instance == c])
            d = trees[c].query(seeds[i])[0]
            if best is None or d < best[0]: best = (d, c)
        return best[1]
    inside = np.all((seeds[:, None, :] >= low[None]) & (seeds[:, None, :] <= high[None]), axis=2)
    for i in np.flatnonzero(owner < 0):
        boxes = instances[inside[i]]
        if len(boxes) == 1: instance[i] = boxes[0]; kind[i] = KIND_GT_BOX
        elif len(boxes) > 1: instance[i] = nearest_of(i, boxes.tolist()); kind[i] = KIND_GT_BOX
        elif component_instance[seed_component[i]] >= 0: instance[i] = component_instance[seed_component[i]]; kind[i] = KIND_COMPONENT
        else: kind[i] = KIND_NEAREST
    rest = np.flatnonzero(kind == KIND_NEAREST)
    if len(rest):
        instance[rest] = gt_point_instance[cKDTree(gt_points).query(seeds[rest])[1]]
    assert np.all(instance >= 0)
    return instance, kind


# ---------------------------------------------------------------- step4: folds by instance
def instance_folds(instances, low, high, counts, folds, seed):
    """Overlapping GT boxes stay together; groups (largest first, seeded order for ties) go to the lightest fold."""
    instances = [int(i) for i in instances]; n = len(instances); parent = list(range(n))
    def find(i):
        while parent[i] != i: i = parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if np.all(low[i] <= high[j]) and np.all(low[j] <= high[i]): parent[find(i)] = find(j)
    groups = {}
    for i, identifier in enumerate(instances): groups.setdefault(find(i), []).append(identifier)
    groups = [sorted(g) for g in groups.values()]
    order = np.random.default_rng(seed).permutation(len(groups))
    weight = [sum(int(counts.get(i, 0)) for i in groups[g]) for g in order]
    ranked = sorted(range(len(order)), key=lambda t: -weight[t])
    totals = [0]*folds; fold_of = {}; assignment = []
    for t in ranked:
        f = int(np.argmin(totals)); totals[f] += weight[t]
        for i in groups[order[t]]: fold_of[i] = f
        assignment.append(dict(group=groups[order[t]], samples=weight[t], fold=f))
    return fold_of, dict(groups=assignment, fold_samples=totals)


# ---------------------------------------------------------------- build
def resolve_input_root(spec):
    for root in spec['input_roots']:
        if Path(root).is_dir(): return root
    raise FileNotFoundError('No input root exists on this host: '+json.dumps(spec['input_roots']))


def flow_setup(spec, index):
    physical = json.loads(Path(spec['physical_config']).read_text())
    flow = spec['flows'][index]; assert physical['flows'][index]['name'] == flow['name']
    steps = half_steps(flow['half_lengths'], flow['ds'])
    return physical, flow, steps


def load_scene(spec, index):
    physical, flow, steps = flow_setup(spec, index)
    scene = coverage.load_scene(physical, index, resolve_input_root(spec))
    scene['threshold'] = float(physical['flows'][index]['lambda2_threshold']); scene['steps'] = steps; scene['flow_spec'] = flow
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    ids, low, high, cell_labels = gt_instance_bounds(scene['gt'])
    centers = vtk.vtkCellCenters(); centers.SetInputData(scene['gt']); centers.Update()
    scene.update(instances=[int(i) for i in ids], box_low=low, box_high=high,
                 gt_points=vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).copy(), gt_point_instance=cell_labels.astype(np.int64))
    return scene


def local_grid_scale(points, axes):
    return np.exp(np.log(coverage.spacing_at(points, axes)).mean(1))


def prepare(spec, index, write, sha, identity, pilot=False):
    started = time.time(); scene = load_scene(spec, index); flow = scene['flow_spec']; name = flow['name']
    sampling = dict(spec['sampling']); integration = spec['integration']
    if pilot: sampling.update(spec['pilot'])
    out = Path(spec['output'])/('pilot' if pilot else 'physical')/name; out.mkdir(parents=True, exist_ok=False)
    axes, lam, oyf, thr = scene['axes'], scene['lambda2'], scene['oyf'], scene['threshold']
    mask = candidate_mask(lam, oyf, thr); labels, sizes, count = candidate_components(mask)
    cells = candidate_cells(mask)
    pool, pool_stats = initial_pool(axes, lam, oyf, thr, cells, sampling['pool_per_flow'], sampling['pool_seed']+index, sampling['pool_chunk'])
    order = np.random.default_rng([sampling['poisson_seed'], index]).permutation(len(pool)); pool = pool[order]
    target = sampling['samples_per_flow']
    guess = (pool_stats['candidate_cell_volume']*pool_stats['acceptance']/target)**(1/3)
    selected, radius, radius_log = poisson_disk(pool, target, guess, sampling['radius_relative_tolerance'], sampling['grid_capacity_per_cell'])
    seeds = pool[selected]; pool_index = order[selected]
    print(json.dumps(dict(flow=name, stage='poisson_done', radius=radius, bisections=len(radius_log), seconds=time.time()-started)), flush=True)
    batch = integration['trace_batch_seeds']; parts = []
    for first in range(0, len(seeds), batch):
        parts.append(trace_curves(scene['grid'], seeds[first:first+batch], flow['ds'], scene['steps'], integration))
        print(json.dumps(dict(flow=name, stage='traced', seeds=min(first+batch, len(seeds)), seconds=time.time()-started)), flush=True)
    t = {key: np.concatenate([p[key] for p in parts]) for key in ('curves', 'valid', 'relaxed', 'arcs', 'counts', 'bounds', 'termination')}
    stats = dict(termination_reasons={}, curve_rejections={})
    for p in parts:
        for key in stats:
            for k, v in p['stats'][key].items(): stats[key][k] = stats[key].get(k, 0)+v
    keep = t['valid'].all(1); valid = t['valid']
    rows = build_rows(scene, index, seeds[keep], {k: v[keep] for k, v in t.items()}, pool_index[keep], mask, labels, sizes, radius, spec)
    report = dict(version=spec['version'], flow=flow, pilot=pilot, identity=identity(), input_root=resolve_input_root(spec),
                  candidate=dict(threshold=thr, grid_points=int(mask.size), candidate_points=int(mask.sum()), components=count,
                                 component_size_quantiles_50_90_99_100=np.quantile(sizes[1:], [.5, .9, .99, 1]).tolist()),
                  pool=dict(pool_stats, seed=sampling['pool_seed']+index, poisson_seed=[sampling['poisson_seed'], index]),
                  poisson=dict(radius=radius, radius_guess=guess, target=target, bisection=radius_log),
                  integration=dict(ds=flow['ds'], half_lengths=flow['half_lengths'], half_steps=scene['steps'], **stats),
                  cleaning=dict(integrated=int(len(seeds)), kept=int(keep.sum()), removed=int((~keep).sum()),
                                valid_per_length=valid.sum(0).tolist(), relaxed_per_length_among_kept=(t['relaxed'] & keep[:, None]).sum(0).tolist(),
                                kept_with_any_out_of_domain_half=int(np.sum(keep & np.any(t['termination'] == OUT_OF_DOMAIN, axis=1))),
                                removed_rule='all three curves must be valid; no top-up', relaxed_rule='half lines that left the domain need [0.25L, 1.25L]'),
                  seconds=time.time()-started)
    report.update(save(rows, out, sha))
    report['complete'] = True; write(out/'preparation.json', report)
    print(json.dumps(dict(flow=name, stage='complete', kept=report['cleaning']['kept'], seconds=report['seconds'])), flush=True)
    return report


def build_rows(scene, index, seeds, traced, pool_index, mask, labels, sizes, radius, spec):
    axes = scene['axes']; thr = scene['threshold']
    owner, _ = sample_gt(scene['gt'], seeds, scene['locator'])
    comp_instance, overlap, candidate_points = component_instances(mask, labels, axes, scene['gt'], scene['locator'])
    k, j, i = np.nonzero(mask); comp_of_point = labels[k, j, i]
    seed_component = comp_of_point[cKDTree(candidate_points).query(seeds)[1]]
    instance, kind = assign_instances(seeds, owner, scene['instances'], scene['box_low'], scene['box_high'], scene['gt_points'], scene['gt_point_instance'], seed_component, comp_instance)
    per_instance = {int(v): int(c) for v, c in zip(*np.unique(instance, return_counts=True))}
    fold_of, fold_report = instance_folds(scene['instances'], scene['box_low'], scene['box_high'], per_instance, spec['folds']['count'], spec['folds']['seed']+index)
    fold = np.array([fold_of[int(v)] for v in instance], np.int64); split = np.where(fold == spec['folds']['test_fold'], SPLIT_TEST, SPLIT_TRAIN).astype(np.int64)
    l, inside, _ = interpolate_scalar(seeds, axes, scene['lambda2']); o, _, _ = interpolate_scalar(seeds, axes, scene['oyf'])
    assert inside.all() and np.all((l < thr) & (o > 0)), 'A kept seed violates step1'
    meta = dict(label=(owner >= 0).astype(np.int64), instance=instance, assignment_kind=kind, gt_owner=owner, fold=fold, split=split,
                component=seed_component.astype(np.int64), component_size=sizes[seed_component].astype(np.int64), lambda2=l, oyf=o,
                local_grid_scale=local_grid_scale(seeds, axes), half_arc_lengths=traced['arcs'], half_step_counts=traced['counts'],
                half_termination=traced['termination'].astype(np.int64), boundary_relaxed=traced['relaxed'], curve_bounds=traced['bounds'],
                requested_half_length=np.asarray(scene['flow_spec']['half_lengths'], np.float64), half_steps=np.asarray(scene['steps'], np.int64),
                ds=np.float64(scene['flow_spec']['ds']), poisson_radius=np.float64(radius), pool_index=pool_index.astype(np.int64), lambda2_threshold=np.float64(thr))
    assert set(meta) == set(METADATA_KEYS)
    return dict(seeds=seeds.astype(np.float64), curves=traced['curves'].astype(np.float32), metadata=meta,
                folds=dict(fold_report, fold_of_instance={str(k): v for k, v in fold_of.items()}, test_fold=spec['folds']['test_fold']),
                assignment=dict(per_kind={str(k): int(np.sum(kind == k)) for k in range(4)}, per_instance=per_instance,
                                components_with_instance=int(np.sum(comp_instance >= 0)), positive_per_instance={str(v): int(np.sum((instance == v) & (owner >= 0))) for v in scene['instances']}))


def save(rows, out, sha):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/'seeds.npy', rows['seeds']); np.save(out/'curves.npy', rows['curves']); np.savez_compressed(out/'metadata.npz', **rows['metadata'])
    m = rows['metadata']
    return dict(files={f: sha(out/f) for f in FILES}, samples=int(len(rows['seeds'])), classes=dict(non_hairpin=int(np.sum(m['label'] == 0)), hairpin=int(np.sum(m['label'] == 1))),
                splits=dict(train=int(np.sum(m['split'] == SPLIT_TRAIN)), test=int(np.sum(m['split'] == SPLIT_TEST))),
                folds=rows['folds'], assignment=rows['assignment'])


def load(folder):
    folder = Path(folder); seeds = np.load(folder/'seeds.npy'); curves = np.load(folder/'curves.npy')
    with np.load(folder/'metadata.npz') as z: meta = {k: z[k] for k in z.files}
    return seeds, curves, meta
