"""Task4-c fixed bundle dataset v1: seed-point samples, instance-aware split, >=17 lines (1.1).

Rules (user, 2026-09-18; protocol section 2b):
* A sample is a 3-D seed point.  Its line is the center line; the 26 stencil points
  (0.25h / 0.5h / 1h) give the neighbour lines.  Only the CENTER keeps the step-1 filter
  (lambda2 < threshold and oyf > 0, same head component; GT-head centers additionally need
  GT membership and the head angle).  Neighbours only need to lie inside the domain.
* A sample is kept only when the center line and >= 16 neighbour lines survive cleaning.
* After ``normal_attempts`` consecutive failures of one head the center filter is relaxed
  (center anywhere in the head's cell box); the label of a relaxed center is the GT
  membership of the center point.
* Hairpin GT instances are split into a covered group (80 %) and a held-out group (20 %);
  overlapping GT boxes stay together.  Training/validation never enter a held-out instance
  (GT box plus one grid spacing).  Training covers every covered instance with >= N head
  centers; the test set covers EVERY instance: offset points (>= 3h from any training
  center, <= 6h from a same-instance training center) for covered instances, fresh points
  for held-out instances.  Negative heads follow the group of the nearest hairpin instance.
* Every line stores its physical seed point and the flags ``head_candidate`` / ``oyf_positive``.
"""
from pathlib import Path
import json
import time
import zlib
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_GTHeadCoverage_1_1 as coverage
from FMT_Utils import Task4C_FPS16_Data_1_2 as rules
from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt, gt_instance_bounds
from FMT_Utils.Task4C_InstanceCoverage_9_15 import vector_at, head_mask
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
from FMT_Utils.Task4C_PaperBundles_3_1 import cell_centers
from experiments import Task4C_PhysicalLength_4_14 as physical

ROLES = ('train', 'validation', 'test')
KIND_HEAD_REGION, KIND_GT_HEAD = 0, 1
FILTER_IN_DOMAIN, FILTER_RELAXED = rules.FILTER_IN_DOMAIN, rules.FILTER_RELAXED
ATTRIBUTE_FILE = rules.ATTRIBUTE_FILE
INDEX_FILE = 'index.npz'
METADATA_KEYS = ('labels', 'counts', 'head_component', 'instance', 'scale_id', 'center_number', 'source_cell', 'center',
                 'centroid', 'radius', 'bounds', 'neighbor_distance', 'seed_rms_distance', 'measured_mean_arc_length',
                 'half_arc_lengths', 'half_step_counts', 'ds', 'maxiteration', 'requested_half_length', 'local_grid_scale',
                 'nearest_train_center_distance', 'nearest_same_head_train_center_distance')
stencil, cell_of, head_center_and_neighbors, gt_head_sample = rules.stencil, rules.cell_of, rules.head_center_and_neighbors, rules.gt_head_sample
physical_seeds_of_row, line_attributes = rules.physical_seeds_of_row, rules.line_attributes


def instance_groups(identifiers, low, high, seed, fraction):
    """Held-out instances: exactly round(fraction*n) when overlapping boxes are kept together."""
    n = len(identifiers); parent = list(range(n))
    def find(i):
        while parent[i] != i: i = parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if np.all(low[i] <= high[j]) and np.all(low[j] <= high[i]): parent[find(i)] = find(j)
    groups = {}
    for i, identifier in enumerate(identifiers): groups.setdefault(find(i), []).append(int(identifier))
    groups = list(groups.values()); order = np.random.default_rng(seed).permutation(len(groups))
    target = int(round(n*fraction)); possible = {0: []}
    for g in order:
        for size, selected in list(possible.items()):
            new = size+len(groups[g])
            if new <= target and new not in possible: possible[new] = selected+[int(g)]
    best = max(k for k in possible if k <= target)
    heldout = sorted(i for g in possible[best] for i in groups[g])
    return dict(heldout=heldout, covered=sorted(int(i) for i in identifiers if int(i) not in set(heldout)),
                target_heldout=target, achieved_heldout=best, overlap_groups=groups)


class Builder:
    def __init__(self, spec, index, pilot=False):
        self.spec = spec; self.index = index; self.pilot = pilot; self.rule = spec['proposals']; self.dist = spec['distances']
        self.physical = json.loads(Path(spec['physical_config']).read_text())
        self.flow = self.physical['flows'][index]['name']; self.threshold = float(self.physical['flows'][index]['lambda2_threshold'])
        self.scene = coverage.load_scene(self.physical, index)
        from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow
        flow = self.physical['flows'][index]
        self.axes, _, heads, self.oyf, _, _ = load_flow(Path(self.physical['input_root'])/flow['flow'], flow['lambda2_threshold'])
        self.lambda2 = self.scene['lambda2']
        self.components, self.catalog, self.catalog_report = physical.build_catalog(self.axes, heads, self.scene['gt'], self.physical)
        self.plan = physical.scale_plan(self.physical, self.flow)
        # GT instances, their cells and the covered / held-out groups.
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        gt = self.scene['gt']; ids, low, high, self.cell_labels = gt_instance_bounds(gt)
        centers = vtk.vtkCellCenters(); centers.SetInputData(gt); centers.Update()
        points = vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).copy()
        self.gt_cells = {int(i): points[self.cell_labels == i] for i in ids}
        self.instances = [int(i) for i in ids]; self.bounds = {int(i): (low[k], high[k]) for k, i in enumerate(ids)}
        self.groups = instance_groups(ids, low, high, spec['instance_split']['seed']+index, spec['instance_split']['held_out_fraction'])
        self.heldout = set(self.groups['heldout'])
        spacing = np.array([np.median(np.diff(a)) for a in self.axes]); margin = spacing.max()*self.dist['heldout_exclusion_margin_over_h']
        self.exclusion = [(self.bounds[i][0]-margin, self.bounds[i][1]+margin) for i in self.heldout]
        gt_tree_points = np.concatenate([self.gt_cells[i] for i in self.instances]); owners = np.concatenate([[i]*len(self.gt_cells[i]) for i in self.instances])
        gt_tree = cKDTree(gt_tree_points)
        # Head pools per split: positive heads follow their instance, negatives the nearest instance.
        self.pools = {r: [] for r in ROLES}; self.head_group = {}
        for head in self.catalog:
            if head['label'] == 1: instance = head['instance']
            else:
                d, j = gt_tree.query(cell_centers(head['cell_ids'], self.components.shape, self.axes)); instance = int(owners[j[np.argmin(d)]])
            heldout = instance in self.heldout; self.head_group[head['head_component']] = dict(nearest_instance=instance, heldout=heldout)
            if heldout: self.pools['test'].append(dict(head, cell_ids=head['cell_ids']))
            else:
                for r in ROLES:
                    if len(head['split_cells'][r]): self.pools[r].append(dict(head, cell_ids=head['split_cells'][r]))
        self.rejections = {}; self.trace_calls = 0; self.relaxed = {r: 0 for r in ROLES}; self.failures = {}
        self.final = {}; self.final_trees = {}; self.instance_trees = {}; self.reserved = {r: [] for r in ROLES}; self.attempts = {r: 0 for r in ROLES}
        self.box_cache = {}; self.reserved_cache = {}

    # ------------------------------------------------------------------ helpers
    def reject(self, reason): self.rejections[reason] = self.rejections.get(reason, 0)+1

    def in_exclusion(self, point):
        return any(np.all(point >= lo) and np.all(point <= hi) for lo, hi in self.exclusion)

    def reserved_distance(self, role, point):
        points = self.reserved[role]; tree, n = self.reserved_cache.get(role, (None, 0))
        if len(points)-n >= 256: tree = cKDTree(np.asarray(points)); n = len(points); self.reserved_cache[role] = (tree, n)
        result = float(tree.query(point)[0]) if tree is not None else float('inf')
        if len(points) > n: result = min(result, float(np.linalg.norm(np.asarray(points[n:])-point, axis=1).min()))
        return result

    def legal_center(self, role, point, h, instance, label):
        if self.reserved_distance(role, point) <= 1e-10: self.reject('duplicate_center'); return False
        if role != 'test' and self.in_exclusion(point): self.reject('inside_heldout_exclusion'); return False
        if role == 'train': return True
        train = self.final_trees['train']
        if train.query(point)[0] < self.dist['eval_min_to_any_train_over_h']*h-1e-12: self.reject('too_close_to_train'); return False
        if role == 'test':
            if label == 1 and instance not in self.heldout:
                tree = self.instance_trees.get(instance)
                if tree is None or tree.query(point)[0] > self.dist['test_max_to_same_instance_train_over_h']*h+1e-12:
                    self.reject('too_far_from_same_instance_train'); return False
            if self.final_trees['validation'].query(point)[0] < self.dist['test_min_to_validation_over_h']*h-1e-12:
                self.reject('too_close_to_validation'); return False
        return True

    def head_sample(self, role, head, number, scale, sid, relaxed, rng):
        if relaxed:
            key = head['head_component']
            if key not in self.box_cache:
                pts = cell_centers(head['cell_ids'], self.components.shape, self.axes); sp = coverage.spacing_at(pts, self.axes)
                self.box_cache[key] = ((pts-sp).min(0), (pts+sp).max(0))
            lo, hi = self.box_cache[key]; center = rng.uniform(lo, hi)
            h = float(np.prod(coverage.spacing_at(center[None], self.axes))**(1/3)); distance = h*scale['neighbor_grid_scale']
            seeds = stencil(center, distance); _, inside, _ = interpolate_scalar(seeds, self.axes, self.oyf)
            if not inside[0]: self.reject('relaxed_center_outside_domain'); return None
            valid = inside.copy(); valid[0] = True
            owner, _ = sample_gt(self.scene['gt'], center[None], self.scene['locator'])
            label = int(owner[0] >= 0); instance = int(owner[0]) if label else -1
            sample = dict(center=center, seeds=seeds[valid], source_cell=cell_of(center, self.axes), neighbor_distance=distance,
                          seed_rms_distance=float(np.sqrt(np.mean(np.sum((seeds[valid]-center)**2, axis=1)))), local_grid_scale=h)
        else:
            sample = head_center_and_neighbors(head, self.components, self.axes, self.oyf, number, scale, self.rule['seed']+self.index*100000)
            if sample is None: self.reject('center_failed_oyf_or_head'); return None
            label = head['label']; instance = head['instance']; sample['local_grid_scale'] = sample['neighbor_distance']/scale['neighbor_grid_scale']
        sample.update(head_component=head['head_component'], instance=instance, label=label, scale_id=sid, center_number=number,
                      nearest_train_center_distance=0., nearest_same_head_train_center_distance=0., kind=KIND_HEAD_REGION,
                      relaxed=relaxed, nearest_instance=self.head_group[head['head_component']]['nearest_instance'])
        return sample

    def gt_sample(self, role, instance, number, scale, sid, relaxed, rng):
        pool = self.gt_cells[instance] if relaxed else self.scene['pools'][instance]
        if not len(pool): self.reject('empty_gt_pool'); return None
        center = pool[int(rng.integers(len(pool)))].copy()+rng.uniform(-.24, .24, 3)*coverage.spacing_at(pool[:1], self.axes)[0]
        owner, _ = sample_gt(self.scene['gt'], center[None], self.scene['locator'])
        if owner[0] != instance: self.reject('GT_membership'); return None
        if not relaxed:
            angle, _ = head_mask(vector_at(center[None], self.axes, self.scene['velocity']), vector_at(center[None], self.axes, self.scene['omega']))
            if not angle[0]: self.reject('GT_head_angle'); return None
        sample = gt_head_sample(self.scene, center, instance, scale, sid, number, instance, role, relaxed)
        if sample is None: self.reject('gt_center_failed_domain_lambda2_oyf'); return None
        sample.update(kind=KIND_GT_HEAD, relaxed=relaxed, nearest_instance=instance); return sample

    def finish(self, sample, role, h):
        if len(sample['seeds']) < self.rule['minimum_valid_lines']: self.reject('fewer_than_17_in_domain_seed_points'); return None
        if not self.legal_center(role, sample['center'], sample['local_grid_scale'], sample['instance'], sample['label']): return None
        return sample

    def trace(self, samples, sid):
        if not samples: return []
        rows, rejected, _ = physical.trace_batch(self.scene['grid'], samples, self.plan[sid], self.physical); self.trace_calls += 1
        for r in rejected: self.reject(r['reason'])
        accepted = []
        for row in rows:
            if row['line_count'] < self.rule['minimum_valid_lines']: self.reject('fewer_than_17_clean_lines'); continue
            try: row['physical_seeds'], row['stencil_slots'] = physical_seeds_of_row(row)
            except ValueError: self.reject('original_center_removed'); continue
            accepted.append(row)
        return accepted

    # ------------------------------------------------------------------ generation
    def generate_role(self, role):
        counts = self.spec['dataset']['per_flow_counts'][role]; minimum = self.spec['coverage']['min_head_centers_per_instance_per_split']
        rows = []; started = time.perf_counter()
        # 1. GT-head coverage rows: covered instances for train / validation, every instance for test.
        instances = self.instances if role == 'test' else self.groups['covered']
        if role == 'validation' and not self.spec['coverage']['validation_gt_head_rows']: instances = []
        for instance in instances:
            rows.extend(self.fill(role, [('gt', instance)], minimum, tag=f'gt{instance}'))
        # 2. Head-region rows per scale, cycling eligible heads as in 4.14.
        remaining = counts-len(rows); per_scale = [remaining//len(self.plan)+int(s < remaining % len(self.plan)) for s in range(len(self.plan))]
        for sid, quota in enumerate(per_scale):
            rows.extend(self.fill(role, [('head', h) for h in self.pools[role]], quota, tag=f'scale{sid}', fixed_sid=sid))
        rng = np.random.default_rng([self.rule['seed'], self.index, ROLES.index(role), 7]); rng.shuffle(rows)
        print(json.dumps(dict(flow=self.flow, role=role, rows=len(rows), seconds=round(time.perf_counter()-started, 1),
                              relaxed=self.relaxed[role], rejections=self.rejections)), flush=True)
        return rows

    def fill(self, role, sources, quota, tag, fixed_sid=None):
        """Draw ``quota`` accepted rows from ``sources`` (heads or a GT instance); relax a source after normal_attempts failures."""
        if quota <= 0 or not sources: return []
        accepted = []; failures = {k: 0 for k in range(len(sources))}; attempt = 0; batch = self.rule['integration_batch_templates']
        order = np.random.default_rng([self.rule['seed'], self.index, ROLES.index(role), zlib.crc32(tag.encode())])
        while len(accepted) < quota:
            pending = {s: [] for s in range(len(self.plan))}; proposed = 0
            for _ in range(min(batch, 4*(quota-len(accepted)))):
                k = int(order.integers(len(sources))); kind, source = sources[k]
                stage = failures[k]; attempt += 1; self.attempts[role] += 1
                if stage >= self.rule['normal_attempts']+self.rule['relaxed_attempts']:
                    raise ValueError(f'{self.flow}/{role}/{tag}: source {k} exhausted relaxed proposals; {self.rejections}')
                relaxed = stage >= self.rule['normal_attempts']
                sid = fixed_sid if fixed_sid is not None else int(order.integers(len(self.plan))); scale = self.plan[sid]
                number = 700000000+ROLES.index(role)*100000000+self.attempts[role]
                rng = np.random.default_rng([self.rule['seed'], self.index, ROLES.index(role), number])
                sample = self.gt_sample(role, source, number, scale, sid, relaxed, rng) if kind == 'gt' else self.head_sample(role, source, number, scale, sid, relaxed, rng)
                if sample is not None: sample = self.finish(sample, role, sample['local_grid_scale'])
                if sample is None: failures[k] += 1; continue
                sample['source_index'] = k; pending[sid].append(sample); proposed += 1
            for sid, samples in pending.items():
                for row in self.trace(samples, sid):
                    if len(accepted) >= quota: break
                    if not self.legal_center(role, row['center'], row['local_grid_scale'], row['instance'], row['label']): failures[row['source_index']] += 1; continue
                    failures[row['source_index']] = 0; accepted.append(row); self.reserved[role].append(row['center'])
                    if row['relaxed']: self.relaxed[role] += 1
            if proposed == 0 and all(f >= self.rule['normal_attempts']+self.rule['relaxed_attempts'] for f in failures.values()):
                raise ValueError(f'{self.flow}/{role}/{tag}: no source can produce a legal center; {self.rejections}')
            if attempt % 2048 < batch:
                print(json.dumps(dict(flow=self.flow, role=role, tag=tag, accepted=len(accepted), quota=quota, attempts=attempt)), flush=True)
        return accepted[:quota]

    def build(self, out):
        out.mkdir(parents=True, exist_ok=False); reports = {}
        for role in ROLES:
            rows = self.generate_role(role); folder = out/role; folder.mkdir()
            reports[role] = self.save(role, rows, folder)
            centers = np.stack([r['center'] for r in rows]); self.final[role] = rows; self.final_trees[role] = cKDTree(centers)
            if role == 'train':
                for instance in self.groups['covered']:
                    pts = np.array([r['center'] for r in rows if r['label'] == 1 and r['instance'] == instance])
                    if len(pts): self.instance_trees[instance] = cKDTree(pts)
        return reports

    def save(self, role, rows, folder):
        n = len(rows); g = np.lib.format.open_memmap(folder/'geometry.npy', mode='w+', dtype=np.float32, shape=(n, 27, 32, 3))
        seeds = np.lib.format.open_memmap(folder/'seeds.npy', mode='w+', dtype=np.float32, shape=(n, 27, 3))
        for i, r in enumerate(rows): g[i] = r['geometry']; seeds[i] = r['normalized_seeds']
        g.flush(); seeds.flush(); del g, seeds
        m = {}
        m['labels'] = np.array([r['label'] for r in rows], np.int8); m['counts'] = np.array([r['line_count'] for r in rows], np.int16)
        for key in ('head_component', 'instance', 'scale_id', 'center_number', 'source_cell'): m[key] = np.array([r[key] for r in rows], np.int64)
        for key in ('center', 'centroid', 'radius', 'bounds', 'neighbor_distance', 'seed_rms_distance', 'measured_mean_arc_length',
                    'half_arc_lengths', 'half_step_counts', 'ds', 'maxiteration', 'requested_half_length', 'local_grid_scale'):
            m[key] = np.array([r[key] for r in rows])
        m['nearest_train_center_distance'] = np.zeros(n); m['nearest_same_head_train_center_distance'] = np.zeros(n)
        if role != 'train':
            train = self.final_trees['train']; m['nearest_train_center_distance'] = train.query(m['center'])[0]
            for i, r in enumerate(rows):
                tree = self.instance_trees.get(r['instance']) if r['label'] == 1 else None
                m['nearest_same_head_train_center_distance'][i] = tree.query(r['center'])[0] if tree is not None else np.inf
            assert np.all(m['nearest_train_center_distance'] >= self.dist['eval_min_to_any_train_over_h']*m['local_grid_scale']-1e-12)
            if role == 'test':
                offset = np.array([r['label'] == 1 and r['instance'] not in self.heldout for r in rows])
                assert np.all(m['nearest_same_head_train_center_distance'][offset] <= self.dist['test_max_to_same_instance_train_over_h']*m['local_grid_scale'][offset]+1e-12)
                assert np.all(self.final_trees['validation'].query(m['center'])[0] >= self.dist['test_min_to_validation_over_h']*m['local_grid_scale']-1e-12)
        assert set(m) == set(METADATA_KEYS)
        signature = np.column_stack([m['center'], m['scale_id']]); assert len(np.unique(signature, axis=0)) == n
        points = np.stack([r['physical_seeds'] for r in rows]); slots = np.stack([r['stencil_slots'] for r in rows])
        valid = np.arange(27)[None] < m['counts'][:, None]; assert np.array_equal(np.isfinite(points).all(-1), valid) and np.all(slots[:, 0] == 0)
        attributes = line_attributes(points, self.axes, self.lambda2, self.oyf, self.threshold)
        relaxed = np.array([r['relaxed'] for r in rows]); kind = np.array([r['kind'] for r in rows], np.int8)
        assert np.all(attributes['head_candidate'][~relaxed, 0]), 'non-relaxed center violates step 1'
        owner, _ = sample_gt(self.scene['gt'], m['center'], self.scene['locator'])
        angle, _ = head_mask(vector_at(m['center'], self.axes, self.scene['velocity']), vector_at(m['center'], self.axes, self.scene['omega']))
        in_gt_head = (owner >= 0) & angle
        cov = {str(i): int(np.sum(in_gt_head & (owner == i))) for i in self.instances}
        np.savez_compressed(folder/'metadata.npz', **m)
        np.savez_compressed(folder/INDEX_FILE, sample_kind=kind, relaxed=relaxed, neighbor_filter=np.where(relaxed, FILTER_RELAXED, FILTER_IN_DOMAIN).astype(np.int8),
                            original_center_id=np.zeros(n, np.int64), nearest_instance=np.array([r['nearest_instance'] for r in rows], np.int64),
                            heldout_instance=np.array([r['nearest_instance'] in self.heldout for r in rows]), center_gt_owner=owner,
                            center_in_gt_head=in_gt_head, offset_test=np.array([role == 'test' and r['label'] == 1 and r['instance'] not in self.heldout for r in rows]))
        np.savez_compressed(folder/ATTRIBUTE_FILE, seed_points=points, stencil_slot=slots, exact_stencil_coordinates=np.ones(n, bool),
                            lambda2_threshold=np.array(self.threshold), **attributes)
        return dict(samples=n, classes=np.bincount(m['labels'], minlength=2).tolist(), gt_head_rows=int((kind == KIND_GT_HEAD).sum()),
                    relaxed=int(relaxed.sum()), heldout_instance_rows=int(np.sum([r['nearest_instance'] in self.heldout for r in rows])),
                    line_count_histogram=np.bincount(m['counts'], minlength=28)[self.rule['minimum_valid_lines']:].tolist(),
                    coverage_head_centers_per_instance=cov,
                    lines=dict(total=int(valid.sum()), head_candidate=int(attributes['head_candidate'].sum()), oyf_positive=int(attributes['oyf_positive'].sum())),
                    files={})


def prepare(spec, index, write, sha, identity, pilot=False):
    spec = json.loads(json.dumps(spec))
    if pilot:
        spec['dataset']['per_flow_counts'] = spec['pilot']['per_flow_counts']; spec['coverage']['min_head_centers_per_instance_per_split'] = spec['pilot']['min_head_centers_per_instance_per_split']
    builder = Builder(spec, index, pilot); flow = builder.flow
    out = Path(spec['output'])/('pilot' if pilot else 'physical')/flow
    reports = builder.build(out)
    for role in ROLES: reports[role]['files'] = {f.name: sha(f) for f in sorted((out/role).iterdir())}
    covered = builder.groups['covered']; minimum = spec['coverage']['min_head_centers_per_instance_per_split']
    coverage_ok = dict(train=all(reports['train']['coverage_head_centers_per_instance'][str(i)] >= minimum for i in covered),
                       test=all(reports['test']['coverage_head_centers_per_instance'][str(i)] >= minimum for i in builder.instances))
    report = dict(complete=True, pilot=pilot, identity=identity(), flow=flow, splits=reports, instance_split=builder.groups,
                  heldout_exclusion_boxes=[[lo.tolist(), hi.tolist()] for lo, hi in builder.exclusion], head_groups={str(k): v for k, v in builder.head_group.items()},
                  catalog=builder.catalog_report, rejections=builder.rejections, relaxed=builder.relaxed, trace_calls=builder.trace_calls,
                  coverage_requirement_met=coverage_ok, distances=spec['distances'], proposals=spec['proposals'],
                  spatial_cell_and_physical_length_audit_passed=True, scale_plan=builder.plan)
    write(out/'preparation.json', report)
    if not all(coverage_ok.values()): raise ValueError(f'{flow}: coverage requirement not met: {coverage_ok}')
    return report
