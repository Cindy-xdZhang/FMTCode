"""GT-head sampling diagnostics; preserve original field integration routines."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def load_scene(spec, index, input_root=None):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow
    from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset
    from FMT_Utils.Task4C_InstanceCoverage_9_15 import vector_at, head_mask
    from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
    flow = spec['flows'][index]; root = Path(input_root or spec['input_root'])
    for key in ('flow', 'gt'):
        assert sha(root / flow[key]) == flow[key + '_sha256']
    axes, _, _, oyf, grid, info = load_flow(root / flow['flow'], flow['lambda2_threshold'])
    native = read_dataset(root / flow['flow']); shape = tuple(len(a) for a in axes[::-1])
    velocity = vtk_to_numpy(native.GetPointData().GetArray('velocity')).reshape(*shape, 3).copy()
    lam = vtk_to_numpy(native.GetPointData().GetArray('lambda2')).reshape(shape).copy()
    omega = vtk_to_numpy(grid.GetPointData().GetArray('vorticity')).reshape(*shape, 3)
    gt = read_dataset(root / flow['gt'])
    labels = vtk_to_numpy(gt.GetCellData().GetArray('VortexIds')).astype(np.int64)
    center_filter = vtk.vtkCellCenters(); center_filter.SetInputData(gt); center_filter.Update()
    points = vtk_to_numpy(center_filter.GetOutput().GetPoints().GetData()).copy()
    angle, _ = head_mask(vector_at(points, axes, velocity), vector_at(points, axes, omega))
    lv, inside, _ = interpolate_scalar(points, axes, lam)
    ov, _, _ = interpolate_scalar(points, axes, oyf)
    good = angle & inside & (lv < flow['lambda2_threshold']) & (ov > 0)
    pools = {int(i): points[(labels == i) & good] for i in np.unique(labels)}
    assert all(len(p) for p in pools.values()), 'Every GT must have head candidates'
    locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(gt); locator.BuildLocator()
    return dict(flow=flow, axes=axes, velocity=velocity, omega=omega, lambda2=lam, oyf=oyf,
        grid=grid, gt=gt, locator=locator, pools=pools, info=info)


def read_source(source, name):
    result = {}
    folder = Path(source) / name
    manifest = json.loads((folder / 'preparation.json').read_text())
    for split in ('train', 'validation', 'test'):
        path = folder / split / 'metadata.npz'
        if not path.exists(): path = folder / (split + '.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['splits'][split]['files']['metadata.npz']
        with np.load(path) as z: result[split] = {k: z[k] for k in z.files}
    return result


def spacing_at(points, axes):
    return np.column_stack([np.diff(a)[np.clip(np.searchsorted(a, points[:, d], side='right') - 1, 0, len(a)-2)]
                           for d, a in enumerate(axes)])


def candidate_points(scene, source, instance, seed, count=16000):
    from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
    from FMT_Utils.Task4C_InstanceCoverage_9_15 import vector_at, head_mask
    from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
    pool = scene['pools'][instance]
    # Existing evaluation centers only guide geometry sampling, never scores.
    original = source['test']['center']
    original_ids, _ = sample_gt(scene['gt'], original, scene['locator'])
    old = original[original_ids == instance]
    pool = np.concatenate((pool, old))
    rng = np.random.default_rng(seed)
    chosen = pool[rng.integers(len(pool), size=count)]
    xyz = chosen + rng.uniform(-.24, .24, chosen.shape) * spacing_at(chosen, scene['axes'])
    ids, _ = sample_gt(scene['gt'], xyz, scene['locator'])
    angle, cosine = head_mask(vector_at(xyz, scene['axes'], scene['velocity']), vector_at(xyz, scene['axes'], scene['omega']))
    lv, inside, _ = interpolate_scalar(xyz, scene['axes'], scene['lambda2'])
    ov, _, _ = interpolate_scalar(xyz, scene['axes'], scene['oyf'])
    good = (ids == instance) & angle & inside & (lv < scene['flow']['lambda2_threshold']) & (ov > 0)
    points = xyz[good]
    return points, np.prod(spacing_at(points, scene['axes']), axis=1)**(1/3)


def clearance(points, tree, centers, scales, factor=1.):
    result = np.ones(len(points), bool)
    for k, neighbors in enumerate(tree.query_ball_point(points, float(scales.max())*factor)):
        if neighbors:
            result[k] = np.all(np.linalg.norm(centers[neighbors]-points[k], axis=1) >= scales[neighbors]*factor)
    return result


def coordinate_pilot(spec, index, source, input_root, output):
    scene = load_scene(spec, index, input_root)
    original = read_source(source, scene['flow']['name'])
    train_tree = cKDTree(original['train']['center'])
    eval_points = np.concatenate([original[s]['center'] for s in ('validation', 'test')])
    eval_scales = np.concatenate([original[s]['local_grid_scale'] for s in ('validation', 'test')])
    eval_tree = cKDTree(eval_points)
    val_tree = cKDTree(original['validation']['center'])
    report = {'version': 'Verify_Task4C_GTHeadCoverage_1.1', 'flow': scene['flow']['name'], 'instances': []}
    for iid in scene['pools']:
        points, h = candidate_points(scene, original, iid, 97001 + index*100000 + iid)
        test = (train_tree.query(points)[0] >= h) & clearance(points, val_tree, original['validation']['center'], original['validation']['local_grid_scale'], .5)
        train = clearance(points, eval_tree, eval_points, eval_scales)
        a, b = np.flatnonzero(test), np.flatnonzero(train)
        pairs = 0
        if len(a) and len(b):
            tree = cKDTree(points[b])
            for t in a:
                near = tree.query_ball_point(points[t], 4*h[t])
                near = b[near]
                legal = np.linalg.norm(points[near]-points[t], axis=1) >= np.maximum(h[t], h[near])
                if legal.sum() >= 3: pairs += 1
                if pairs >= 10: break
        row = {'instance': iid, 'head_pool_cells': len(scene['pools'][iid]), 'valid_points': len(points),
            'legal_test_candidates': len(a), 'legal_train_candidates': len(b), 'test_centers_with_3_near_train_up_to_10': pairs}
        report['instances'].append(row)
        print(json.dumps(dict(flow=scene['flow']['name'], **row)), flush=True)
    report['missing_legal_pairs'] = [r['instance'] for r in report['instances'] if not r['test_centers_with_3_near_train_up_to_10']]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('MISSING', report['missing_legal_pairs'], flush=True)


def make_sample(scene, center, instance, scale, sid, number, group, role):
    from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
    h = float(np.prod(spacing_at(center[None], scene['axes']))**(1/3))
    offsets = np.array([[x,y,z] for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)], float)
    offsets[[0,13]] = offsets[[13,0]]
    distance = h * scale['neighbor_grid_scale']
    seeds = center + offsets * distance
    lam, inside, _ = interpolate_scalar(seeds, scene['axes'], scene['lambda2'])
    oyf, _, _ = interpolate_scalar(seeds, scene['axes'], scene['oyf'])
    good = inside & (lam < scene['flow']['lambda2_threshold']) & (oyf > 0)
    if not good[0] or good.sum() < 10: return None
    seeds = seeds[good]
    xyz = [int(np.clip(np.searchsorted(a, center[d], side='right')-1, 0, len(a)-2)) for d,a in enumerate(scene['axes'])]
    cell = int(np.ravel_multi_index(xyz[::-1], tuple(len(a)-1 for a in scene['axes'][::-1])))
    return dict(center=center, seeds=seeds, label=1, instance=instance, head_component=-instance-1,
        source_cell=cell, scale_id=sid, center_number=number, group=group, added_role=role,
        local_grid_scale=h, neighbor_distance=distance, seed_rms_distance=float(np.linalg.norm(seeds-center,axis=1).mean()),
        nearest_train_center_distance=0., nearest_same_head_train_center_distance=0.)


def generate(spec, index, source, input_root=None, pilot=False):
    from experiments.Task4C_PhysicalLength_4_14 import scale_plan, trace_batch
    scene = load_scene(spec, index, input_root)
    original = read_source(source, scene['flow']['name'])
    plan = scale_plan(spec, scene['flow']['name'])
    quota = 1 if pilot else spec['head_coverage']['test_bundles_per_instance']
    ntrain = spec['head_coverage']['train_bundles_per_test']
    old_train_tree = cKDTree(original['train']['center'])
    old_test_tree = cKDTree(original['test']['center'])
    ep = np.concatenate([original[s]['center'] for s in ('validation','test')])
    eh = np.concatenate([original[s]['local_grid_scale'] for s in ('validation','test')])
    et = cKDTree(ep); vt = cKDTree(original['validation']['center'])
    pools = {}
    for iid in scene['pools']:
        points,h = candidate_points(scene,original,iid,spec['head_coverage']['seed']+index*100000+iid)
        test = (old_train_tree.query(points)[0]>=h) & (old_test_tree.query(points)[0]>1e-10)
        test &= clearance(points,vt,original['validation']['center'],original['validation']['local_grid_scale'],.5)
        train = clearance(points,et,ep,eh) & (old_train_tree.query(points)[0]>1e-10)
        a,b = np.flatnonzero(test),np.flatnonzero(train)
        if not len(a) or not len(b): raise ValueError(f'No legal center pool for {iid}')
        pools[iid] = dict(points=points,h=h,test=a,train=b,tree=cKDTree(points[b]),cursor=0)
    reserved_test=[]; reserved_test_h=[]; reserved_train=[]; serial=0
    accepted={}; generation=[]
    targets=[(iid,q) for iid in sorted(pools) for q in range(quota)]

    def propose(iid,q,attempt):
        nonlocal serial
        pool=pools[iid];points,h=pool['points'],pool['h']
        sid=(q+attempt)%len(plan);scale=plan[sid]
        for _ in range(len(pool['test'])):
            pos=int(pool['test'][pool['cursor']%len(pool['test'])]);pool['cursor']+=1
            center=points[pos]
            if reserved_train and np.min(np.linalg.norm(np.asarray(reserved_train)-center,axis=1))<h[pos]:continue
            if reserved_test and np.min(np.linalg.norm(np.asarray(reserved_test)-center,axis=1))<.02*h[pos]:continue
            group=iid*quota+q
            item=make_sample(scene,center,iid,scale,sid,serial,group,'test')
            serial+=1
            if item is None:continue
            near=pool['train'][pool['tree'].query_ball_point(center,4*h[pos])]
            # Independent ordering avoids selecting the nearest almost-identical three centers.
            near=np.random.default_rng([spec['head_coverage']['seed'],index,iid,q,attempt,pos]).permutation(near)
            selected=[];selected_points=[]
            for j in near:
                if np.linalg.norm(points[j]-center)<max(h[pos],h[j]):continue
                if reserved_test and np.any(np.linalg.norm(np.asarray(reserved_test)-points[j],axis=1)<np.asarray(reserved_test_h)):continue
                if reserved_train and np.min(np.linalg.norm(np.asarray(reserved_train)-points[j],axis=1))<.02*h[j]:continue
                if selected_points and np.min(np.linalg.norm(np.asarray(selected_points)-points[j],axis=1))<.02*h[j]:continue
                candidate=make_sample(scene,points[j],iid,scale,sid,serial,group,'train');serial+=1
                if candidate is None:continue
                selected.append(candidate);selected_points.append(points[j])
                if len(selected)==ntrain:break
            if len(selected)<ntrain:continue
            reserved_test.append(center);reserved_test_h.append(h[pos]);reserved_train.extend(selected_points)
            return [item,*selected]
        raise ValueError(f'Exhausted paired center candidates for {scene["flow"]["name"]} instance {iid}')

    for attempt in range(spec['head_coverage']['maximum_geometry_rounds']):
        pending={i:[] for i in range(len(plan))}
        for iid,q in targets:
            key=(iid,q)
            if key in accepted:continue
            samples=propose(iid,q,attempt)
            for row in samples:row['target_instance']=iid;row['target_number']=q
            pending[samples[0]['scale_id']].extend(samples)
        measured={};failures=[]
        for sid,samples in pending.items():
            for first in range(0,len(samples),64):
                rows,rejected,stats=trace_batch(scene['grid'],samples[first:first+64],plan[sid],spec)
                for row in rows:measured.setdefault((row['target_instance'],row['target_number']),[]).append(row)
                failures.extend(rejected)
        for key,rows in measured.items():
            if len(rows)==ntrain+1 and sum(r['added_role']=='test' for r in rows)==1:
                accepted[key]=rows
        missing=[dict(instance=iid,number=q,valid_test=sum(r['added_role']=='test' for r in measured.get((iid,q),[])),
            valid_train=sum(r['added_role']=='train' for r in measured.get((iid,q),[]))) for iid,q in targets if (iid,q) not in accepted]
        record=dict(round=attempt,accepted_groups=len(accepted),required_groups=len(targets),rejected_bundles=len(failures),missing=missing)
        generation.append(record);print(json.dumps(dict(flow=scene['flow']['name'],**record)),flush=True)
        if len(accepted)==len(targets):break
    if len(accepted)!=len(targets):
        missing=[list(k) for k in targets if k not in accepted]
        raise ValueError(f'Incomplete GT geometry groups: {missing}; no hidden relaxation')
    result={role:[] for role in ('train','test')}
    for key in targets:
        for row in accepted[key]:result[row['added_role']].append(row)
    all_train=np.concatenate((original['train']['center'],np.array([r['center'] for r in result['train']])))
    tree=cKDTree(all_train)
    for row in result['test']:
        row['nearest_train_center_distance']=float(tree.query(row['center'])[0])
        local=np.array([r['center'] for r in accepted[(row['instance'],row['target_number'])] if r['added_role']=='train'])
        row['nearest_same_head_train_center_distance']=float(np.linalg.norm(local-row['center'],axis=1).min())
        assert row['nearest_train_center_distance']>=row['local_grid_scale']-1e-12
        assert row['nearest_same_head_train_center_distance']<=4*row['local_grid_scale']+1e-12
    return result,dict(flow=scene['flow']['name'],pilot=pilot,generation=generation,instances=sorted(pools),
        counts={k:len(v) for k,v in result.items()},label_rule='added_center_in_annotated_GT_head',
        reserved_failed_proposals_remain_excluded=True,maximum_geometry_rounds=spec['head_coverage']['maximum_geometry_rounds'])


def prepare(spec, index, source, input_root, output, pilot=False):
    import shutil
    from FMT_Utils.Task4C_BottomDensity_1_1 import save_added
    source=Path(source);name=spec['flows'][index]['name'];root=Path(output)/'physical'/name
    root.mkdir(parents=True,exist_ok=False)
    rows,report=generate(spec,index,source,input_root,pilot)
    metadata=read_source(source,name)
    report['source_hashes']={s:{'metadata.npz':hashlib.sha256((source/name/(s+'.npz')).read_bytes()).hexdigest()}
        for s in metadata} if (source/name/'train.npz').exists() else json.loads((source/name/'preparation.json').read_text())['splits']
    report['splits']={}
    for role in ('train','test'):
        ordered=rows[role]
        np.savez_compressed(root/('added_'+role+'_index.npz'),instance=np.array([r['instance'] for r in ordered]),
            paired_test_number=np.array([r['target_number'] for r in ordered]))
        save_added(ordered,root/('added_'+role),spec)
    if not pilot:
        for role in ('train','validation','test'):
            dest=root/role;dest.mkdir()
            src=source/name/role
            if role=='validation':
                for filename in ('geometry.npy','seeds.npy','metadata.npz'):shutil.copyfile(src/filename,dest/filename)
            else:
                extra=root/('added_'+role)
                for filename in ('geometry.npy','seeds.npy'):
                    a=np.load(src/filename,mmap_mode='r');b=np.load(extra/filename,mmap_mode='r')
                    out=np.lib.format.open_memmap(dest/filename,mode='w+',dtype=a.dtype,shape=(len(a)+len(b),*a.shape[1:]))
                    for first in range(0,len(a),512):out[first:first+512]=a[first:first+512]
                    out[len(a):]=b;out.flush();del out
                with np.load(extra/'metadata.npz') as z:added={k:z[k] for k in z.files}
                assert metadata[role].keys()==added.keys()
                np.savez_compressed(dest/'metadata.npz',**{k:np.concatenate((v,added[k].astype(v.dtype))) for k,v in metadata[role].items()})
            report['splits'][role]={'samples':len(metadata[role]['labels'])+(len(rows.get(role,[]))),
                'files':{f:sha(dest/f) for f in ('geometry.npy','seeds.npy','metadata.npz')}}
    (root/'preparation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report
