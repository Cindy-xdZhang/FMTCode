"""Incremental Couette-only Task4C construction; existing flows are read-only references."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy.spatial import cKDTree

from experiments.Inspect_Task4C_Couette_1_1 import ROOT as INPUT, read, sha
from FMT_Utils import Task4C_FixedDataset_2_1 as frozen
from FMT_Utils import Task4C_RawCurl_2_1 as rawcurl
from FMT_Utils import Task4C_FixedDataset_3_2 as quota
from FMT_Utils.Task4B_CrossFlow_3D import finite_difference_curl_zyx
from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt, gt_instance_bounds
from FMT_Utils.Task4C_OldCandidateQuick_1_1 import candidate_at_seeds
from FMT_Utils.Task4C_GTHeadLeg_2_2 import classify
from FMT_Utils.Task4C_BallQuery_2_5 import ball_tables

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/mainExp_Task4C_CouetteDataset_2.1'
CONFIG = ROOT/'config/mainExp_Task4C_CouetteDataset_2.1.json'
SCENE = None
WORK = None


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def status(stage, **values):
    data = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **values)
    write(OUT/'status.json', data); print(json.dumps(data), flush=True)


def config():
    return json.loads(CONFIG.read_text(encoding='utf8'))


def prepare():
    import shutil
    cfg=config();assert cfg['protocol_confirmed'];OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'pool_complete.json').exists():
        assert json.loads((OUT/'pool_complete.json').read_text())['config_sha256']==sha(CONFIG)
        return
    old=ROOT/'outputs/mainExp_Task4C_CouetteDataset_1.1'
    previous=json.loads((old/'pool_complete.json').read_text())
    assert sha(old/'pool.npy')==previous['pool_sha256']
    (OUT/'field_cache').mkdir(exist_ok=True)
    for src in [old/'pool.npy',old/'preserved_sources.json',*(old/'field_cache').glob('*.npy')]:
        dest=OUT/src.relative_to(old)
        if not dest.exists():
            try:os.link(src,dest)
            except OSError:shutil.copy2(src,dest)
        assert sha(src)==sha(dest)
    previous.update(config_sha256=sha(CONFIG),reused_positions_only=True,source_version='mainExp_Task4C_CouetteDataset_1.1')
    write(OUT/'pool_complete.json',previous)
    status('pool_ready',seeds=2000000,raw_curl=True)


def initialize():
    global SCENE, WORK
    s = config(); cache = OUT/'field_cache'
    arrays = {n:np.load(cache/(n+'.npy'), mmap_mode='r') for n in ('xyz','velocity','omega','lambda2','oyf')}
    axes = [np.load(cache/f'axis{d}.npy') for d in range(3)]
    grid = vtk.vtkStructuredGrid(); grid.SetDimensions([len(a) for a in axes])
    points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(arrays['xyz'].reshape(-1, 3), deep=False)); grid.SetPoints(points)
    omega = numpy_to_vtk(arrays['omega'].reshape(-1, 3), deep=False); omega.SetName('vorticity'); grid.GetPointData().AddArray(omega)
    gt = read(INPUT/'couette_GTs.vtk'); locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(gt); locator.BuildLocator()
    integration = json.loads((ROOT/'config/mainExp_Task4C_V2NewLabel_1.1.json').read_text())['integration']
    SCENE = dict(s=s, grid=grid, gt=gt, locator=locator, axes=axes, integration=integration, **arrays)
    WORK = {'proposal':np.load(OUT/'pool.npy', mmap_mode='r')}
    if (OUT/'selected_seeds.npy').exists():
        WORK['final'] = np.load(OUT/'selected_seeds.npy', mmap_mode='r')


def task(job):
    phase, first, stop = job; dest = OUT/'chunks'/phase/f'{first:07d}.npz'
    if dest.exists():
        return first, stop, 'cached'
    t0 = time.perf_counter(); s = SCENE; cfg = s['s']; seeds = WORK[phase][first:stop]
    steps = [35] if phase == 'proposal' else [35,50,65]
    halves, _, termination = rawcurl.trace(s['axes'], s['omega'], seeds, cfg['ds'], steps[-1])
    n = len(seeds); curves = np.zeros((n,len(steps),32,3), np.float32); valid = np.zeros((n,len(steps)), bool)
    relaxed = np.zeros_like(valid); arcs = np.zeros((n,len(steps),2)); counts = np.zeros_like(arcs, dtype=np.int32)
    bounds = np.zeros((n,len(steps),2,3))
    for j, seed in enumerate(seeds):
        back = halves[0].get(j, seed[None]); front = halves[1].get(j, seed[None])
        curves[j],valid[j],relaxed[j],arcs[j],counts[j],bounds[j],_ = rawcurl.clean(back, front, termination[j], cfg['ds'], steps, s['integration'])
    short_count = np.full(n,-1,np.int16); short_owner = np.full((n,32),-1,np.int32)
    ids = np.flatnonzero(valid[:,0])
    found, _ = sample_gt(s['gt'], curves[ids,0].reshape(-1,3), s['locator'])
    short_owner[ids] = found.reshape(-1,32); short_count[ids] = (short_owner[ids]>=0).sum(1)
    common = dict(first=np.int64(first), stop=np.int64(stop), short_count=short_count, valid=valid)
    if phase == 'final':
        selected = np.flatnonzero(valid.all(1)&(short_count>=17)); parts=[]; offsets=[0]
        for j in selected:
            back = halves[0][j]; front = halves[1][j]
            line = np.concatenate((back[:0:-1],front)); line = line[np.r_[True,np.linalg.norm(np.diff(line,axis=0),axis=1)>0]]
            parts.append(line); offsets.append(offsets[-1]+len(line))
        raw = np.concatenate(parts) if parts else np.empty((0,3),np.float32)
        owners,_ = sample_gt(s['gt'],raw,s['locator'])
        v,w = [np.column_stack([frozen.interpolate_scalar(raw,s['axes'],s[name][...,d])[0] for d in range(3)]) for name in ('velocity','omega')]
        head = classify(v,w)['head']; norms=(v*v).sum(1)*(w*w).sum(1);dot=(v*w).sum(1)
        np.testing.assert_array_equal(head,(w[:,1]>0)&(norms>0)&(2*dot*dot<norms))
        head_count=np.full(n,-1,np.int32);leg_count=head_count.copy();raw_count=np.zeros(n,np.int32)
        for j,a,b in zip(selected,offsets[:-1],offsets[1:]):
            head_count[j]=np.sum((owners[a:b]>=0)&head[a:b]);leg_count[j]=np.sum((owners[a:b]>=0)&~head[a:b]);raw_count[j]=b-a
        common.update(curves=curves,relaxed=relaxed,arcs=arcs,counts=counts,bounds=bounds,termination=termination,
            short_owner=short_owner,head_count=head_count,leg_count=leg_count,raw_count=raw_count,
            raw_points=raw,raw_owners=owners.astype(np.int32),raw_head=head,raw_rows=selected,raw_offsets=np.array(offsets,np.int64))
    dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=dest.with_suffix('.tmp.npz');np.savez_compressed(temporary,**common);temporary.replace(dest)
    return first,stop,round(time.perf_counter()-t0,2)


def run_phase(phase, workers):
    name='pool.npy' if phase=='proposal' else 'selected_seeds.npy'
    n=len(np.load(OUT/name,mmap_mode='r'));batch=config()['execution']['batch_size']
    jobs=[(phase,i,min(i+batch,n)) for i in range(0,n,batch)]
    status(phase,done=0,total=n,workers=workers)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as executor:
        for first,stop,seconds in executor.map(task,jobs):
            status(phase,done=stop,total=n,last_batch_seconds=seconds)


def select():
    if (OUT/'selection.json').exists():return
    cfg=config();pool=np.load(OUT/'pool.npy',mmap_mode='r');proposal=np.zeros(len(pool),np.int8);valid=np.zeros(len(pool),bool)
    for path in sorted((OUT/'chunks/proposal').glob('*.npz')):
        with np.load(path) as z:
            a,b=int(z['first']),int(z['stop']);proposal[a:b]=z['short_count']>=17;valid[a:b]=z['valid'][:,0]
    target=cfg['sampling']['samples_per_flow'];available=np.bincount(proposal,minlength=2)
    wanted=int(round(target/3));positive=min(wanted,int(available[1]*.98))
    assert positive>0 and available[0]>=target-positive,available.tolist()
    limits=np.array([target-positive,positive],np.int64)
    order=np.random.default_rng([cfg['sampling']['poisson_seed'],2]).permutation(len(pool))
    p=json.loads((OUT/'pool_complete.json').read_text());guess=(p['pool_stats']['candidate_cell_volume']/target)**(1/3)
    status('poisson',available=available.tolist(),limits=limits.tolist())
    chosen,radius,log=quota.poisson_disk(pool[order],proposal[order],limits,guess)
    ids=order[chosen];seeds=np.asarray(pool[ids]);np.testing.assert_array_equal(np.bincount(proposal[ids],minlength=2),limits)
    assert cKDTree(seeds).query(seeds,k=2,workers=4)[0][:,1].min()>=radius*(1-1e-10)
    np.save(OUT/'selected_seeds.npy',seeds);np.save(OUT/'selected_pool_indices.npy',ids);np.save(OUT/'proposal_labels.npy',proposal)
    write(OUT/'selection.json',dict(complete=True,target=target,available=available.tolist(),selected_by_short_label=limits.tolist(),
        intended_positive_to_negative=.5,quota_limited_by_available_positive=positive<wanted,poisson_radius=radius,bisection=log))


def finish():
    if (OUT/'data_audit.json').exists():
        existing=json.loads((OUT/'data_audit.json').read_text())
        assert existing['complete'] and existing['config_sha256']==sha(CONFIG)
        for name,digest in existing['files'].items():assert sha(OUT/'physical/couette'/name)==digest
        print('Frozen Couette data already complete; preserving files and load-index manifest.',flush=True)
        return
    cfg=config();initialize();s=SCENE;seeds=np.load(OUT/'selected_seeds.npy');pool_ids=np.load(OUT/'selected_pool_indices.npy')
    keys=('curves','valid','relaxed','arcs','counts','bounds','termination','short_count','short_owner','head_count','leg_count','raw_count')
    arrays={key:[] for key in keys}
    for path in sorted((OUT/'chunks/final').glob('*.npz')):
        with np.load(path) as z:
            for key in keys:arrays[key].append(z[key])
    a={key:np.concatenate(parts) for key,parts in arrays.items()};assert len(a['curves'])==len(seeds)
    keep=a['valid'].all(1);seeds=seeds[keep];pool_ids=pool_ids[keep];a={k:v[keep] for k,v in a.items()}
    labels=((a['short_count']>=17)&(a['head_count']>0)&(a['leg_count']>0)).astype(np.int64)
    assert candidate_at_seeds(s['axes'],s['lambda2'],s['oyf'],seeds,cfg['lambda2_threshold']).all()
    proposal=np.load(OUT/'proposal_labels.npy');np.testing.assert_array_equal(proposal[pool_ids],a['short_count']>=17)
    mask=frozen.candidate_mask(s['lambda2'],s['oyf'],cfg['lambda2_threshold']);components,sizes,_=frozen.candidate_components(mask)
    instances,lo,hi,cell_ids=gt_instance_bounds(s['gt']);cf=vtk.vtkCellCenters();cf.SetInputData(s['gt']);cf.Update()
    s.update(threshold=cfg['lambda2_threshold'],instances=instances.tolist(),box_low=lo,box_high=hi,
        gt_points=vtk_to_numpy(cf.GetOutput().GetPoints().GetData()).copy(),gt_point_instance=cell_ids,
        flow_spec=dict(name='couette',ds=cfg['ds'],half_lengths=cfg['half_lengths']),steps=[35,50,65])
    selection=json.loads((OUT/'selection.json').read_text());radius=selection['poisson_radius']
    status('instance_assignment',samples=len(seeds),positive=int(labels.sum()))
    rows=frozen.build_rows(s,2,seeds,{k:a[k] for k in ('curves','arcs','counts','termination','relaxed','bounds')},pool_ids,mask,components,sizes,radius,cfg)
    meta=rows['metadata'];meta['seed_label']=meta['label'].copy();meta['label']=labels
    meta.update(previous_label=(a['short_count']>=17).astype(np.int64),short_curve_gt_owner=a['short_owner'],
        short_curve_gt_count=a['short_count'],short_curve_gt_fraction=a['short_count'].astype(np.float32)/32,
        longest_head_count=a['head_count'],longest_leg_count=a['leg_count'],longest_evaluated=a['short_count']>=17,
        longest_raw_point_count=a['raw_count'],strict_candidate=np.ones(len(seeds),bool))
    meta['requested_parameter_interval']=meta.pop('requested_half_length')
    meta['integration_mode']=np.array('raw_curl_fixed_RK4')
    meta['local_grid_scale']=frozen.local_grid_scale(seeds,s['axes'])
    dest=OUT/'physical/couette';dest.mkdir(parents=True,exist_ok=True)
    saved=frozen.save(rows,dest,sha)
    # Independent exact GT lookup for every shortest curve; the persisted float32 geometry is the label source.
    for first in range(0,len(seeds),8192):
        owner,_=sample_gt(s['gt'],a['curves'][first:first+8192,0].reshape(-1,3),s['locator'])
        np.testing.assert_array_equal(owner.reshape(-1,32),meta['short_curve_gt_owner'][first:first+8192])
    for f in range(5):
        test=meta['fold']==f;assert not set(meta['instance'][test])&set(meta['instance'][~test])
        split=OUT/f'fold{f}';split.mkdir(exist_ok=True)
        np.save(split/'train_seed_indices.npy',np.flatnonzero(~test));np.save(split/'test_seed_indices.npy',np.flatnonzero(test))
    coverage={str(i):int(np.sum((meta['gt_owner']==i)&meta['strict_candidate'])) for i in instances}
    assert all(coverage.values()),coverage
    h=min(float(axis[-1]-axis[0])/(len(axis)-1) for axis in s['axes'])
    status('neighbors',samples=len(seeds),positive=int(labels.sum()),head_coverage=coverage)
    tables=ball_tables(seeds,3*h,workers=8);nbdir=OUT/'neighbors/couette';nbdir.mkdir(parents=True,exist_ok=True)
    for key,value in tables.items():np.save(nbdir/(key+'.npy'),value)
    write(nbdir/'manifest.json',dict(complete=True,h=h,radius_h=3,ks=[6,16],samples=len(seeds),
        files={p.name:sha(p) for p in nbdir.glob('*.npy')}))
    folds={str(f):dict(samples=int(np.sum(meta['fold']==f)),positive=int(np.sum(labels[meta['fold']==f]))) for f in range(5)}
    record=dict(complete=True,version=cfg['version'],flow='couette',config_sha256=sha(CONFIG),
        samples=len(seeds),positive=int(labels.sum()),negative=int(np.sum(labels==0)),rejected=int(np.sum(~keep)),
        half_lengths=cfg['half_lengths'],ds=cfg['ds'],steps=[35,50,65],h=h,lambda2_threshold=cfg['lambda2_threshold'],
        integration_mode=cfg['integration_mode'],parameter_intervals=cfg['parameter_intervals'],validity=cfg['validity'],strict_candidate_all=True,head_coverage=coverage,per_fold=folds,folds=rows['folds'],
        label_rule=cfg['labeling'],selection=selection,files=saved['files'],validation_samples=0,
        all_shortest_gt_requeried=True,long_head_rule_independently_checked=True)
    write(dest/'preparation.json',record);write(OUT/'data_audit.json',record)
    preserved=json.loads((OUT/'preserved_sources.json').read_text())
    for path,digest in preserved['files'].items():assert sha(Path(preserved['gallery_root'])/path)==digest,path
    write(OUT/'dataset.json',dict(version=cfg['version'],complete=True,flows={**cfg['existing_dataset'],
        'couette':dict(path='physical/couette',data_audit='data_audit.json',sha256=sha(OUT/'data_audit.json'))},
        old_flows_modified=False,validation_samples=0))
    status('complete',samples=len(seeds),positive=int(labels.sum()),negative=int(np.sum(labels==0)),per_fold=folds)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','proposal','select','final','finish','all']);parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    for phase in (['prepare','proposal','select','final','finish'] if args.phase=='all' else [args.phase]):
        if phase in ('proposal','final'):run_phase(phase,args.workers)
        else:globals()[phase]()
