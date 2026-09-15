"""Versioned length-range sampling; the original v9.15 implementation is frozen."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from FMT_Utils.Task4C_InstanceCoverage_9_15 import (
    write, load_scene, scene_report, sample_bundle, scale_plan, accept_bundle,
    sparse_voxels, dense_sparse_batch, head_mask, local_label,
    save_instance as save_original_instance,
)
from experiments.Task4C_PhysicalLength_4_14 import trace_batch as original_trace_batch
from FMT_Utils.Task4C_PaperBundles_3_1 import normalize_view


def configure_lengths(spec,definition):
    """Resolve the user's length convention; retain an explicit integer step cap."""
    import copy
    spec=copy.deepcopy(spec)
    assert definition in ('per_direction','bidirectional_total')
    spec['integration']['length_definition']=definition
    factor=1 if definition=='per_direction' else 2
    for flow,limits,preferred in [('channel',(.001,.005),[.001,.002,.003,.004,.005]),
                                  ('tbl',(.01,.05),[.01,.02,.03,.04,.05])]:
        lower,upper=spec['integration']['length_ranges'][flow]
        # Small interior margins accommodate polyline arc-length roundoff;
        # every retained raw and resampled curve still passes the explicit range.
        margin=.01*(upper-lower)
        nominal=spec['integration']['mandatory_head_target'][flow]
        targets=np.unique(np.round(np.r_[np.linspace(lower+margin,upper-margin,11),nominal],10))
        steps=[]
        for i,target in enumerate(targets):
            half=target/factor
            desired=(.002 if flow=='channel' else .05) if abs(target-nominal)<1e-9 else preferred[i%len(preferred)]
            minimum=int(np.ceil(half/limits[1]-1e-10));maximum=int(np.floor(half/limits[0]+1e-10))
            count=int(np.clip(round(half/desired),minimum,maximum))
            steps.append(dict(ds=float(half/count),maxiteration=count,requested_user_length=float(target)))
        spec['integration'][flow]=steps
    validate_spec(spec)
    return spec


def validate_spec(spec):
    assert spec['user_version'] == 'task4-c-v9.15_v2'
    assert spec['integration']['length_definition'] in ('per_direction', 'bidirectional_total')
    assert spec['sampling']['mandatory_head_bundles'] == 3
    assert spec['expected_counts']['train'] == 119*2*spec['sampling']['bundles_per_class_per_instance']
    assert spec['expected_counts']['test'] == 13*2*spec['sampling']['bundles_per_class_per_instance']
    for flow, limits in [('channel', (.001,.005)),('tbl',(.01,.05))]:
        lower,upper=spec['integration']['length_ranges'][flow]
        factor=1 if spec['integration']['length_definition']=='per_direction' else 2
        lengths=[]
        for step in spec['integration'][flow]:
            assert limits[0]-1e-12<=step['ds']<=limits[1]+1e-12
            assert isinstance(step['maxiteration'],int) and step['maxiteration']>=2
            length=factor*step['ds']*step['maxiteration']
            assert lower-1e-10<=length<=upper+1e-10
            lengths.append(length)
        assert len(lengths)>=9 and len(np.unique(np.round(lengths,9)))==len(lengths)


def trace_batch(grid,samples,scale,spec,flow):
    """Check raw integration and the actual resampled model-input curve lengths."""
    rows,failures,stats=original_trace_batch(grid,samples,scale,spec)
    lower,upper=spec['integration']['length_ranges'][flow]
    half_factor=1 if spec['integration']['length_definition']=='per_direction' else .5
    half_low,half_high=lower*half_factor,upper*half_factor
    total_low,total_high=2*half_low,2*half_high
    epsilon=1e-6*total_high
    accepted=[]
    for row in rows:
        n=row['line_count']; arcs=row['half_arc_lengths'][:n]
        world=row['geometry'][:n].astype(np.float64)*row['radius']+row['centroid']
        seeds=row['normalized_seeds'][:n].astype(np.float64)*row['radius']+row['centroid']
        sampled_arcs=np.linalg.norm(np.diff(world,axis=1),axis=-1).sum(1)
        good=((arcs>=half_low-epsilon)&(arcs<=half_high+epsilon)).all(1)
        good&=(sampled_arcs>=total_low-epsilon)&(sampled_arcs<=total_high+epsilon)
        if good.sum()<spec['bundles']['minimum_valid_lines']:
            failures.append(dict(reason='fewer_than_10_lines_in_explicit_length_range',label=row['label']))
            continue
        physical=world[good]
        geometry,normalized_seeds,centroid,radius=normalize_view(physical,seeds[good],max_lines=27)
        count=int(good.sum()); kept_arcs=np.full((27,2),np.nan); kept_steps=np.zeros((27,2),np.int32)
        kept_arcs[:count]=arcs[good];kept_steps[:count]=row['half_step_counts'][:n][good]
        totals=np.full(27,np.nan);resampled=np.full(27,np.nan)
        totals[:count]=kept_arcs[:count].sum(1);resampled[:count]=sampled_arcs[good]
        row.update(geometry=geometry,normalized_seeds=normalized_seeds,centroid=centroid,radius=radius,
            line_count=count,bounds=np.stack((physical.min((0,1)),physical.max((0,1)))),
            half_arc_lengths=kept_arcs,half_step_counts=kept_steps,raw_total_arc_lengths=totals,
            resampled_total_arc_lengths=resampled,requested_total_length=2*row['requested_half_length'])
        accepted.append(row)
    return accepted,failures,stats


def generate_instance(scene,item,spec,target_per_class=None):
    target=int(target_per_class or spec['sampling']['bundles_per_class_per_instance'])
    required=spec['sampling']['mandatory_head_bundles'];assert target>=required
    rows=[]; rejected={};started=time.perf_counter()
    scales=scale_plan(spec,scene['flow']['name'])
    lengths=spec['integration'][scene['flow']['name']];nlength=len(lengths)
    nradius=len(spec['sampling']['neighbor_grid_scales'])
    nominal=spec['integration']['mandatory_head_target'][scene['flow']['name']]
    factor=1 if spec['integration']['length_definition']=='per_direction' else 2
    middle=int(np.argmin([abs(factor*s['ds']*s['maxiteration']-nominal) for s in lengths]))
    for label,mandatory,quota in [(1,True,required),(1,False,target-required),(0,False,target)]:
        kept=iteration=0
        while kept<quota and iteration<spec['sampling']['maximum_batches_per_stage']:
            scale_id=iteration%len(scales) if not mandatory else (iteration%nradius)*nlength+middle
            # All lengths at a radius share source centers; successive cycles are new centers.
            base=(iteration//len(scales))*nradius*8+(scale_id//nlength)*8
            if mandatory:base=iteration*8+10_000_000
            samples=[]
            for offset in range(8):
                number=base+offset+(20_000_000 if label==0 else 0)
                sample=sample_bundle(scene,item,spec,number,scale_id,mandatory,label)
                if sample is not None:samples.append(sample)
            iteration+=1
            if not samples:
                rejected['no_valid_seed_neighborhood']=rejected.get('no_valid_seed_neighborhood',0)+8
                continue
            generated,failures,_=trace_batch(scene['grid'],samples,scales[scale_id],spec,scene['flow']['name'])
            for failure in failures:
                reason=failure['reason'];rejected[reason]=rejected.get(reason,0)+1
            for row in generated:
                keep,reason=accept_bundle(scene,item,row)
                if keep and kept<quota:rows.append(row);kept+=1
                elif not keep:rejected[reason]=rejected.get(reason,0)+1
        if kept!=quota:
            raise RuntimeError(f"{scene['flow']['name']} instance {item['instance']}: label={label}, head={mandatory}, "
                f"{kept}/{quota}; no relaxed labels, length limits or duplicated quota fillers; {rejected}")
    centers=np.stack([r['center'] for r in rows]);flags=np.array([r['mandatory_head'] for r in rows])
    assert np.unique(centers[flags],axis=0).shape[0]==required
    assert len({(r['label'],r['mandatory_head'],r['center_number'],r['scale_id']) for r in rows})==len(rows)
    raw=[];sampled=[]
    for row in rows:
        n=row['line_count'];g=row['geometry'][:n];assert 10<=n<=27
        assert np.isfinite(g).all() and np.all(np.ptp(g,axis=1)>0)
        assert np.allclose(g.mean((0,1)),0,atol=2e-6) and abs(np.linalg.norm(g,axis=-1).max()-1)<2e-6
        assert np.all(row['geometry'][n:]==0) and np.all(row['half_step_counts'][:n]<=row['maxiteration'])
        if row['mandatory_head']:
            assert row['label']==1 and row['center_is_head'] and row['center_abs_cosine']**2<.5
        raw.extend(row['raw_total_arc_lengths'][:n]);sampled.extend(row['resampled_total_arc_lengths'][:n])
    return rows,dict(instance=item['instance'],role=item['role'],bundles=len(rows),positive=target,negative=target,
        mandatory_head_bundles=required,distinct_centers=len(np.unique(centers,axis=0)),rejected=rejected,
        length_definition=spec['integration']['length_definition'],
        raw_total_length_minmax=[float(np.min(raw)),float(np.max(raw))],
        resampled_total_length_minmax=[float(np.min(sampled)),float(np.max(sampled))],
        integration_index_counts={str(i):sum(r['scale_id']%nlength==i for r in rows) for i in range(nlength)},
        seconds=time.perf_counter()-started)


def save_instance(path,rows,report):
    save_original_instance(path,rows,report)
    path=Path(path)/'metadata.npz'
    with np.load(path) as z:metadata={k:z[k] for k in z.files}
    for key in ('requested_total_length','raw_total_arc_lengths','resampled_total_arc_lengths'):
        metadata[key]=np.asarray([r[key] for r in rows])
    np.savez_compressed(path,**metadata)
