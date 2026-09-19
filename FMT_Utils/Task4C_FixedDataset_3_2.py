"""Class-weighted v3 sampling with global Poisson separation and frozen v2 geometry."""
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
import numpy as np
from numba import njit
from FMT_Utils import Task4C_FixedDataset_3_1 as dense

v2=dense.v2


def quotas(total, positive_fraction):
    assert total>=2 and 0<positive_fraction<1
    positive=int(round(total*positive_fraction))
    assert 0<positive<total
    return np.array([total-positive,positive],np.int64)


def corrected_fraction(selected, kept, ratio=2.):
    selected=np.asarray(selected,np.float64);kept=np.asarray(kept,np.float64)
    assert (selected>0).all() and (kept>0).all() and (kept<=selected).all()
    survival=kept/selected
    p=ratio*survival[0]/(survival[1]+ratio*survival[0])
    assert .5<p<.85, 'Unexpected cleaning imbalance; inspect before building'
    return float(p),survival.tolist()


def weighted_pool(axes,lam,oyf,threshold,cells,count,seed,chunk,positive_fraction,classify,box_low,box_high):
    """Uniform physical-volume draws within each exact GT class; fixed class quotas."""
    low,size=v2.cell_corners(cells,axes);volumes=size.prod(1)
    overlaps=np.zeros(len(cells),bool)
    for lo,hi in zip(box_low,box_high):
        overlaps|=np.all(low<=hi,1)&np.all(low+size>=lo,1)
    assert overlaps.any(), 'No candidate cell overlaps a positive GT box'
    rng=np.random.default_rng([seed]);targets=quotas(count,positive_fraction)
    pool=np.empty((count,3),np.float64);offset=0;stats=[]
    for label,target in enumerate(targets):
        ids=np.flatnonzero(overlaps) if label else np.arange(len(cells))
        probability=volumes[ids]/volumes[ids].sum();found=0;drawn=0;accepted=0
        for attempt in range(10000):
            chosen=ids[rng.choice(len(ids),size=chunk,p=probability)]
            points=low[chosen]+rng.random((chunk,3))*size[chosen]
            l,inside,_=v2.interpolate_scalar(points,axes,lam);o,_,_=v2.interpolate_scalar(points,axes,oyf)
            points=points[inside&(l<threshold)&(o>0)]
            points=points[np.asarray(classify(points))==label]
            take=min(len(points),int(target)-found)
            pool[offset+found:offset+found+take]=points[:take]
            found+=take;drawn+=chunk;accepted+=len(points)
            if found==target:break
        assert found==target, 'Class-conditional candidate sampler exhausted'
        stats.append(dict(label=label,requested=int(target),drawn=drawn,accepted_before_truncation=accepted,
                          proposal_cells=len(ids),proposal_cell_volume=float(volumes[ids].sum())))
        offset+=found
    drawn=sum(s['drawn'] for s in stats);accepted=sum(s['accepted_before_truncation'] for s in stats)
    return pool,dict(requested=int(count),drawn=drawn,accepted_before_truncation=accepted,acceptance=accepted/drawn,
        candidate_cells=len(cells),candidate_cell_volume=float(volumes.sum()),positive_fraction=positive_fraction,
        class_quotas=targets.tolist(),class_sampling=stats,rule='uniform volume within exact GT class; disjoint class quotas')


@njit(cache=True)
def _quota_dart(points,labels,radius,low,dims,limits,table_size):
    target=int(limits.sum());keys=np.full(table_size,-1,np.int64);heads=np.full(table_size,-1,np.int32)
    links=np.full(target,-1,np.int32);accepted=np.empty(target,np.int64);class_counts=np.zeros(2,np.int64)
    found=0;r2=radius*radius
    for idx in range(len(points)):
        label=labels[idx]
        if class_counts[label]>=limits[label]:continue
        x,y,z=points[idx,0],points[idx,1],points[idx,2]
        cx=int((x-low[0])/radius);cy=int((y-low[1])/radius);cz=int((z-low[2])/radius);ok=True
        for a in range(max(cx-1,0),min(cx+2,dims[0])):
            for b in range(max(cy-1,0),min(cy+2,dims[1])):
                for c in range(max(cz-1,0),min(cz+2,dims[2])):
                    key=(a*dims[1]+b)*dims[2]+c;slot=dense._slot(keys,key);entry=heads[slot]
                    while entry!=-1:
                        j=accepted[entry];dx=points[j,0]-x;dy=points[j,1]-y;dz=points[j,2]-z
                        if dx*dx+dy*dy+dz*dz<r2:ok=False;break
                        entry=links[entry]
                    if not ok:break
                if not ok:break
            if not ok:break
        if ok:
            key=(cx*dims[1]+cy)*dims[2]+cz;slot=dense._slot(keys,key);keys[slot]=key
            links[found]=heads[slot];heads[slot]=found;accepted[found]=idx
            found+=1;class_counts[label]+=1
            if found==target:break
    return accepted[:found]


def dart_throw(points,labels,radius,limits):
    points=np.ascontiguousarray(points,np.float64);labels=np.ascontiguousarray(labels,np.int64)
    limits=np.asarray(limits,np.int64);target=int(limits.sum())
    assert np.isfinite(points).all() and radius>0 and len(labels)==len(points)
    assert ((labels==0)|(labels==1)).all() and (limits>0).all() and (np.bincount(labels,minlength=2)>=limits).all()
    low=points.min(0)-1e-12;dims=np.floor((points.max(0)-low)/radius).astype(np.int64)+1
    assert int(dims[0])*int(dims[1])*int(dims[2])<2**63 and target<2**31
    table_size=1<<(2*target-1).bit_length()
    return _quota_dart(points,labels,float(radius),low,dims,limits,table_size)


def poisson_disk(points,labels,limits,guess,tolerance=1e-4,capacity=8):
    fn=FunctionType(v2.poisson_disk.__code__,dict(v2.__dict__,
        dart_throw=lambda p,r,target,cap:dart_throw(p,labels,r,limits)),argdefs=v2.poisson_disk.__defaults__)
    return fn(points,int(np.sum(limits)),guess,tolerance,capacity)


def fraction(spec,index,pilot=False):
    if pilot:return spec['balance']['positive_to_negative_ratio']/(1+spec['balance']['positive_to_negative_ratio'])
    report=json.loads((Path(spec['output'])/'calibration.json').read_text())
    assert report['complete']
    return report['flows'][spec['flows'][index]['name']]['positive_fraction']


def adapter(spec,index=None,pilot=False):
    """Bind sampling to the currently loaded scene; retain all downstream frozen functions."""
    state={}
    def scene_loader(s,i):
        state['scene']=scene=v2.load_scene(s,i);state['index']=i;state['fraction']=fraction(spec,i,pilot)
        return scene
    def classify(points):
        scene=state['scene'];owner,_=v2.sample_gt(scene['gt'],points,scene['locator'])
        return (owner>=0).astype(np.int64)
    def pool_fn(axes,lam,oyf,thr,cells,count,seed,chunk):
        scene=state['scene']
        return weighted_pool(axes,lam,oyf,thr,cells,count,seed,chunk,state['fraction'],classify,scene['box_low'],scene['box_high'])
    def dart_fn(points,radius,target,capacity=8):
        return dart_throw(points,classify(points),radius,quotas(target,state['fraction']))
    def poisson_fn(points,target,guess,tolerance=1e-4,capacity=8):
        return poisson_disk(points,classify(points),quotas(target,state['fraction']),guess,tolerance,capacity)
    def folds(instances,low,high,counts,count,seed):
        reference=spec['reference_folds'][spec['flows'][state['index']]['name']]
        return dense.fixed_instance_folds(instances,low,high,counts,count,seed,reference)
    rows=FunctionType(v2.build_rows.__code__,dict(v2.__dict__,instance_folds=folds))
    return SimpleNamespace(**dict(v2.__dict__,load_scene=scene_loader,initial_pool=pool_fn,dart_throw=dart_fn,
        poisson_disk=poisson_fn,build_rows=rows))


def prepare(spec,index,write,sha,identity,pilot=False):
    api=adapter(spec,index,pilot)
    fn=FunctionType(v2.prepare.__code__,api.__dict__,argdefs=v2.prepare.__defaults__)
    def write_report(path,report):
        p=fraction(spec,index,pilot);target=spec['pilot' if pilot else 'sampling']['samples_per_flow']
        selected=quotas(target,p);kept=[report['classes']['non_hairpin'],report['classes']['hairpin']]
        ratio=kept[1]/kept[0]
        report['balance']=dict(positive_fraction=p,selected_by_class=selected.tolist(),kept_by_class=kept,
            survival_by_class=(np.asarray(kept)/selected).tolist(),positive_to_negative_ratio=ratio,
            target_ratio=spec['balance']['positive_to_negative_ratio'])
        if not pilot:
            assert abs(ratio/spec['balance']['positive_to_negative_ratio']-1)<=spec['balance']['ratio_relative_tolerance'], report['balance']
        write(path,report)
    return fn(spec,index,write_report,sha,identity,pilot)
