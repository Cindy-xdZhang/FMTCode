"""Candidate-only sampling around unfiltered C++ label geometry."""
import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from FMT_Utils.Task6Streamlines_3D import candidate_cells,seed_candidate_region


@njit(cache=True)
def point_segment_distances(points,a,b,ids,offsets):
    result=np.full(len(points),np.inf)
    for i in range(len(points)):
        for j in range(offsets[i],offsets[i+1]):
            k=ids[j];d=b[k]-a[k];s=np.dot(d,d)
            t=min(1.,max(0.,np.dot(points[i]-a[k],d)/s)) if s else 0.
            q=points[i]-a[k]-t*d
            value=np.sqrt(np.dot(q,q))
            if value<result[i]:result[i]=value
    return result


class SegmentQuery:
    """Independent midpoint-tree bound followed by exact segment projection."""
    def __init__(self,cores):
        self.a=np.concatenate([c[:-1] for c in cores]).astype(np.float64) if cores else np.empty((0,3))
        self.b=np.concatenate([c[1:] for c in cores]).astype(np.float64) if cores else np.empty((0,3))
        self.tree=cKDTree((self.a+self.b)/2) if len(self.a) else None
        self.half_max=float(np.linalg.norm(self.b-self.a,axis=1).max()/2) if len(self.a) else 0.

    def distance(self,points,radius,workers=4):
        points=np.asarray(points,np.float64)
        if self.tree is None:return np.full(len(points),np.inf)
        output=np.empty(len(points))
        for first in range(0,len(points),4096):
            p=points[first:first+4096]
            groups=self.tree.query_ball_point(p,np.nextafter(radius+self.half_max,np.inf),workers=workers)
            counts=np.array([len(g) for g in groups],np.int64);offsets=np.r_[0,np.cumsum(counts)]
            ids=np.concatenate(groups).astype(np.int64) if offsets[-1] else np.empty(0,np.int64)
            output[first:first+len(p)]=point_segment_distances(p,self.a,self.b,ids,offsets)
        return output


class CandidateSampler:
    def __init__(self,ivd,axes,cores,h,threshold,seed=96611):
        self.ivd,self.axes,self.cores,self.h,self.threshold=ivd,axes,cores,h,threshold
        self.rng=np.random.default_rng(seed)
        self.interpolate=RegularGridInterpolator(axes,ivd,bounds_error=False,fill_value=0.)
        self.cells=[];self.queries=[];self.support=[]
        mask=ivd>threshold
        for cid,core in enumerate(cores):
            core=np.asarray(core,np.float64);radius=2*h
            lo=core.min(0)[::-1]-radius;hi=core.max(0)[::-1]+radius
            start=np.array([max(0,np.searchsorted(a,v,side='right')-1) for a,v in zip(axes,lo)])
            stop=np.array([min(len(a)-1,np.searchsorted(a,v,side='left')+1) for a,v in zip(axes,hi)])
            slices=tuple(slice(a,b+1) for a,b in zip(start,stop))
            cells=candidate_cells(mask[slices])+start
            query=SegmentQuery([core])
            if len(cells):
                centers=np.column_stack([(a[cells[:,i]]+a[cells[:,i]+1])/2 for i,a in enumerate(axes)])[:,::-1]
                half_diagonal=max(np.linalg.norm(np.array([np.diff(a).max() for a in axes]))/2,0.)
                distance=query.distance(centers,radius+half_diagonal)
                cells=cells[distance<=radius+half_diagonal]
            self.cells.append(cells);self.queries.append(query)
            row=dict(core=cid,possible_candidate_cells=len(cells),candidate_support=False)
            if len(cells):
                points=self._draw(cid,256,limit=200000)
                row.update(candidate_support=True,pilot_accepted=len(points))
            else:row['reason']='No candidate-containing grid cell intersects conservative 2h tube bound'
            self.support.append(row)
        self.eligible=[r['core'] for r in self.support if r['candidate_support']]

    def _draw(self,cid,count,limit=None):
        cells=self.cells[cid];pieces=[];accepted=0;drawn=0
        if not len(cells):raise ValueError('No candidate support')
        limit=max(100000,count*1000) if limit is None else limit
        while accepted<count:
            n=max(1024,min(100000,4*(count-accepted)))
            selected=cells[self.rng.integers(len(cells),size=n)];alpha=self.rng.random((n,3))
            points=np.column_stack([a[selected[:,i]]+alpha[:,i]*(a[selected[:,i]+1]-a[selected[:,i]]) for i,a in enumerate(self.axes)])[:,::-1].astype(np.float32)
            valid=self.interpolate(points[:,::-1])>self.threshold
            points=points[valid]
            if len(points):points=points[self.queries[cid].distance(points,2*self.h)<2*self.h]
            points=points[:count-accepted];pieces.append(points);accepted+=len(points);drawn+=n
            if accepted<count and drawn>limit:
                raise RuntimeError(f'Core {cid}: candidate tube support unresolved after {drawn} draws; do not silently omit it')
        return np.concatenate(pieces)

    def targets(self,count):
        ids=np.full(count,-1,np.int32)
        if self.eligible:ids[count//2:]=np.array(self.eligible)[np.arange(count-count//2)%len(self.eligible)]
        return ids

    def propose(self,targets):
        seeds,values,_=seed_candidate_region(self.ivd,self.axes,len(targets),self.rng,self.threshold)
        for cid in np.unique(targets[targets>=0]):
            rows=np.flatnonzero(targets==cid);seeds[rows]=self._draw(int(cid),len(rows))
            values[rows]=self.interpolate(seeds[rows,::-1])
        return seeds,values
