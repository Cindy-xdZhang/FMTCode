"""Unnormalized curl, fixed RK4 updates, and user-approved boundary validity."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from FMT_Utils.Task4C_Bundles_1_1 import resample_line


def trace(axes, omega, seeds, dt, maximum):
    sample=RegularGridInterpolator(tuple(axes[::-1]),omega,bounds_error=False,fill_value=np.nan)
    low=np.array([a[0] for a in axes]);high=np.array([a[-1] for a in axes])
    n=len(seeds);halves=[];termination=np.full((n,2),5,np.int64)
    for direction,sign in enumerate((-1,1)):
        points=np.full((n,maximum+1,3),np.nan,np.float64);points[:,0]=seeds
        counts=np.zeros(n,np.int32);alive=np.ones(n,bool);h=sign*dt
        def field(x):return sample(x[:,::-1])
        for step in range(maximum):
            ids=np.flatnonzero(alive)
            if not len(ids):break
            x=points[ids,step];k1=field(x);k2=field(x+h*k1/2);k3=field(x+h*k2/2);k4=field(x+h*k3)
            new=x+h*(k1+2*k2+2*k3+k4)/6
            inside=np.isfinite(new).all(1)&(new>=low).all(1)&(new<=high).all(1)
            moving=np.linalg.norm(new-x,axis=1)>1e-14
            good=inside&moving;ok=ids[good];points[ok,step+1]=new[good];counts[ok]+=1
            termination[ids[~inside],direction]=1
            termination[ids[inside&~moving],direction]=6
            alive[ids[~good]]=False
        halves.append({i:points[i,:counts[i]+1].copy() for i in range(n)})
    return halves,{},termination


def clean(back,front,termination,dt,steps,integration):
    k=len(steps);curves=np.zeros((k,32,3),np.float32);valid=np.zeros(k,bool);relaxed=np.zeros(k,bool)
    arcs=np.zeros((k,2));counts=np.zeros((k,2),np.int32);bounds=np.zeros((k,2,3));reasons=[]
    for j,requested in enumerate(steps):
        halves=(back[:requested+1],front[:requested+1]);counts[j]=[len(p)-1 for p in halves]
        arcs[j]=[np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for p in halves]
        enough=True
        for direction in range(2):
            incomplete=counts[j,direction]<requested
            relaxed[j]|=incomplete and termination[direction]==1
            minimum=int(np.ceil(.25*requested)) if termination[direction]==1 else requested
            enough &= counts[j,direction]>=minimum
        line=np.concatenate((halves[0][:0:-1],halves[1]));line=line[np.r_[True,np.linalg.norm(np.diff(line,axis=0),axis=1)>0]]
        if not enough or len(line)<=2 or not np.isfinite(line).all():reasons.append('insufficient_updates');continue
        sampled=resample_line(line,32)
        if sampled is None or np.any(np.ptp(sampled,axis=0)==0):reasons.append('axis_degenerate');continue
        curves[j]=sampled;valid[j]=True;bounds[j]=line.min(0),line.max(0)
    return curves,valid,relaxed,arcs,counts,bounds,reasons


def trace_curves(scene,seeds,steps=(35,50,65)):
    halves,_,termination=trace(scene['axes'],scene['omega'],seeds,.0002,steps[-1])
    parts=[clean(halves[0][i],halves[1][i],termination[i],.0002,steps,{}) for i in range(len(seeds))]
    return {key:np.array([p[j] for p in parts]) for j,key in enumerate(('curves','valid','relaxed','arcs','counts','bounds'))}
