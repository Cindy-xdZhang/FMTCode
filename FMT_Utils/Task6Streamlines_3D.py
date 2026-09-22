"""Arc-length streamline integration in a fixed instantaneous v-u snapshot."""
import numpy as np
from numba import njit,prange


@njit(cache=True)
def sample_vector(field,p,lower,spacing):
    nz,ny,nx,_=field.shape
    q=(p-lower)/spacing
    if q[0]<0 or q[1]<0 or q[2]<0 or q[0]>nx-1 or q[1]>ny-1 or q[2]>nz-1:
        return np.zeros(3),False
    x=min(int(q[0]),nx-2);y=min(int(q[1]),ny-2);z=min(int(q[2]),nz-2)
    a=q[0]-x;b=q[1]-y;c=q[2]-z;v=np.zeros(3)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                weight=(a if dx else 1-a)*(b if dy else 1-b)*(c if dz else 1-c)
                for j in range(3):v[j]+=weight*field[z+dz,y+dy,x+dx,j]
    speed=np.sqrt(np.dot(v,v))
    if speed<1e-10:return v,False
    return v/speed,True


@njit(cache=True)
def trace_one(field,start,lower,spacing,step,count,sign):
    points=np.empty((count+1,3),np.float64);points[0]=start;total=1;reason=0
    for i in range(count):
        p=points[i];k1,ok=sample_vector(field,p,lower,spacing)
        if not ok:reason=1;break
        k2,ok=sample_vector(field,p+sign*step*k1*.5,lower,spacing)
        if not ok:reason=1;break
        k3,ok=sample_vector(field,p+sign*step*k2*.5,lower,spacing)
        if not ok:reason=1;break
        k4,ok=sample_vector(field,p+sign*step*k3,lower,spacing)
        if not ok:reason=1;break
        new=p+sign*step*(k1+2*k2+2*k3+k4)/6
        _,ok=sample_vector(field,new,lower,spacing)
        if not ok:reason=1;break
        points[total]=new;total+=1
    return points[:total],reason


@njit(parallel=True,cache=True)
def integrate_samples(field,seeds,lower,spacing,half_length,step,stored_points=65):
    count=int(np.ceil(half_length/step));step=half_length/count
    curves=np.empty((len(seeds),stored_points,3),np.float32)
    lengths=np.zeros((len(seeds),2),np.float32);reasons=np.zeros((len(seeds),2),np.int8)
    for n in prange(len(seeds)):
        back,rb=trace_one(field,seeds[n],lower,spacing,step,count,-1)
        forward,rf=trace_one(field,seeds[n],lower,spacing,step,count,1)
        points=np.empty((len(back)+len(forward)-1,3),np.float64)
        for i in range(len(back)):points[i]=back[len(back)-1-i]
        for i in range(1,len(forward)):points[len(back)-1+i]=forward[i]
        distance=np.zeros(len(points))
        for i in range(1,len(points)):
            diff=points[i]-points[i-1];distance[i]=distance[i-1]+np.sqrt(np.dot(diff,diff))
        lengths[n,0]=distance[len(back)-1];lengths[n,1]=distance[-1]-lengths[n,0]
        reasons[n,0]=rb;reasons[n,1]=rf
        for j in range(3):curves[n,:,j]=np.interp(np.linspace(0,distance[-1],stored_points),distance,points[:,j])
    return curves,lengths,reasons


def candidate_cells(mask):
    shape=np.array(mask.shape)-1;cells=np.zeros(tuple(shape),bool)
    for dz in (0,1):
        for dy in (0,1):
            for dx in (0,1):cells|=mask[dz:dz+shape[0],dy:dy+shape[1],dx:dx+shape[2]]
    return np.argwhere(cells)


def seed_candidate_region(ivd,axes,count,rng,threshold=None):
    from scipy.interpolate import RegularGridInterpolator
    threshold=float(ivd.max())*.5 if threshold is None else float(threshold)
    cells=candidate_cells(ivd>threshold)
    if not len(cells):raise ValueError('No candidate cells')
    interpolation=RegularGridInterpolator(axes,ivd,bounds_error=False,fill_value=0)
    output=[];values=[];accepted=0;drawn=0
    while accepted<count:
        n=max(4096,min(200000,4*(count-accepted)))
        index=cells[rng.integers(len(cells),size=n)];fraction=rng.random((n,3))
        coordinates=np.stack([axes[a][index[:,a]]+(axes[a][index[:,a]+1]-axes[a][index[:,a]])*fraction[:,a] for a in range(3)],-1)
        # Accept using the exact coordinates that will be written to disk.
        coordinates=coordinates.astype(np.float32).astype(np.float64)
        val=interpolation(coordinates);mask=val>threshold
        valid=coordinates[mask][:count-accepted];valid_values=val[mask][:count-accepted]
        output.append(valid[:,::-1]);values.append(valid_values);accepted+=len(valid);drawn+=n
        if accepted<count and drawn>max(100000,count*1000):raise RuntimeError('Candidate-region rejection sampler could not reach its target')
    return np.ascontiguousarray(np.concatenate(output),dtype=np.float32),np.concatenate(values).astype(np.float32),drawn
