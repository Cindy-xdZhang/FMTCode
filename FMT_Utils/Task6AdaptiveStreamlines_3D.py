"""Adaptive Dormand-Prince 5(4) integration of normalized snapshot velocity."""
import numpy as np
from numba import njit,prange
from FMT_Utils.Task6Streamlines_3D import sample_vector


@njit(cache=True)
def trace_adaptive(field,start,lower,spacing,max_step,half_length,sign,tolerance=1e-8,max_steps=20000):
    points=np.empty((max_steps+1,3),np.float64);points[0]=start
    p=start.astype(np.float64);total=1;arc=0.;ds=max_step;minimum=max_step*1e-4;reason=0
    if not np.isfinite(p).all():return points[:1].copy(),3
    for attempt in range(max_steps):
        ds=min(ds,half_length-arc)
        if ds<=1e-13:break
        k1,ok=sample_vector(field,p,lower,spacing)
        if not ok:reason=1;break
        k1*=sign
        k2,ok2=sample_vector(field,p+ds*k1/5,lower,spacing);k2*=sign
        k3,ok3=sample_vector(field,p+ds*(3*k1/40+9*k2/40),lower,spacing);k3*=sign
        k4,ok4=sample_vector(field,p+ds*(44*k1/45-56*k2/15+32*k3/9),lower,spacing);k4*=sign
        k5,ok5=sample_vector(field,p+ds*(19372*k1/6561-25360*k2/2187+64448*k3/6561-212*k4/729),lower,spacing);k5*=sign
        k6,ok6=sample_vector(field,p+ds*(9017*k1/3168-355*k2/33+46732*k3/5247+49*k4/176-5103*k5/18656),lower,spacing);k6*=sign
        new=p+ds*(35*k1/384+500*k3/1113+125*k4/192-2187*k5/6784+11*k6/84)
        k7,ok7=sample_vector(field,new,lower,spacing);k7*=sign
        if not (ok2 and ok3 and ok4 and ok5 and ok6 and ok7):
            ds*=.5
            if ds<minimum:reason=1;break
            continue
        low=p+ds*(5179*k1/57600+7571*k3/16695+393*k4/640-92097*k5/339200+187*k6/2100+k7/40)
        error=np.sqrt(np.sum((new-low)**2))
        if error<=tolerance:
            points[total]=new;total+=1;p=new;arc+=ds
        factor=5. if error<1e-30 else min(5.,max(.1,.9*(tolerance/error)**.2))
        ds=min(max_step,ds*factor)
        if ds<minimum:reason=3;break
    else:reason=4
    return points[:total].copy(),reason


@njit(parallel=True,cache=True)
def integrate_samples_adaptive(field,seeds,lower,spacing,half_length=.25,max_step=.003,tolerance=1e-8,stored_points=65):
    curves=np.empty((len(seeds),stored_points,3),np.float32)
    lengths=np.zeros((len(seeds),2),np.float32);reasons=np.zeros((len(seeds),2),np.int8)
    for n in prange(len(seeds)):
        back,rb=trace_adaptive(field,seeds[n],lower,spacing,max_step,half_length,-1,tolerance)
        forward,rf=trace_adaptive(field,seeds[n],lower,spacing,max_step,half_length,1,tolerance)
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
