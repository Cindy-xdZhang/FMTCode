"""Version 1.1: three material shells and same-time Gram Fourier features."""
from itertools import product
import numpy as np


def shell_offsets(radius):
    """Center, then eight cube-vertex directions on each of r, 1.5r and 2r."""
    radius=float(radius)
    if not np.isfinite(radius) or radius<=0:raise ValueError('radius must be positive')
    directions=np.array(list(product((-1.,1.),repeat=3)))/np.sqrt(3.)
    return np.concatenate([np.zeros((1,3))]+[directions*radius*scale for scale in (1.,1.5,2.)])


def large_neighbor_fmt(pathlines,num_freq=6,chunk_size=256):
    """Return [N,3300] using all 32 times, with DC and five nonzero bins.

    d_i(t)=x_i(t)-x_0(t); G_ij(t)=dot(d_i(t),d_j(t)).
    Under x*=Q(t)x+c(t), d*=Qd and G*=G at each time. Fourier
    coefficients of these scalar sequences are therefore invariant even
    for a changing rotation. No temporal point/vector differences enter.
    This extends the frozen six-neighbor fmt_all_v2 construction to 24.
    """
    x=np.asarray(pathlines)
    if x.ndim!=4 or x.shape[1]!=25 or x.shape[-1]<3 or x.shape[2]<2:
        raise ValueError('Expected [N,25,L,C>=3] material paths')
    num_freq=int(num_freq);chunk_size=int(chunk_size)
    if not 1<=num_freq<=x.shape[2]//2+1 or chunk_size<1:raise ValueError('Invalid Fourier/chunk size')
    upper=np.triu_indices(24);result=np.empty((len(x),300*(2*num_freq-1)),np.float32)
    for start in range(0,len(x),chunk_size):
        a=np.asarray(x[start:start+chunk_size,...,:3],np.float64)
        if not np.isfinite(a).all():raise ValueError('Nonfinite paths')
        offsets=(a[:,1:]-a[:,:1]).transpose(0,2,1,3)
        gram=offsets@offsets.swapaxes(-1,-2)
        scale=np.diagonal(gram[:,0],axis1=-2,axis2=-1).mean(axis=-1)
        if np.any(scale<=0):raise ValueError('Zero initial shell scale')
        sequence=gram[:,:,upper[0],upper[1]]/scale[:,None,None]
        sequence-=sequence[:,:1].copy()
        coefficients=np.fft.rfft(sequence,axis=1)[:,:num_freq]
        result[start:start+len(a)]=np.concatenate([
            coefficients.real.transpose(0,2,1).reshape(len(a),-1),
            coefficients.imag[:,1:].transpose(0,2,1).reshape(len(a),-1)],axis=1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite Fourier features')
    return result


def integrate_shells(field,seeds,dt,radius,steps=48,samples=32,chunk_size=2048):
    """Integrate 25 material lines using the frozen production RK4 kernel."""
    from FLowUtils.flowlineIntegral import compute_pathlines_3D_batch
    target=dt*steps
    if target>field.tmax+1e-12:raise ValueError('Insufficient source frames')
    offsets=shell_offsets(radius);seeds=np.asarray(seeds,np.float64)
    expanded=(seeds[:,None]+offsets[None]).reshape(-1,3)
    expanded=np.column_stack([expanded,np.zeros(len(expanded))])
    positions=[];lengths=[]
    for start in range(0,len(expanded),chunk_size):
        p,n=compute_pathlines_3D_batch(field,expanded[start:start+chunk_size],
              min_time=0.,max_time=target,step_size=dt,max_iteration=steps,method='RK4')
        positions.append(p[:,:steps+1]);lengths.append(n)
    p=np.concatenate(positions).reshape(-1,25,steps+1,4)
    n=np.concatenate(lengths).reshape(-1,25)
    xyz=p[...,:3];lo=np.asarray(field.domainMinBoundary);hi=np.asarray(field.domainMaxBoundary)
    valid=(n==steps+1).all(axis=1)&np.isfinite(p).all(axis=(1,2,3))&((xyz>=lo)&(xyz<=hi)).all(axis=(1,2,3))
    selected=np.rint(np.linspace(0,steps,samples)).astype(int)
    return p[valid][:,:,selected],valid,n


def relative_raw(pathlines):
    """Raw comparison: initial-center-local center and same-time neighbors."""
    x=np.asarray(pathlines,np.float32)
    center=x[:,:1];neighbors=x[:,1:]-center
    return np.concatenate([center.reshape(len(x),-1),neighbors.reshape(len(x),-1)],axis=1)


def objectivity_certificate(pathlines):
    """Check implementation on the same material lines; not on resampled seeds."""
    x=np.asarray(pathlines[:32],np.float64);rng=np.random.default_rng(77129)
    matrices=[]
    for _ in range(x.shape[2]):
        q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=np.linalg.det(q);matrices.append(q)
    transformed=np.einsum('tij,nktj->nkti',np.array(matrices),x)
    transformed+=rng.normal(size=(1,1,x.shape[2],3))*max(float(np.ptp(x)),1.)
    a,b=large_neighbor_fmt(x),large_neighbor_fmt(transformed)
    relative_error=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-5)
    return {'status':'PASS','material_samples':len(x),'max_abs_error':float(np.max(np.abs(a-b))),
            'relative_error':relative_error,'transform':'independent proper rotation and translation at every time',
            'scope':'encoder only; does not certify coordinate-based Raw network'}
