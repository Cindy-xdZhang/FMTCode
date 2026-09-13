"""Versioned diagnostic encoders; never modify the frozen FMT implementation."""
from __future__ import annotations
import numpy as np
import torch
from FMT_Utils.DFT_FMT_3D import dft_rotation_invariants_3d, pathline_dft_features_3d
from FMT_Utils.FMTAllV2_3D import fmt_all_v2
from FMT_Utils.Task12Data_3D import feature_matrix

ARMS = ['raw', 'old_full', 'old_no_center', 'old_neighbor_only',
        'old_recomputed_full', 'old_recomputed_no_center',
        'neighbor_undifferenced_kin', 'gram6', 'gram6_kin',
        'gram6_unscaled_kin', 'gram_delta6_kin', 'gram_full_kin',
        'gram_time_kin', 'aivd_kin']
WIDTH = 700


def gram_series(x, normalize=True):
    x = np.asarray(x, dtype=np.float64)
    d = (x[:,1:] - x[:,:1]).transpose(0,2,1,3)
    g = d @ d.swapaxes(-1,-2)
    i,j = np.triu_indices(6)
    s = g[:,:,i,j]
    if normalize:
        scale = np.diagonal(g[:,0],axis1=-2,axis2=-1).mean(-1)
        if np.any(scale <= 0):
            raise ValueError('Degenerate initial cross')
        s = s / scale[:,None,None]
    return s-s[:,:1]


def scalar_spectrum(s, bins=6, full=False):
    n = s.shape[1]
    f = np.fft.rfft(s,axis=1)
    if not full:
        f = f[:,:bins]
    imag = f[:,1:]
    if full and n % 2 == 0:
        imag = imag[:,:-1]  # Nyquist imaginary coefficient is identically zero.
    return np.concatenate([f.real.transpose(0,2,1).reshape(len(s),-1),
                           imag.imag.transpose(0,2,1).reshape(len(s),-1)],axis=1).astype(np.float32)


def feature_set(x, cached_fmt=None):
    x = np.asarray(x)
    tensor = torch.as_tensor(x,dtype=torch.float32)
    recomputed = pathline_dft_features_3d(tensor,neighbor_scale=1.0,neighbor_weight=1.0)
    old = recomputed if cached_fmt is None else np.asarray(cached_fmt)
    adapter = {'raw': x.reshape(len(x),-1), 'fmt': old, 'features':{}}
    kin = feature_matrix(adapter,'kin4','cpu')
    aivd = feature_matrix(adapter,'aivd1w3_dft','cpu')
    relative = x[:,1:]-x[:,:1]
    # Same frozen frequency invariants, scale, sorting and weight as old neighbours;
    # only omit the difference operator. This remains NON-objective under Q(t).
    undiff = dft_rotation_invariants_3d(
        torch.as_tensor(relative,dtype=torch.float32).reshape(-1,x.shape[2],3),6
    ).reshape(len(x),6,23).sort(dim=1,descending=True).values.flatten(1).numpy()
    s = gram_series(x)
    raw_center = x[:,:1]-x[:,:1,:1]
    raw = np.concatenate([raw_center,relative],axis=1).reshape(len(x),-1)
    cat = lambda a: np.concatenate([a,kin],axis=1).astype(np.float32)
    no_center = old.copy(); no_center[:,:23] = 0
    recomputed_no_center = recomputed.copy();recomputed_no_center[:,:23]=0
    neighbor_only = np.concatenate([no_center,np.zeros_like(kin)],axis=1)
    values = dict(raw=raw, old_full=cat(old), old_no_center=cat(no_center),
                  old_neighbor_only=neighbor_only,
                  old_recomputed_full=cat(recomputed),old_recomputed_no_center=cat(recomputed_no_center),
                  neighbor_undifferenced_kin=cat(np.pad(undiff,((0,0),(23,0)))),
                  gram6=fmt_all_v2(x), gram6_kin=cat(fmt_all_v2(x)),
                  gram6_unscaled_kin=cat(scalar_spectrum(gram_series(x,False))),
                  gram_delta6_kin=cat(scalar_spectrum(np.diff(s,axis=1))),
                  gram_full_kin=cat(scalar_spectrum(s,full=True)),
                  gram_time_kin=cat(s.transpose(0,2,1).reshape(len(x),-1)),
                  aivd_kin=cat(aivd))
    for key,value in values.items():
        if value.shape[1]>WIDTH or not np.isfinite(value).all():
            raise ValueError(f'Invalid feature {key}: {value.shape}')
        values[key]=np.asarray(value,dtype=np.float32)
    return values


def pad(x):
    return np.pad(x,((0,0),(0,WIDTH-x.shape[1]))).astype(np.float32)


def observe(x, local=False, translation=False):
    """One shared smooth rigid observer, or explicitly NON-observer local rotations."""
    x=np.asarray(x,dtype=np.float64)
    phase=np.linspace(0,1,x.shape[2])
    rates=np.linspace(.3,1.7,len(x)) if local else np.ones(len(x))
    angle=rates[:,None]*phase[None,:]*1.3
    c,s=np.cos(angle),np.sin(angle)
    q=np.zeros((len(x),x.shape[2],3,3))
    q[...,0,0]=c;q[...,0,1]=-s;q[...,1,0]=s;q[...,1,1]=c;q[...,2,2]=1
    if translation:
        return x+np.stack([phase,phase**2,np.sin(phase)],-1)[None,None]
    if local:
        return np.einsum('ntij,nktj->nkti',q,x-x[:,:1])+x[:,:1]
    return np.einsum('ntij,nktj->nkti',q,x)


def analytic_checks():
    """Rigid local crosses share identical Gram series but different vorticity deviations."""
    offsets=np.vstack([np.zeros(3),np.eye(3)[0],-np.eye(3)[0],
                       np.eye(3)[1],-np.eye(3)[1],np.eye(3)[2],-np.eye(3)[2]])
    x=np.broadcast_to(offsets[None,:,None,:],(3,7,32,3)).copy()
    rates=np.array([0.,1.,3.])
    t=np.linspace(0,.1,32)
    for n,w in enumerate(rates):
        a=w*t;c,s=np.cos(a),np.sin(a)
        x[n,:,:,0]=offsets[:,0,None]*c-offsets[:,1,None]*s
        x[n,:,:,1]=offsets[:,0,None]*s+offsets[:,1,None]*c
    base=feature_set(x)
    expected=2*np.abs(rates-rates.mean())
    from FMT_Utils.DFT_FMT_3D import pathline_velocity_gradient_scalar_sequences_3d
    estimated=pathline_velocity_gradient_scalar_sequences_3d(
        torch.as_tensor(x),sample_times=torch.as_tensor(t))[:,0,0].numpy()
    np.testing.assert_allclose(estimated,expected,rtol=3e-4,atol=1e-8)
    assert np.max(np.abs(base['gram6']))<1e-10
    assert np.linalg.norm(base['old_neighbor_only'])>1
    checks=[]
    for name,y in [('global_rotation',observe(x)),('translation',observe(x,translation=True)),
                   ('independent_local_rotation_NOT_observer',observe(x,local=True))]:
        after=feature_set(y)
        for arm in ARMS:
            diff=float(np.linalg.norm(after[arm]-base[arm]))
            checks.append({'transform':name,'arm':arm,'absolute_l2':diff})
        np.testing.assert_allclose(fmt_all_v2(y),fmt_all_v2(x),atol=2e-5,rtol=2e-5)
    return {'status':'PASS','analytic_vorticity_deviation':expected.tolist(),
            'finite_difference_estimate':estimated.tolist(),
            'gram6_max_absolute':float(np.max(np.abs(base['gram6']))),
            'interpretation':'Three disjoint locally rigid regions, not three global observers; local scalar Gram cannot identify inter-region rotation deviations.',
            'drift':checks}
