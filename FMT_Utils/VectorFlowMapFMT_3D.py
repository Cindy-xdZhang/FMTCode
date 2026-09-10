"""Frozen six-frequency directional flow-map tokens, version 1.1."""
import numpy as np
import torch
from FMT_Utils.FlowMapData_3D import cross_offsets


def vector_fmt(paths, radius, frequencies=6):
    """Keep complex vector coefficients and line identity; no learned encoder."""
    x=np.asarray(paths,np.float32)
    if x.shape[1:]!=(7,32,3) or frequencies!=6:
        raise ValueError('Version 1.1 freezes seven lines, 32 points and six frequencies')
    local=(x-x[:,:1,:1])/np.asarray(radius,np.float32)[:,None,None,None]
    relative=local[:,1:]-local[:,:1]
    series=np.concatenate([np.diff(local[:,:1],axis=2),np.diff(relative,axis=2)],axis=1)
    spectrum=torch.fft.rfft(torch.from_numpy(series.copy()),dim=2,norm='forward')[:,:,:frequencies]
    # The DC imaginary component is identically zero and need not be stored.
    values=torch.cat([spectrum.real.flatten(1),spectrum.imag[:,:,1:].flatten(1)],dim=1)
    result=values.numpy().astype(np.float32)
    assert result.shape==(len(x),231) and np.isfinite(result).all()
    return result


def reconstruct_support(tokens, origin, radius):
    """Deterministic low-pass support reconstruction; no hidden particle truth."""
    tokens=torch.as_tensor(tokens,dtype=torch.float32)
    n=len(tokens)
    real=tokens[:,:126].reshape(n,7,6,3)
    imag=torch.cat([torch.zeros((n,7,1,3)),tokens[:,126:].reshape(n,7,5,3)],dim=2)
    full=torch.zeros((n,7,16,3),dtype=torch.complex64)
    full[:,:,:6]=torch.complex(real,imag)
    increments=torch.fft.irfft(full,n=31,dim=2,norm='forward').numpy()
    sequence=np.concatenate([np.zeros((n,7,1,3),np.float32),np.cumsum(increments,axis=2)],axis=2)
    center=sequence[:,:1]
    local=np.concatenate([center,center+sequence[:,1:]+cross_offsets()[None,1:,None]],axis=1)
    return (local*np.asarray(radius)[:,None,None,None]+np.asarray(origin)[:,None,None]).astype(np.float32)


def features(data):
    out={}
    for key in ('support0','support1','context'):
        paths=data[key]; flat=paths.reshape(-1,7,32,3)
        r=data['radius1'] if key=='support1' else data['radius0']
        if key=='context':r=np.repeat(r,6)
        out['vector_fmt6__'+key]=vector_fmt(flat,r).reshape(*paths.shape[:-3],231)
    return out


def reconstructed_data(data):
    out=dict(data)
    for key in ('support0','support1','context'):
        x=data['vector_fmt6__'+key]; r=data['radius1'] if key=='support1' else data['radius0']
        origin=data['origin1'] if key=='support1' else data['origin0']
        if key=='context':
            r=np.repeat(r,6); origin=data['context_origins'].reshape(-1,3)
        out[key]=reconstruct_support(x.reshape(-1,231),origin,r).reshape(data[key].shape)
    return out
