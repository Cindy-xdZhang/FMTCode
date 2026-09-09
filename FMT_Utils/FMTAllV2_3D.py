"""Objective neighbour geometry followed by temporal Fourier coefficients.

fmt_all_v2 version 1.0: six MATERIAL neighbours relative to their centre at
each time, 21 Gram scalars, initial-scale normalization and initial-Gram
subtraction, then six rFFT bins. Output 21 * (6+5) = 231 scalars.
There is no temporal difference of point coordinates or vector components.
"""
from __future__ import annotations
import numpy as np


def fmt_all_v2(pathlines, num_freq=6, chunk_size=4096):
    """Return float32 [N,231]; do geometric arithmetic in float64.

Under x_i*(t)=Q(t)x_i(t)+c(t), D*(t)=D(t)Q(t)^T and
G*(t)=D*(t)D*(t)^T=G(t). Initial normalization/subtraction and the
Fourier transform act only on invariant scalar sequences. Neighbour identity
and sample times must match across observers; no reseeding along new axes.
Frequencies are cycles per sampled trajectory window, not necessarily Hz.
"""
    if hasattr(pathlines, 'detach'):
        pathlines=pathlines.detach().cpu().numpy()
    x=np.asarray(pathlines)
    if x.ndim!=4 or x.shape[1]!=7 or x.shape[-1]<3 or x.shape[2]<2:
        raise ValueError('Require [N,7,L,C>=3] with corresponding material neighbours.')
    if not 1<=int(num_freq)<=x.shape[2]//2+1:
        raise ValueError('Invalid number of Fourier bins.')
    if int(chunk_size)<=0:
        raise ValueError('chunk_size must be positive.')
    out=np.empty((len(x),21*(2*int(num_freq)-1)),dtype=np.float32)
    upper=np.triu_indices(6)
    for start in range(0,len(x),int(chunk_size)):
        a=np.asarray(x[start:start+int(chunk_size),...,:3],dtype=np.float64)
        if not np.isfinite(a).all():raise ValueError('Nonfinite coordinates; do not filter per method.')
        offsets=(a[:,1:]-a[:,:1]).transpose(0,2,1,3)
        gram=offsets@offsets.swapaxes(-1,-2)
        scale=np.diagonal(gram[:,0],axis1=-2,axis2=-1).mean(axis=-1)
        if np.any(scale<=0):raise ValueError('Zero initial neighbour scale.')
        sequence=gram[:,:,upper[0],upper[1]]/scale[:,None,None]
        sequence=sequence-sequence[:,:1]
        spectrum=np.fft.rfft(sequence,axis=1)[:,:int(num_freq)]
        value=np.concatenate((spectrum.real.transpose(0,2,1).reshape(len(a),-1),
                              spectrum.imag[:,1:].transpose(0,2,1).reshape(len(a),-1)),axis=1)
        if not np.isfinite(value).all():raise ValueError('Nonfinite invariant coefficients.')
        out[start:start+len(a)]=value
    return out
