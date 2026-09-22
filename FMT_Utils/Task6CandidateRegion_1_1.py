"""Explicit invalid boundary-layer exclusions for candidate selection only."""
import numpy as np


def crop_candidate_region(ivd,axes_xyz,trim_xyz):
    trim=np.asarray(trim_xyz,int)
    if trim.shape!=(3,2) or np.any(trim<0):raise ValueError('Expected three lower/upper layer counts')
    slices=[];axes=[]
    for a,(lower,upper) in zip(axes_xyz,trim):
        stop=len(a)-int(upper)
        if stop-int(lower)<2:raise ValueError('Candidate domain must retain at least two grid points per axis')
        s=slice(int(lower),stop);slices.append(s);axes.append(a[s])
    return ivd[tuple(slices[::-1])],axes
