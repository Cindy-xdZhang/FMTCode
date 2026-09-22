"""Geometry-only tangent jitter, independent of labels, flow names and grids."""
import torch


@torch.no_grad()
def tangent_jitter(points, noise, sigma=.10, maximum=.25):
    """Shift interior vertices along centered tangents; fix both endpoints.

    The unit of displacement is the smaller adjacent segment length. Noise
    is supplied by the caller so augmentation need not consume dropout RNG.
    """
    if points.shape[-1] != 3 or points.shape[-2] < 3 or noise.shape != points.shape[:-1]:
        raise ValueError('Expected [..., point, 3] geometry and one scalar per point')
    if not (0 <= sigma <= maximum < .5):
        raise ValueError('Require 0 <= sigma <= maximum < 0.5')
    if sigma == 0:
        return points.clone()
    segment = torch.linalg.vector_norm(points[...,1:,:]-points[...,:-1,:],dim=-1)
    local = torch.minimum(segment[...,:-1],segment[...,1:])
    tangent = points[...,2:,:]-points[...,:-2,:]
    norm = torch.linalg.vector_norm(tangent,dim=-1,keepdim=True)
    unit = tangent / norm.clamp_min(torch.finfo(points.dtype).tiny)
    offset = (noise[...,1:-1]*sigma).clamp(-maximum,maximum)*local
    result = points.clone()
    result[...,1:-1,:] += unit*offset[...,None]
    return result


def set_dropout(model, probability):
    changed = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Dropout):
            changed.append(dict(name=name, before=layer.p, after=probability))
            layer.p = probability
    if len(changed) != 2 or any(x['before'] != .15 for x in changed):
        raise ValueError(f'Unexpected frozen p35 dropout layout: {changed}')
    return changed
