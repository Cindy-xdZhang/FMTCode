"""Geometry sensitivity and local straightening for the frozen p35 classifier.

No new model, surrogate attention, or curvature-based coloring is introduced.
The six seed neighbors stay fixed while the observed curve geometry is varied.
"""
from __future__ import annotations

import numpy as np
import torch

from FMT_Utils.FMT_P35_NormFrequency_3_1 import normalize_geometry, primitive_features


WINDOWS = tuple((start, min(start + 8, 31)) for start in range(0, 25, 4))


def frozen_neighbors(seeds, counts):
    """Call on the original contiguous 32-row encoding batches, including padding."""
    mask = torch.arange(seeds.shape[1], device=seeds.device)[None] < counts[:, None]
    distances = torch.cdist(seeds, seeds)
    distances.masked_fill_(~mask[:, None, :], torch.inf)
    distances.diagonal(dim1=1, dim2=2).fill_(torch.inf)
    nearest = torch.argsort(distances, dim=-1, stable=True)[..., :6]
    anchors = torch.arange(seeds.shape[1], device=seeds.device)[None, :, None].expand(len(seeds), -1, -1)
    return torch.cat((anchors, nearest), -1)


def differentiable_tokens(geometry, counts, neighbors, norm):
    """Exact p35 operations, retaining autograd back to all valid input points."""
    x, mask = normalize_geometry(geometry, counts, 'max_radius')
    bi, li = mask.nonzero(as_tuple=True)
    result = x.new_zeros((*x.shape[:2], 141))
    for start in range(0, len(bi), 1024):
        bs, ls = bi[start:start + 1024], li[start:start + 1024]
        primitives = x[bs[:, None], neighbors[bs, ls]]
        result[bs, ls] = primitive_features(primitives, 6)
    mean, std = (torch.as_tensor(a, device=x.device, dtype=torch.float64) for a in norm)
    # The original NumPy standardizer uses float64, followed by float32 casting.
    result = ((result.double() - mean) / std).float() * mask[..., None]
    return torch.cat((result, mask[..., None].to(result.dtype)), -1)


def evaluate(model, geometry, counts, neighbors, norm):
    logits = model(differentiable_tokens(geometry, counts, neighbors, norm))
    return logits[:, 1] - logits[:, 0], logits.softmax(-1)[:, 1]


def point_gradients(model, geometry, counts, neighbors, norm):
    x = geometry.detach().clone().requires_grad_(True)
    margin, probability = evaluate(model, x, counts, neighbors, norm)
    gradient, = torch.autograd.grad(margin.sum(), x)
    if not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite geometry gradients')
    return gradient.detach(), margin.detach(), probability.detach()


def straighten(geometry, line, start, stop):
    """Replace one segment by its endpoint chord; retain endpoints and 32 points."""
    out = geometry.clone()
    t = torch.linspace(0, 1, stop - start + 1, device=out.device, dtype=out.dtype)[:, None]
    out[line, start:stop + 1] = geometry[line, start] * (1 - t) + geometry[line, stop] * t
    return out


def attribute_bundle(model, geometry, count, neighbors, norm, noise_seed,
                     smooth_samples=16, noise_sigma=.002, batch_size=32):
    """Return raw values, not display-normalized colors; score is a logit margin."""
    device = geometry.device
    counts = torch.tensor([count], device=device)
    g = geometry[None]
    ids = neighbors[None]
    gradient, margin, probability = point_gradients(model, g, counts, ids, norm)
    generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    noise = torch.randn((smooth_samples, *geometry.shape), generator=generator, device=device) * noise_sigma
    noise[:, count:] = 0
    noisy = g + noise
    sg, _, _ = point_gradients(model, noisy, counts.expand(smooth_samples),
                               ids.expand(smooth_samples, -1, -1), norm)
    # SmoothGrad averages gradient vectors before taking the spatial norm.
    smoothed = sg.mean(0).norm(dim=-1)
    patches = [(line, a, b) for line in range(count) for a, b in WINDOWS]
    patch_margin, patch_probability = [], []
    with torch.no_grad():
        for first in range(0, len(patches), batch_size):
            chosen = patches[first:first + batch_size]
            changed = torch.stack([straighten(geometry, *patch) for patch in chosen])
            m, p = evaluate(model, changed, counts.expand(len(chosen)), ids.expand(len(chosen), -1, -1), norm)
            patch_margin.extend(m.cpu().tolist())
            patch_probability.extend(p.cpu().tolist())
    delta = float(margin[0]) - np.asarray(patch_margin)
    delta_p = float(probability[0]) - np.asarray(patch_probability)
    point_delta = np.zeros((27, 32), np.float64)
    multiplicity = np.zeros_like(point_delta)
    for (line, start, stop), change in zip(patches, delta):
        point_delta[line, start + 1:stop] += change
        multiplicity[line, start + 1:stop] += 1
    point_delta /= np.maximum(multiplicity, 1)
    result = dict(gradient=gradient[0].norm(dim=-1).cpu().numpy(),
                  smooth_gradient=smoothed.cpu().numpy(), local_shape_delta=point_delta,
                  patch_indices=np.asarray(patches, np.int64), patch_margin_delta=delta,
                  patch_probability_delta=delta_p, patch_probability=np.asarray(patch_probability),
                  probability=float(probability[0]), margin=float(margin[0]))
    for key in ('gradient', 'smooth_gradient', 'local_shape_delta'):
        assert np.isfinite(result[key]).all() and np.all(result[key][count:] == 0)
    return result
