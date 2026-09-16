"""Only change the six neighbor lines of each frozen p35 Task4-c token."""
from __future__ import annotations
import torch
from FMT_Utils.FMT_P35_NormFrequency_3_1 import normalize_geometry, primitive_features

STRATEGIES = ('nearest6', 'fps6', 'nearest3_fps3')


@torch.no_grad()
def neighbor_indices(seeds, counts, strategy):
    """FPS starts with the anchor; subsequent picks maximize distance to the selected set."""
    if strategy not in STRATEGIES: raise ValueError(strategy)
    b, n, _ = seeds.shape
    valid = torch.arange(n, device=seeds.device)[None] < counts[:, None]
    if torch.any(counts < 7) or torch.any(counts > n): raise ValueError('Need at least seven valid lines')
    distances = torch.cdist(seeds, seeds)
    excluded = (~valid[:, None, :]).expand(b, n, n).clone()
    excluded.diagonal(dim1=1, dim2=2).fill_(True)
    nearest = torch.argsort(distances.masked_fill(excluded, torch.inf), dim=-1, stable=True)[..., :6]
    if strategy == 'nearest6': return nearest
    # The original cached line index resolves equal-distance ties deterministically.
    chosen = excluded.clone()
    minimum = distances.clone()
    selected = []
    batch = torch.arange(b, device=seeds.device)[:, None]
    if strategy == 'nearest3_fps3':
        for j in range(3):
            pick = nearest[..., j]
            selected.append(pick)
            chosen.scatter_(-1, pick[..., None], True)
            minimum = torch.minimum(minimum, distances[batch, pick])
    for _ in range(6-len(selected)):
        pick = minimum.masked_fill(chosen, -torch.inf).max(-1).indices
        selected.append(pick)
        chosen.scatter_(-1, pick[..., None], True)
        minimum = torch.minimum(minimum, distances[batch, pick])
    return torch.stack(selected, -1)


@torch.no_grad()
def encode(geometry, seeds, counts, definition, strategy):
    x, mask = normalize_geometry(geometry, counts, definition['geometry'])
    neighbors = neighbor_indices(seeds, counts, strategy)
    center = torch.arange(x.shape[1], device=x.device)[None, :, None].expand(len(x), -1, -1)
    ids = torch.cat((center, neighbors), -1)
    bi, li = mask.nonzero(as_tuple=True)
    result = x.new_zeros((len(x), x.shape[1], definition['feature_dimensions']))
    for start in range(0, len(bi), 1024):
        bs, ls = bi[start:start+1024], li[start:start+1024]
        primitive = x[bs[:, None], ids[bs, ls]]
        result[bs, ls] = primitive_features(primitive, definition['frequencies'])
    return result


@torch.no_grad()
def selection_statistics(seeds, counts, indices):
    """Distance and coverage in the shared normalized seed coordinates; no labels."""
    b, n, _ = seeds.shape
    valid = torch.arange(n, device=seeds.device)[None] < counts[:, None]
    distance = torch.cdist(seeds.double(), seeds.double())
    batch = torch.arange(b, device=seeds.device)[:, None]
    minimum = distance.clone()
    for j in range(6): minimum = torch.minimum(minimum, distance[batch, indices[..., j]])
    selected = distance.gather(-1, indices)
    diameter = distance.masked_fill(~valid[:, None, :], 0).max(-1).values.clamp_min(1e-12)
    covered_radius = minimum.masked_fill(~valid[:, None, :], 0).max(-1).values
    return dict(anchors=int(valid.sum()), selected_distance_sum=float(selected[valid].sum()),
                farthest_selected_ratio_sum=float((selected.max(-1).values/diameter)[valid].sum()),
                coverage_radius_ratio_sum=float((covered_radius/diameter)[valid].sum()))
