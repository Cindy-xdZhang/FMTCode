"""Frozen-bundle resampling, train-only augmentation, and Fourier classifiers."""
from __future__ import annotations

import math
import torch
from torch import nn
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_P35_NormFrequency_3_1 import direction_spectrum
from FMT_Utils.FMT_V8_Search_2_1 import pooling_candidates, pool_neighbors, task4_model
from FMT_Utils.Task4C_NeighborSelection_1_1 import neighbor_indices

POOLS = {p['id']: p for p in pooling_candidates()}
ARCHITECTURES = ('original', 'wide_residual', 'attention_pool', 'set_attention',
                 'fps_graph', 'block_branches')
AUGMENTATIONS = ('none', 'jitter', 'rotate15', 'rotate_so3',
                 'jitter_rotate15', 'jitter_rotate_so3')


def line_mask(counts, lines):
    return torch.arange(lines, device=counts.device)[None] < counts[:, None]


def _interpolate(points, cumulative, query):
    """Linear interpolation on each polyline; all three arrays share leading axes."""
    ids = torch.searchsorted(cumulative.contiguous(), query.contiguous(), right=True)
    ids = ids.clamp(1, points.shape[-2]-1)
    lo = cumulative.gather(-1, ids-1)
    hi = cumulative.gather(-1, ids)
    fraction = ((query-lo)/(hi-lo).clamp_min(1e-15)).clamp(0, 1)
    left = points.gather(-2, (ids-1)[..., None].expand(*ids.shape, 3))
    right = points.gather(-2, ids[..., None].expand(*ids.shape, 3))
    return left+(right-left)*fraction[..., None]


def arc_coordinates(x):
    length = x.diff(dim=-2).norm(dim=-1)
    return torch.cat((torch.zeros_like(length[..., :1]), length.cumsum(-1)), -1)


@torch.no_grad()
def resample(x, points=32, sampling='uniform'):
    """Resample the frozen polyline, never reintegrate or add spatial detail."""
    if points not in (32, 48) or sampling not in ('uniform', 'curvature'):
        raise ValueError((points, sampling))
    if points == x.shape[-2] and sampling == 'uniform':
        return x.clone()  # Preserve the original 32-point reference exactly.
    work = x.double()
    arc = arc_coordinates(work)
    ds = arc.diff(dim=-1)
    if sampling == 'curvature':
        tangent = work.diff(dim=-2)/ds[..., None].clamp_min(1e-15)
        rate = (tangent[..., 1:, :]-tangent[..., :-1, :]).norm(dim=-1)
        rate = rate/(.5*(ds[..., 1:]+ds[..., :-1])).clamp_min(1e-15)
        vertex = torch.cat((rate[..., :1], rate, rate[..., -1:]), -1)
        curvature = .5*(vertex[..., :-1]+vertex[..., 1:])
        # Half uniform arclength mass retains low-curvature legs. The other
        # half is capped curvature mass, avoiding a single corner taking all samples.
        weighted_mean = (curvature*ds).sum(-1, keepdim=True)/ds.sum(-1, keepdim=True).clamp_min(1e-15)
        capped = torch.minimum(curvature, 4*weighted_mean)
        uniform = ds/ds.sum(-1, keepdim=True).clamp_min(1e-15)
        curved = capped*ds
        curved = torch.where(curved.sum(-1, keepdim=True)>1e-15,
                             curved/curved.sum(-1, keepdim=True).clamp_min(1e-15), uniform)
        mass = .5*uniform+.5*curved
        cumulative = torch.cat((torch.zeros_like(mass[..., :1]), mass.cumsum(-1)), -1)
    else:
        cumulative = arc
    query = cumulative[..., -1:]*torch.linspace(0, 1, points, device=x.device, dtype=work.dtype)
    out = _interpolate(work, cumulative, query)
    out[..., 0, :] = work[..., 0, :]
    out[..., -1, :] = work[..., -1, :]
    return out.to(x.dtype)


def rotation_matrices(batch, mode, generator, device):
    if mode == 'rotate_so3':
        q = torch.randn(batch, 4, generator=generator, device=device, dtype=torch.float64)
        q = q/q.norm(dim=-1, keepdim=True)
    elif mode == 'rotate15':
        axis = torch.randn(batch, 3, generator=generator, device=device, dtype=torch.float64)
        axis = axis/axis.norm(dim=-1, keepdim=True)
        angle = (2*torch.rand(batch, 1, generator=generator, device=device, dtype=torch.float64)-1)*math.pi/12
        q = torch.cat((torch.cos(angle/2), axis*torch.sin(angle/2)), -1)
    else:
        raise ValueError(mode)
    w, a, b, c = q.unbind(-1)
    return torch.stack((1-2*(b*b+c*c), 2*(a*b-c*w), 2*(a*c+b*w),
                        2*(a*b+c*w), 1-2*(a*a+c*c), 2*(b*c-a*w),
                        2*(a*c-b*w), 2*(b*c+a*w), 1-2*(a*a+b*b)), -1).reshape(batch, 3, 3)


@torch.no_grad()
def augment(x, counts, mode, generator, jitter_fraction=.15):
    if mode not in AUGMENTATIONS:
        raise ValueError(mode)
    result = x.clone()
    if 'jitter' in mode:
        # Slide each interior point along its polyline, by at most 15% of the
        # shorter adjacent interval. Order cannot reverse; endpoints never move.
        work = result.double()
        arc = arc_coordinates(work)
        ds = arc.diff(dim=-1)
        bound = jitter_fraction*torch.minimum(ds[..., :-1], ds[..., 1:])
        noise = 2*torch.rand(bound.shape, device=x.device, generator=generator, dtype=work.dtype)-1
        query = arc.clone()
        query[..., 1:-1] += noise*bound
        result = _interpolate(work, arc, query).to(x.dtype)
        result[..., 0, :] = x[..., 0, :]
        result[..., -1, :] = x[..., -1, :]
    if 'rotate' in mode:
        kind = 'rotate_so3' if 'so3' in mode else 'rotate15'
        rotation = rotation_matrices(len(x), kind, generator, x.device)
        result = torch.einsum('bij,bntj->bnti', rotation, result.double()).to(x.dtype)
    return result*line_mask(counts, x.shape[1])[..., None, None]


@torch.no_grad()
def normalize(x, counts):
    work = x.double()
    mask = line_mask(counts, x.shape[1])
    work = work*mask[..., None, None]
    center = work.sum((1, 2), keepdim=True)/(counts[:, None, None, None]*x.shape[2])
    work = (work-center)*mask[..., None, None]
    radius = work.square().sum(-1).amax((1, 2)).sqrt()
    if not torch.all(radius > 1e-12):
        raise ValueError('Degenerate bundle')
    return (work/radius[:, None, None, None]).float(), mask


@torch.no_grad()
def fourier_tokens(geometry, counts, neighbors, pool, chunk=1024):
    """Six true DFT bins, original center/direction and selected neighbor recipe."""
    definition = POOLS[pool]
    x, mask = normalize(geometry, counts)
    center = torch.arange(x.shape[1], device=x.device)[None, :, None].expand(len(x), -1, -1)
    ids = torch.cat((center, neighbors), -1)
    bi, li = mask.nonzero(as_tuple=True)
    out = x.new_zeros((len(x), x.shape[1], definition['feature_dimensions']+1))
    for first in range(0, len(bi), chunk):
        bs, ls = bi[first:first+chunk], li[first:first+chunk]
        primitive = x[bs[:, None], ids[bs, ls]]
        base = pathline_dft_features_3d(primitive, num_freq=6, neighbor_scale=1.,
                                      neighbor_weight=1., neighbor_pool='sort',
                                      mode='gram', include_chirality=True, return_numpy=False)
        values = torch.cat((base[:, :23], direction_spectrum(primitive[:, 0], 6),
                            pool_neighbors(base[:, 23:], definition)), -1)
        out[bs, ls, :-1] = values
    out[..., -1] = mask
    if not torch.isfinite(out).all():
        raise ValueError('Nonfinite Fourier features')
    return out


def pooled(x, mask):
    mean = (x*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
    maximum = x.masked_fill(~mask[..., None], -torch.inf).amax(1)
    return torch.cat((mean, maximum), -1)


def mlp(cin, cout, dropout=.15):
    return nn.Sequential(nn.Linear(cin, cout), nn.LayerNorm(cout), nn.GELU(), nn.Dropout(dropout))


class Residual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width*2, width), nn.Dropout(dropout))

    def forward(self, x):
        return x+self.layers(x)


class AttentionBlock(nn.Module):
    """Explicit attention avoids nondeterministic fused GPU kernels."""
    def __init__(self, width=192, heads=6, dropout=.15):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.output = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.ffn = Residual(width, dropout)

    def forward(self, x, mask):
        b, n, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, w//self.heads).permute(2, 0, 3, 1, 4)
        score = (q@k.transpose(-1, -2))/math.sqrt(w//self.heads)
        weight = score.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
        update = (self.drop(weight)@v).transpose(1, 2).reshape(b, n, w)
        return self.ffn(x+self.drop(self.output(update)))*mask[..., None]


class FourierClassifier(nn.Module):
    def __init__(self, pool='p35', architecture='original', profile='h0'):
        super().__init__()
        self.architecture = architecture
        width = POOLS[pool]['feature_dimensions']
        if architecture == 'original':
            self.network = task4_model(width, profile)
            return
        if pool != 'p35' or profile != 'h0':
            raise ValueError('New architecture comparison fixes the p35 Fourier representation')
        if architecture == 'wide_residual':
            self.line = nn.Sequential(mlp(width, 256), Residual(256, .15), Residual(256, .15), Residual(256, .15))
            output_width = 512
        elif architecture in ('attention_pool', 'set_attention'):
            self.line = nn.Sequential(mlp(width, 192), Residual(192, .15))
            if architecture == 'attention_pool':
                self.scores = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 4))
                output_width = 192*6  # Four learned pools plus mean/max.
            else:
                self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
                output_width = 384
        elif architecture == 'fps_graph':
            self.line = mlp(width, 128)
            self.edges = nn.ModuleList([nn.Sequential(mlp(256, 128), mlp(128, 128)) for _ in range(2)])
            output_width = 256
        elif architecture == 'block_branches':
            self.branches = nn.ModuleList([nn.Sequential(mlp(d, 96), Residual(96, .15)) for d in (23, 72, 46)])
            self.line = nn.Sequential(mlp(288, 256), Residual(256, .15))
            output_width = 512
        else:
            raise ValueError(architecture)
        self.head = nn.Sequential(mlp(output_width, 256), mlp(256, 128), nn.Linear(128, 2))

    def forward(self, tokens, neighbors=None):
        if self.architecture == 'original':
            return self.network(tokens)
        mask = tokens[..., -1] > .5
        x = tokens[..., :-1]
        if self.architecture == 'block_branches':
            x = torch.cat([branch(block) for branch, block in zip(self.branches, x.split((23, 72, 46), -1))], -1)
        x = self.line(x)*mask[..., None]
        if self.architecture == 'set_attention':
            for block in self.blocks:
                x = block(x, mask)
        elif self.architecture == 'fps_graph':
            if neighbors is None:
                raise ValueError('FPS graph requires frozen neighbor indices')
            batch = torch.arange(len(x), device=x.device)[:, None, None]
            for edge in self.edges:
                other = x[batch, neighbors]
                anchor = x[:, :, None, :].expand_as(other)
                x = (x+edge(torch.cat((anchor, other-anchor), -1)).amax(-2))*mask[..., None]
        summary = pooled(x, mask)
        if self.architecture == 'attention_pool':
            weights = self.scores(x).masked_fill(~mask[..., None], -torch.inf).softmax(1)
            attended = torch.einsum('bnq,bnw->bqw', weights, x).flatten(1)
            summary = torch.cat((summary, attended), -1)
        return self.head(summary)


def parameter_count(pool, architecture, profile='h0'):
    return sum(p.numel() for p in FourierClassifier(pool, architecture, profile).parameters())
