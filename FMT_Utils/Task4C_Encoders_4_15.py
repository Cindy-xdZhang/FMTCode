"""Pure seven-line encoders with the frozen Task4-c local-neighbor rule."""
from __future__ import annotations

import torch
from torch import nn

from FMT_Utils.fmt_v5 import fmt_v5
from FMT_Utils.objective_fmt_nTDO_v2 import objective_fmt_nTDO_v2
from FMT_Utils.fmt_objective_ntod_v2 import fmt_objective_ntod_v2
from FMT_Utils.Task4C_LinePooling_4_3 import LearnedLinePooling, WiderConv3D

ENCODERS = {
    'fmt_v5': (fmt_v5, 326, '5.1'),
    'objective_fmt_nTDO_v2': (objective_fmt_nTDO_v2, 231, '2.1'),
    'fmt_objective_ntod_v2': (fmt_objective_ntod_v2, 396, '2.2'),
}


def local_indices(geometry, seeds, counts):
    """Use each valid line as center and its six closest valid seed neighbors."""
    b, n, p, d = geometry.shape
    if (p, d) != (32, 3) or seeds.shape != (b, n, 3) or counts.shape != (b,):
        raise ValueError('Expected aligned 32-point bundles, seeds and counts')
    if not torch.isfinite(geometry).all() or not torch.isfinite(seeds).all():
        raise ValueError('Nonfinite geometry or seed coordinates')
    if torch.any(counts < 10) or torch.any(counts > n) or torch.any(counts != counts.long()):
        raise ValueError('Invalid valid-line counts')
    mask = torch.arange(n, device=geometry.device)[None] < counts[:, None]
    distance = torch.cdist(seeds, seeds)
    distance.masked_fill_(~mask[:, None, :], torch.inf)
    distance.diagonal(dim1=1, dim2=2).fill_(torch.inf)
    nearest = torch.argsort(distance, dim=-1, stable=True)[..., :6]
    central = torch.arange(n, device=geometry.device)[None, :, None].expand(b, -1, -1)
    return torch.cat((central, nearest), -1), mask


@torch.no_grad()
def encode_lines(geometry, seeds, counts, encoder):
    """Encoder output plus a mask only; no signed-direction or kinematic block."""
    function, width, _ = ENCODERS[encoder]
    ids, mask = local_indices(geometry, seeds, counts)
    b, n = geometry.shape[:2]
    result = geometry.new_zeros((b, n, width + 1))
    bi, li = mask.nonzero(as_tuple=True)
    for start in range(0, len(bi), 512):
        bs, ls = bi[start:start+512], li[start:start+512]
        primitive = geometry[bs[:, None], ids[bs, ls]]
        result[bs, ls, :-1] = function(primitive, num_freq=6, return_numpy=False)
    result[..., -1] = mask.to(result.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite encoded tokens; no samples may be filtered')
    return result


class EncoderLinePooling(LearnedLinePooling):
    """Frozen 4.3 pooling topology, with only the input width adapted."""
    def __init__(self, width, dropout):
        super().__init__(dropout)
        self.line[0] = nn.Linear(width, 128)


def make_model(encoder, dropout):
    if encoder == 'conv3d':
        return WiderConv3D(dropout)
    return EncoderLinePooling(ENCODERS[encoder][1], dropout)
