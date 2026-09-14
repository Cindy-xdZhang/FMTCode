"""Task4-c 4.14 extensions that retain its coordinate/tangent spectrum."""
from __future__ import annotations

import torch

from FMT_Utils.Task4C_Encoders_4_15 import local_indices, encode_lines as local_encode, EncoderLinePooling
from FMT_Utils.Task4C_LinePooling_4_3 import LearnedLinePooling

WIDTHS = {'fmt_4_14': 233, 'fmt_v5_4_14': 398, 'fmt_objective_ntod_v2_4_14': 468}
VERSIONS = {'fmt_4_14': '4.14', 'fmt_v5_4_14': '5.1', 'fmt_objective_ntod_v2_4_14': '2.2'}
BLOCKS = {
    'fmt_4_14': ['original_local_fmt_161', 'original_coordinate_tangent_spectrum_72'],
    'fmt_v5_4_14': ['original_local_fmt_161', 'original_coordinate_tangent_spectrum_72', 'center_angle_spectrum_165'],
    'fmt_objective_ntod_v2_4_14': ['point_pair_distance_spectrum_231', 'original_coordinate_tangent_spectrum_72', 'center_angle_spectrum_165'],
}


@torch.no_grad()
def encode_lines(geometry, seeds, counts, encoder, original):
    """Copy frozen 4.14 coefficients bit for bit; never recompute their values."""
    if original.shape != (*geometry.shape[:2], 234):
        raise ValueError('Expected the complete frozen 4.14 token including mask')
    _, mask = local_indices(geometry, seeds, counts)
    if not torch.equal(original[..., -1].bool(), mask):
        raise ValueError('Frozen cache and physical valid-line counts differ')
    if encoder == 'fmt_4_14':
        return original
    if encoder == 'fmt_v5_4_14':
        local = local_encode(geometry, seeds, counts, 'fmt_v5')
        # The appended angle block is exactly the existing fmt_v5 5.1 block.
        result = torch.cat((original[..., :233], local[..., 161:326], original[..., -1:]), -1)
    elif encoder == 'fmt_objective_ntod_v2_4_14':
        local = local_encode(geometry, seeds, counts, 'fmt_objective_ntod_v2')
        result = torch.cat((local[..., :231], original[..., 161:233], local[..., 231:396], original[..., -1:]), -1)
    else:
        raise ValueError('Unknown versioned encoder: ' + encoder)
    if result.shape[-1] != WIDTHS[encoder] + 1 or not torch.isfinite(result).all():
        raise ValueError('Invalid extended token')
    if torch.any(result[~mask] != 0):
        raise ValueError('Invalid nonzero padded line')
    return result


def make_model(encoder, dropout):
    if encoder == 'fmt_4_14':
        return LearnedLinePooling(dropout)
    # Same hidden layers and training rule. Only the first layer needs more inputs;
    # as in 4.15, its complete weight matrix receives PyTorch Linear initialization.
    return EncoderLinePooling(WIDTHS[encoder], dropout)
