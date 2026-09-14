"""Versioned 233D and 95D encoders for normalized Task4-c vortex-line bundles."""
from __future__ import annotations

import torch

from FMT_Utils.Task4C_LinePooling_4_3 import line_fmt, LearnedLinePooling
from FMT_Utils.Task4C_Encoders_4_15 import EncoderLinePooling

WIDTHS = {'fmt_V6': 233, 'fmt_V7': 95}
VERSIONS = {'fmt_V6': '6.1', 'fmt_V7': '7.1'}
BLOCKS = {
    'fmt_V6': ['center_step_23', 'neighbor_change_138', 'coordinate_tangent_72'],
    'fmt_V7': ['center_step_23', 'coordinate_tangent_72'],
}


def select_cached(original, encoder):
    """Keep the original coefficient order and append the unchanged valid-line mask."""
    if original.shape[-1] != 234:
        raise ValueError('Expected frozen 4.14 coefficients (233) plus mask (1)')
    if encoder == 'fmt_V6':
        return original
    if encoder == 'fmt_V7':
        return torch.cat((original[..., :23], original[..., 161:234]), -1)
    raise ValueError('Unknown encoder: ' + encoder)


@torch.no_grad()
def fmt_V6(geometry, seeds, counts):
    """Input: bundle-normalized [B,N,32,3], seeds [B,N,3], valid counts [B].

    Output: [B,N,234], with 233 features and a mask. This is exactly the
    frozen 4.14 encoder; preprocessing and nearest-neighbor rules are unchanged.
    """
    return line_fmt(geometry, seeds, counts)


@torch.no_grad()
def fmt_V7(geometry, seeds, counts):
    """Output [B,N,96]: center-step 23, coordinate/tangent 72, then mask.

    Reuse the frozen geometric computation and discard all 138 neighbor
    coefficients. Experiments select cached columns to avoid this recomputation.
    Bundle centering/scaling still uses all valid lines, as in 4.14.
    """
    return select_cached(line_fmt(geometry, seeds, counts), 'fmt_V7')


def make_model(encoder, dropout):
    if encoder == 'fmt_V6':
        return LearnedLinePooling(dropout)
    if encoder == 'fmt_V7':
        return EncoderLinePooling(95, dropout)
    raise ValueError('Unknown encoder: ' + encoder)
