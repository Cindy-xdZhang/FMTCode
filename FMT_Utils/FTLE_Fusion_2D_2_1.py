"""Matched low-grid residual decoders with optional Fourier geometry."""

from FMT_Utils.FMTNoConvolution_1_1 import reject_retired_fmt, assert_no_fmt_convolution
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Removed: convolutional FMT is prohibited by project policy."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        reject_retired_fmt("ResidualBlock")


class FusionSR(nn.Module):
    """Removed: convolutional FMT is prohibited by project policy."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        reject_retired_fmt("FusionSR")
