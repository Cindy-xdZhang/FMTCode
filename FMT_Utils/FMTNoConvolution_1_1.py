"""User-mandated FMT architecture policy, effective 2026-09-17.

The complete FMT prediction path must be convolution-free, including frozen
Raw branches. Independent, explicitly named non-FMT baselines are separate.
"""
from torch import nn

POLICY = 'docs/FMT_no_spatial_convolution_protocol_1.1.md'


class ForbiddenFMTConvolutionError(ValueError):
    """A retired convolutional FMT recipe was requested."""


def reject_retired_fmt(name):
    raise ForbiddenFMTConvolutionError(
        f'{name} was removed by the user on 2026-09-17: FMT must not use '
        f'Conv1d/2d/3d, transposed/spectral convolution, EdgeConv-style spatial '
        f'message passing, or a convolutional Raw branch. '
        f'Historical results are withdrawn from active FMT evidence. See {POLICY}'
    )


def assert_no_fmt_convolution(model):
    """Inspect all registered branches, even frozen or unused submodules.

    Functional/custom convolution is also prohibited by the protocol and must
    be checked in source review; a module inventory alone cannot detect it.
    """
    forbidden = [(name, type(layer).__name__) for name, layer in model.named_modules()
                 if isinstance(layer, nn.modules.conv._ConvNd)]
    if forbidden:
        reject_retired_fmt(f'{type(model).__name__}: {forbidden}')
    return model
