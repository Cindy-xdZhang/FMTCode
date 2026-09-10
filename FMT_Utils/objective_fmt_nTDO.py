"""Independent objective_fmt_nTDO (No timeDirectionOperation), version 1.1.

Only same-time relative vectors and all same-time pair distances enter the
temporal Fourier transform. No temporal differencing, initial-frame subtraction,
center-motion, kinematic, or IVD feature is computed.

The name is the requested method identifier, not an invariance certificate:
the inputs are objective vectors / scalars respectively. The distance Fourier
block is invariant under arbitrary common time-dependent rigid observers.
Vector Fourier descriptors are invariant under constant proper rotations;
arbitrary time-dependent rotation invariance is not guaranteed for that block.
"""
from __future__ import annotations

from numbers import Integral

import torch

VERSION = '1.1'


def _coordinates(pathlines):
    x = torch.as_tensor(pathlines)
    if x.ndim != 4 or x.shape[1] != 7 or x.shape[-1] not in (3, 4):
        raise ValueError('Expected [N,7,L,3 or 4], with center at line 0.')
    if x.shape[0] < 1 or x.shape[2] < 2:
        raise ValueError('Require at least one primitive and two time samples.')
    if x.is_complex() or x.dtype == torch.bool:
        raise ValueError('Coordinates must be real numbers.')
    if x.dtype not in (torch.float32, torch.float64):
        x = x.to(torch.float64)
    xyz = x[..., :3]
    if not torch.isfinite(xyz).all():
        raise ValueError('Coordinates must be finite; samples are not filtered.')
    return xyz


def build_fourier_inputs(pathlines):
    """Return inspectable Torch arrays without any operation between times.

    relative: [N,6,L,3], neighbors minus center at the SAME time.
    distances: [N,21,L], ||relative_i(t)-relative_j(t)||, with relative_0=0.
    pair_indices: [2,21], lexicographic pairs 0<=i<j<=6, including the center.
    Material point identities and time samples must correspond across observers.
    """
    xyz = _coordinates(pathlines)
    relative = xyz[:, 1:] - xyz[:, :1]
    points = torch.cat((torch.zeros_like(relative[:, :1]), relative), dim=1)
    pairs = torch.triu_indices(7, 7, offset=1, device=xyz.device)
    distances = torch.linalg.vector_norm(points[:, pairs[1]] - points[:, pairs[0]], dim=-1)
    return {'relative': relative, 'distances': distances, 'pair_indices': pairs}


def _vector_descriptors(spectrum):
    """Frozen-style frequency invariants, implemented without the old encoder.

    Spectrum [N,6,F,3] -> [N,6,4F-1]. Within each line, first interleave
    real norm / imaginary norm / cosine per bin; then append triple products.
    Sorting each slot across six neighbors matches the old pooling operation.
    All operations here are on Fourier coefficients, not on time differences.
    """
    real, imag = spectrum.real, spectrum.imag
    rn = torch.linalg.vector_norm(real, dim=-1)
    im = torch.linalg.vector_norm(imag, dim=-1)
    cosine = (real * imag).sum(dim=-1) / (rn * im).clamp_min(1e-8)
    descriptors = torch.stack((rn, im, cosine), dim=-1).flatten(2)
    triple = (torch.cross(real[:, :, :-1], imag[:, :, :-1], dim=-1)
              * real[:, :, 1:]).sum(dim=-1)
    denominator = (rn[:, :, :-1] * im[:, :, :-1] * rn[:, :, 1:]).clamp_min(1e-8)
    descriptors = torch.cat((descriptors, triple / denominator), dim=-1)
    return descriptors.sort(dim=1, descending=True).values.flatten(1)


def objective_fmt_nTDO(pathlines, num_freq=6, *, return_blocks=False, return_numpy=True):
    """Encode seven-line primitives using only the two declared Fourier inputs.

    For F bins, output width = 6*(4F-1) + 21*(2F-1); F=6 gives 138+231=369.
    Default output is a NumPy [N,D] matrix. With return_blocks=True, return a
    dictionary containing features, relative_fourier, distance_fourier.
    With return_numpy=False, preserve Torch device, precision and autograd.

    Distances are unsquared Euclidean distances. No scale normalization,
    temporal centering, baseline subtraction, temporal differences, or extra
    feature blocks are inserted. DC values are retained. Frequencies are in
    cycles per uniformly sampled window, not physical hertz.
    """
    inputs = build_fourier_inputs(pathlines)
    relative, distances = inputs['relative'], inputs['distances']
    if (isinstance(num_freq, bool) or not isinstance(num_freq, Integral)
            or not 2 <= num_freq <= relative.shape[2] // 2 + 1):
        raise ValueError('num_freq must be an integer in [2, L//2+1].')
    # These are the ONLY two inputs to rFFT. Their time dimension is L, not L-1.
    relative_spectrum = torch.fft.rfft(relative, dim=2)[:, :, :num_freq, :]
    distance_spectrum = torch.fft.rfft(distances, dim=2)[:, :, :num_freq]
    vectors = _vector_descriptors(relative_spectrum)
    scalars = torch.cat((distance_spectrum.real.flatten(1),
                         distance_spectrum.imag[:, :, 1:].flatten(1)), dim=1)
    combined = torch.cat((vectors, scalars), dim=1)
    if not torch.isfinite(combined).all():
        raise FloatingPointError('Nonfinite Fourier features.')
    blocks = {'features': combined, 'relative_fourier': vectors, 'distance_fourier': scalars}
    if return_numpy:
        blocks = {key: value.detach().cpu().numpy() for key, value in blocks.items()}
    return blocks if return_blocks else blocks['features']


__all__ = ['VERSION', 'build_fourier_inputs', 'objective_fmt_nTDO']
