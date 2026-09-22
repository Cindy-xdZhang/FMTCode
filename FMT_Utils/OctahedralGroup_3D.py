"""Octahedral group action on the 7-line 3D pathline primitive.

This is the 3D counterpart of the 2D ``D8`` action in
``PyflowVis/FMT_Utils/siren_vae.py``.  In 2D the eight neighbours sit on a
circle, so a 45-degree rotation is a cyclic channel shift.  In 3D the six
neighbours are ``+-e_x, +-e_y, +-e_z`` (``FMT_Utils/FMT_3D_pipeline.py``), whose
symmetry group is the full octahedral group ``O_h`` of order 48.

The structural fact that makes this cheap:

    ``O_h`` is exactly the set of 3x3 signed permutation matrices
    (3! axis permutations x 2^3 sign choices = 48; the 24 with det = +1
    form the proper rotation subgroup ``O``).

So every group element acts on a vector by permuting its three components and
flipping signs -- exact in floating point, no interpolation, no trigonometry --
and acts on the neighbour channels by permuting them, because a signed
permutation maps the six axis directions onto themselves.

Line order follows the integrator that produced the caches
(``FMT_3D_pipeline.integrate_cross_primitives_3d``)::

    0 = centre, 1 = x+, 2 = x-, 3 = y+, 4 = y-, 5 = z+, 6 = z-

so neighbour channel ``k`` (0-based, i.e. line ``k+1``) carries direction
``sign * e_axis`` with ``k = 2 * axis + (0 if sign > 0 else 1)``.

Label legitimacy: the Task1 label is ``IVD = ||omega - <omega>||``
(``FLowUtils/ScalarField3d.py``).  Vorticity is a pseudovector, so under any
``M`` in ``O(3)`` both ``omega`` and its spatial mean pick up the same
``det(M) M`` factor and the norm is unchanged.  IVD is therefore invariant under
the full ``O(3)``, reflections included, and all 48 elements are label-exact
augmentations.  See ``docs/octahedral_equivariance_3d.md``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch


# Direction of each of the six neighbour channels, in cache line order.
NEIGHBOUR_DIRECTIONS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)
LINE_COUNT = 7
NEIGHBOUR_COUNT = 6
GROUPS = ("oh", "o")


def _direction_channel(direction):
    """Map a signed axis direction to its neighbour channel index."""
    nonzero = np.nonzero(direction)[0]
    if len(nonzero) != 1 or abs(direction[nonzero[0]]) != 1:
        raise ValueError(f"not a signed axis direction: {direction!r}")
    axis = int(nonzero[0])
    return 2 * axis + (0 if direction[axis] > 0 else 1)


def octahedral_group(group="oh"):
    """Return ``(matrices, permutation)`` for the requested octahedral group.

    ``matrices``    : ``[G, 3, 3]`` float32, the signed permutation matrices.
    ``permutation`` : ``[G, 6]`` int64 gather index on the neighbour channels.

    The gather convention is the one the action needs::

        augmented[j] = M @ original[permutation[j]]

    which requires ``direction(permutation[j]) = M^T direction(j)``; ``M`` is
    orthogonal so ``M^-1 = M^T``.  Element 0 is always the identity, so
    ``exclude_identity`` sampling can simply skip index 0.
    """
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    matrices = []
    for axis_permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int64)
            for axis in range(3):
                matrix[axis_permutation[axis], axis] = signs[axis]
            if group == "o" and round(float(np.linalg.det(matrix))) != 1:
                continue
            matrices.append(matrix)
    matrices = np.stack(matrices)
    expected = 48 if group == "oh" else 24
    if len(matrices) != expected:
        raise RuntimeError(f"expected {expected} elements, built {len(matrices)}")
    if not np.array_equal(matrices[0], np.eye(3, dtype=np.int64)):
        raise RuntimeError("element 0 must be the identity")

    permutation = np.empty((len(matrices), NEIGHBOUR_COUNT), dtype=np.int64)
    for index, matrix in enumerate(matrices):
        for channel in range(NEIGHBOUR_COUNT):
            source = matrix.T @ NEIGHBOUR_DIRECTIONS[channel]
            permutation[index, channel] = _direction_channel(source)
    return matrices.astype(np.float32), permutation


def group_tensors(group="oh", device=None, dtype=torch.float32):
    """``octahedral_group`` as torch tensors on ``device``."""
    matrices, permutation = octahedral_group(group)
    return (torch.as_tensor(matrices, dtype=dtype, device=device),
            torch.as_tensor(permutation, dtype=torch.long, device=device))


def apply_group(signal, element, matrices, permutation):
    """Apply one group element per batch row to ``signal`` ``[B, 7, T, 3]``.

    ``element`` is a long tensor ``[B]`` of group indices.  The centre channel
    keeps its slot and only rotates; the six neighbour channels rotate and
    permute.
    """
    if signal.dim() != 4 or signal.shape[1] != LINE_COUNT or signal.shape[-1] != 3:
        raise ValueError(f"signal must be [B,{LINE_COUNT},T,3], got {tuple(signal.shape)}")
    batch, _, steps, _ = signal.shape
    if element.shape != (batch,):
        raise ValueError(f"element must be [{batch}], got {tuple(element.shape)}")

    rotation = matrices.to(signal.dtype)[element]                 # [B, 3, 3]
    rotated = torch.einsum("bij,bktj->bkti", rotation, signal)    # [B, 7, T, 3]

    gather = permutation[element]                                  # [B, 6]
    index = gather.view(batch, NEIGHBOUR_COUNT, 1, 1).expand(-1, -1, steps, 3)
    neighbours = torch.gather(rotated[:, 1:], dim=1, index=index)
    return torch.cat((rotated[:, :1], neighbours), dim=1)


def random_group(signal, matrices, permutation, generator=None,
                 exclude_identity=True):
    """Per-sample random group element.  Returns ``(augmented, element)``."""
    batch = signal.shape[0]
    low = 1 if exclude_identity else 0
    element = torch.randint(low, len(matrices), (batch,), device=signal.device,
                            generator=generator)
    return apply_group(signal, element, matrices, permutation), element


def group_orbit(signal, matrices, permutation):
    """Yield the full orbit of ``signal`` one group element at a time.

    Generated lazily so that group pooling over 48 elements never materialises
    ``[48, B, 7, T, 3]`` at once.
    """
    batch = signal.shape[0]
    for index in range(len(matrices)):
        element = torch.full((batch,), index, dtype=torch.long, device=signal.device)
        yield apply_group(signal, element, matrices, permutation)


# ---------------------------------------------------------------------------
# Equivariant normalisation
# ---------------------------------------------------------------------------

def normalize_signal_oh(signal, clip_sigma=5.0):
    """Exactly ``O(3)``-equivariant normalisation of ``[N, 7, T, 3]``.

    Two deliberate differences from the 2D ``normalize_signal_d8``:

    1. **No mean subtraction.**  Dividing by a scalar commutes with a rotation,
       but subtracting a scalar ``c`` from a vector does not::

           normalize(M s) = (M s - c*1) / sigma
           M normalize(s) = (M s - c*M 1) / sigma      differ by c (M 1 - 1)/sigma

       In 2D that is a small approximation.  Here the flows carry a strong mean
       streamwise velocity, so ``c`` is far from zero, and the axis-permuting
       elements of ``O_h`` make ``M 1 != 1`` for most elements.  Dividing by a
       scalar RMS taken about zero is exactly equivariant.

    2. **Magnitude clipping instead of componentwise clipping.**  Clamping each
       component independently does not commute with a rotation.  Rescaling any
       3-vector whose norm exceeds the limit does, because ``||M v|| = ||v||``.
       ``clip_sigma`` stays a per-component sigma; since each component has unit
       RMS after scaling, the equivalent vector-magnitude limit is
       ``clip_sigma * sqrt(3)``.

    The scale is shared within each group that the channel permutation acts on
    -- one for the centre, one for all six neighbours -- so the normalisation
    also commutes with the channel permutation.

    Returns ``(signal_n, stats)`` with ``stats = (centre_rms, neighbour_rms)``
    fitted on this tensor; pass them to ``apply_norm_stats_oh`` for evaluation
    data so the statistics stay train-only.
    """
    if signal.dim() != 4 or signal.shape[1] != LINE_COUNT or signal.shape[-1] != 3:
        raise ValueError(f"signal must be [N,{LINE_COUNT},T,3], got {tuple(signal.shape)}")
    centre_rms = signal[:, :1].pow(2).mean().sqrt().clamp_min(1e-8)
    neighbour_rms = signal[:, 1:].pow(2).mean().sqrt().clamp_min(1e-8)
    stats = (centre_rms.detach(), neighbour_rms.detach())
    return apply_norm_stats_oh(signal, stats, clip_sigma), stats


def apply_norm_stats_oh(signal, stats, clip_sigma=5.0):
    """Apply frozen ``normalize_signal_oh`` statistics to a possibly augmented signal."""
    centre_rms, neighbour_rms = stats
    scaled = torch.cat((signal[:, :1] / centre_rms, signal[:, 1:] / neighbour_rms), dim=1)
    if clip_sigma is None or not math.isfinite(float(clip_sigma)) or clip_sigma <= 0:
        return scaled
    limit = float(clip_sigma) * math.sqrt(3.0)
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    return scaled * torch.clamp(limit / norm.clamp_min(1e-12), max=1.0)


def channel_balance_weights_oh(signal_n):
    """Equivariance-preserving reconstruction weights, ``[1, 7, 1, 1]``.

    Same rationale as the 2D ``channel_balance_weights``: after normalisation
    every channel has unit variance globally, but that variance splits very
    differently between "across samples" and "along time", so plain MSE is
    satisfied almost entirely by placing the centre at the right constant level.
    Weighting by 1 / within-sample variance restores the temporal shape.

    The weight must be constant within each group the channel permutation acts
    on -- one for the centre, one shared by all six neighbours -- otherwise the
    weighted loss is no longer invariant under the group.
    """
    within = signal_n.var(dim=2).mean(dim=(0, 2))            # [7]
    weights = torch.empty_like(within)
    weights[0] = 1.0 / within[0].clamp_min(1e-8)
    weights[1:] = 1.0 / within[1:].mean().clamp_min(1e-8)
    return (weights / weights.mean()).view(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Signal construction from the Task1 caches
# ---------------------------------------------------------------------------

def build_signal_from_raw(raw_features, sampled_steps=32):
    """Build the ``[N, 7, T-1, 3]`` group-covariant signal from cached features.

    ``Build_Task2_Universality_Cache._raw_local_features`` stores
    ``x_i(t) - x_0(0)`` flattened to ``7 * sampled_steps * 3``.  The 3D analogue
    of the 2D ``build_signal`` is::

        channel 0    : d/dt x_0(t)                     centre velocity
        channels 1-6 : d/dt ( x_i(t) - x_0(t) )        neighbour rate rel. centre

    The common anchor ``x_0(0)`` cancels in both, so no re-integration is
    needed.  Both quantities are translation-invariant and rotate as vectors,
    which is what the group action requires.

    The 2D ``x100`` neighbour factor is deliberately not carried over: it is a
    documented no-op under group-wise normalisation, which this pipeline always
    uses, so reproducing it would only add a trap.
    """
    values = np.asarray(raw_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != LINE_COUNT * int(sampled_steps) * 3:
        raise ValueError(
            f"expected [N,{LINE_COUNT * int(sampled_steps) * 3}] raw features, "
            f"got {tuple(values.shape)}"
        )
    paths = values.reshape(len(values), LINE_COUNT, int(sampled_steps), 3)
    centre = paths[:, :1]
    relative = paths[:, 1:] - centre
    signal = np.concatenate((np.diff(centre, axis=2), np.diff(relative, axis=2)), axis=1)
    if not np.isfinite(signal).all():
        raise ValueError("non-finite pathline signal")
    return np.ascontiguousarray(signal, dtype=np.float32)
