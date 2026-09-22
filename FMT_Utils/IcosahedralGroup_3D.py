"""Icosahedral symmetry for 13-line 3D pathline primitives (centre + 12 vertices).

The 3D counterpart of the 2D ``D8`` star and a refinement of the octahedral
``O_h`` star used elsewhere in this project.  The neighbours sit at the twelve
vertices of a regular icosahedron, which is the largest vertex-transitive set
whose symmetry group acts on it by permutation, and the group is

    I    -- 60 proper rotations (isomorphic to A5)
    I_h  = I x {+E, -E} -- 120 elements once reflections are included

so a batch row can draw one of **120** discrete views.  Both facts are verified
numerically in ``tests/test_icosahedral_group_3d.py`` rather than asserted.

Why icosahedral rather than octahedral: the octahedral group covers SO(3) with a
mean nearest-element angle of 40.7 degrees, the icosahedral group with 29.5
degrees, so the discrete view set is nearly half as coarse for twice the
elements.  Twelve neighbours also sample the sphere more evenly than six -- the
icosahedron is the unique 12-point arrangement whose points are all equivalent
under its symmetry group.

Conventions match ``OctahedralGroup_3D``: the action on a primitive is

    augmented[0] = M @ original[0]                    (the centre line)
    augmented[j] = M @ original[permutation[j]]       (neighbour channels)

with ``permutation[j] = index of M^T d_j`` among the twelve directions, so the
rotated star is the star that would have been traced had the flow been rotated.
"""
from __future__ import annotations

import numpy as np
import torch

LINE_COUNT = 13
NEIGHBOUR_COUNT = 12
VECTOR_DIM = 3
GROUPS = ("ih", "i")
_TOLERANCE = 1e-9


def icosahedron_vertices():
    """Twelve unit vectors on the regular icosahedron, in a fixed order."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = []
    for sy in (+1, -1):
        for sz in (+1, -1):
            raw.append([0.0, sy * 1.0, sz * phi])
    for sz in (+1, -1):
        for sx in (+1, -1):
            raw.append([sx * phi, 0.0, sz * 1.0])
    for sx in (+1, -1):
        for sy in (+1, -1):
            raw.append([sx * 1.0, sy * phi, 0.0])
    vertices = np.asarray(raw, dtype=np.float64)
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * (cross @ cross))


def _close_group(generators, limit=256):
    """Multiplicative closure with numerical de-duplication."""
    elements = [np.eye(3)]

    def index_of(matrix):
        for position, existing in enumerate(elements):
            if np.abs(existing - matrix).max() < 1e-8:
                return position
        return -1

    frontier = [np.eye(3)]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = generator @ current
            if index_of(candidate) < 0:
                elements.append(candidate)
                frontier.append(candidate)
                if len(elements) > limit:
                    raise ValueError("Group closure exceeded the expected order")
    return np.asarray(elements)


def icosahedral_group(group="ih"):
    """``(matrices, permutation)`` for the icosahedral group.

    ``matrices``    ``[G, 3, 3]`` float64; ``G`` is 60 for ``i`` and 120 for ``ih``.
    ``permutation`` ``[G, 12]`` int64 gather index on the neighbour channels.
    """
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    vertices = icosahedron_vertices()
    anchor = vertices[0]
    # the five vertices nearest the anchor are its icosahedral neighbours
    cosine = vertices @ anchor
    neighbour = vertices[np.argsort(-cosine)[1]]
    rotations = _close_group([
        _rotation(anchor, 2.0 * np.pi / 5.0),                 # 5-fold, vertex axis
        _rotation(anchor + neighbour, np.pi),                 # 2-fold, edge midpoint
    ])
    if len(rotations) != 60:
        raise ValueError(f"Icosahedral rotation group must have 60 elements, got {len(rotations)}")
    matrices = rotations if group == "i" else np.concatenate((rotations, -rotations))

    permutation = np.empty((len(matrices), NEIGHBOUR_COUNT), dtype=np.int64)
    for index, matrix in enumerate(matrices):
        pulled = vertices @ matrix                            # rows are M^T d_j
        similarity = pulled @ vertices.T
        source = similarity.argmax(axis=1)
        if np.abs(similarity[np.arange(NEIGHBOUR_COUNT), source] - 1.0).max() > 1e-9:
            raise ValueError("A group element does not permute the icosahedron vertices")
        if len(np.unique(source)) != NEIGHBOUR_COUNT:
            raise ValueError("A group element gave a non-bijective channel map")
        permutation[index] = source
    return matrices, permutation


def group_tensors(group="ih", device=None, dtype=torch.float32):
    matrices, permutation = icosahedral_group(group)
    return (torch.as_tensor(matrices, dtype=dtype, device=device),
            torch.as_tensor(permutation, dtype=torch.long, device=device))


def apply_group(signal, element, matrices, permutation):
    """``signal`` ``[N, 13, T, 3]`` -> the same primitive under group ``element``.

    ``element`` is ``[N]`` long, so every row may take a different view.
    """
    if signal.shape[1] != LINE_COUNT or signal.shape[-1] != VECTOR_DIM:
        raise ValueError(f"signal must be [N,{LINE_COUNT},T,3], got {tuple(signal.shape)}")
    element = element.to(signal.device)
    gather = permutation.to(signal.device)[element]                  # [N, 12]
    centre = signal[:, :1]
    neighbours = torch.gather(
        signal[:, 1:], 1,
        gather[:, :, None, None].expand(-1, -1, signal.shape[2], VECTOR_DIM))
    rotated = torch.einsum("nij,nktj->nkti", matrices.to(signal.device)[element],
                           torch.cat((centre, neighbours), dim=1))
    return rotated


def apply_group_shells(signal, element, matrices, permutation, shells):
    """`apply_group` for a star of several concentric icosahedral shells.

    ``signal`` is ``[N, 1 + 12*shells, T, 3]``.  The same channel permutation is
    applied within each shell and the same rotation to every vector, which is
    exactly the action on a multi-shell star.
    """
    expected = 1 + NEIGHBOUR_COUNT * shells
    if signal.shape[1] != expected:
        raise ValueError(f"signal must be [N,{expected},T,3], got {tuple(signal.shape)}")
    element = element.to(signal.device)
    gather = permutation.to(signal.device)[element]
    pieces = [signal[:, :1]]
    for shell in range(shells):
        start = 1 + NEIGHBOUR_COUNT * shell
        block = signal[:, start:start + NEIGHBOUR_COUNT]
        pieces.append(torch.gather(
            block, 1,
            gather[:, :, None, None].expand(-1, -1, signal.shape[2], VECTOR_DIM)))
    stacked = torch.cat(pieces, dim=1)
    return torch.einsum("nij,nktj->nkti", matrices.to(signal.device)[element], stacked)


def random_group(signal, matrices, permutation, generator=None, exclude_identity=False):
    count = len(matrices)
    low = 1 if exclude_identity else 0
    element = torch.randint(low, count, (signal.shape[0],), device=signal.device,
                            generator=generator)
    return apply_group(signal, element, matrices, permutation), element


def random_so3(signal, generator=None):
    """Continuous SO(3) view: exact on the vectors, nearest-vertex on the channels.

    A star indexed by twelve fixed channels cannot represent a fractional
    rotation, so the channel map is the nearest icosahedral vertex of the rotated
    direction.  The vectors themselves are rotated exactly.  At the 60 proper
    group elements this reduces to ``apply_group``; between them it interpolates
    the vectors while the channel map stays piecewise constant.
    """
    rows = signal.shape[0]
    noise = torch.randn(rows, 3, 3, device=signal.device, dtype=signal.dtype,
                        generator=generator)
    q, r = torch.linalg.qr(noise)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    flip = torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)
    rotation = torch.cat((q[:, :, :2], q[:, :, 2:] * flip[:, None, None]), dim=-1)

    directions = torch.as_tensor(icosahedron_vertices(), device=signal.device,
                                 dtype=signal.dtype)
    pulled = torch.einsum("nji,kj->nki", rotation, directions)       # R^T d_k
    gather = (pulled @ directions.T).argmax(-1)                      # [N, 12]
    centre = signal[:, :1]
    neighbours = torch.gather(
        signal[:, 1:], 1,
        gather[:, :, None, None].expand(-1, -1, signal.shape[2], VECTOR_DIM))
    return torch.einsum("nij,nktj->nkti", rotation,
                        torch.cat((centre, neighbours), dim=1)), rotation
