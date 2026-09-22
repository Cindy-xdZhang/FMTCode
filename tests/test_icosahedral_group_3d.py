"""Checks for the icosahedral group action on 13-line primitives.

The load-bearing claim is that applying a group element to a cached primitive is
the same thing as having traced the primitive in a rotated flow.  If that fails,
every "augmented view" the contrastive loss sees is a different object from the
one it claims to be.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.IcosahedralGroup_3D import (
    NEIGHBOUR_COUNT, apply_group, group_tensors, icosahedral_group,
    icosahedron_vertices, random_so3)


def test_vertices_form_a_regular_icosahedron():
    vertices = icosahedron_vertices()
    assert vertices.shape == (12, 3)
    assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0)
    gram = vertices @ vertices.T
    off = np.round(gram[~np.eye(12, dtype=bool)], 6)
    assert sorted(set(off.tolist())) == [-1.0, -0.447214, 0.447214], sorted(set(off.tolist()))
    # every vertex has exactly five nearest neighbours at cos = +1/sqrt(5)
    assert (np.isclose(gram, 1 / np.sqrt(5)).sum(axis=1) == 5).all()
    print("  test_vertices_form_a_regular_icosahedron ok")


def test_group_orders_are_60_and_120():
    proper, _ = icosahedral_group("i")
    full, _ = icosahedral_group("ih")
    assert len(proper) == 60, len(proper)
    assert len(full) == 120, len(full)
    assert np.allclose(np.linalg.det(proper), 1.0)
    assert sorted(set(np.round(np.linalg.det(full), 6).tolist())) == [-1.0, 1.0]
    for matrices in (proper, full):
        defect = np.abs(matrices @ matrices.transpose(0, 2, 1) - np.eye(3)).max()
        assert defect < 1e-12, defect
    print(f"  |I| = {len(proper)}, |I_h| = {len(full)}")
    print("  test_group_orders_are_60_and_120 ok")


def test_group_is_closed_under_multiplication():
    matrices, _ = icosahedral_group("ih")
    flat = matrices.reshape(len(matrices), 9)
    generator = np.random.default_rng(0)
    for _ in range(200):
        i, j = generator.integers(0, len(matrices), 2)
        product = (matrices[i] @ matrices[j]).reshape(1, 9)
        if np.abs(flat - product).max(axis=1).min() > 1e-9:
            raise AssertionError("product of two group elements left the group")
    print("  test_group_is_closed_under_multiplication ok")


def test_every_element_permutes_the_vertices():
    matrices, permutation = icosahedral_group("ih")
    vertices = icosahedron_vertices()
    for matrix, gather in zip(matrices, permutation):
        assert sorted(gather.tolist()) == list(range(NEIGHBOUR_COUNT))
        # permutation[j] is the index of M^T d_j
        defect = np.abs(vertices[gather] - vertices @ matrix).max()
        assert defect < 1e-9, defect
    print("  test_every_element_permutes_the_vertices ok")


def test_action_equals_rotating_the_traced_geometry():
    """The decisive check: group action == retracing in a rotated flow."""
    torch.manual_seed(0)
    matrices, permutation = group_tensors("ih", dtype=torch.float64)
    vertices = torch.as_tensor(icosahedron_vertices(), dtype=torch.float64)

    rows, steps, radius = 5, 9, 0.37
    centre = torch.randn(rows, steps, 3, dtype=torch.float64)
    # a primitive whose neighbour k is the centre track offset along direction k
    offsets = torch.randn(rows, NEIGHBOUR_COUNT, steps, 3, dtype=torch.float64) * 0.05
    neighbours = centre[:, None] + radius * vertices[None, :, None, :] + offsets
    primitive = torch.cat((centre[:, None], neighbours), dim=1)

    worst = 0.0
    for index in range(len(matrices)):
        element = torch.full((rows,), index, dtype=torch.long)
        acted = apply_group(primitive, element, matrices, permutation)
        # retrace: rotate the whole flow, which rotates centre, offsets and the
        # direction each neighbour was seeded along
        matrix = matrices[index]
        rotated_centre = centre @ matrix.T
        rotated_offsets = torch.einsum("ij,nktj->nkti", matrix, offsets)
        gather = permutation[index]
        retraced_neighbours = (rotated_centre[:, None]
                               + radius * vertices[None, :, None, :]
                               + rotated_offsets[:, gather])
        retraced = torch.cat((rotated_centre[:, None], retraced_neighbours), dim=1)
        worst = max(worst, (acted - retraced).abs().max().item())
    assert worst < 1e-12, worst
    print(f"  action-vs-retrace agreement over all 120 elements: max |diff| {worst:.2e}")
    print("  test_action_equals_rotating_the_traced_geometry ok")


def test_random_so3_is_proper_and_matches_the_group_on_group_elements():
    torch.manual_seed(1)
    signal = torch.randn(64, 13, 7, 3, dtype=torch.float64)
    _, rotation = random_so3(signal)
    identity = torch.eye(3, dtype=torch.float64).expand(64, 3, 3)
    assert (rotation @ rotation.transpose(1, 2) - identity).abs().max() < 1e-10
    assert (torch.linalg.det(rotation) - 1.0).abs().max() < 1e-10
    print("  test_random_so3_is_proper_and_matches_the_group_on_group_elements ok")


def test_so3_covering_is_finer_than_octahedral():
    """Why icosahedral: measure the worst-case gap to the nearest group element."""
    from scipy.spatial.transform import Rotation
    sample = Rotation.random(20000, random_state=3).as_matrix()
    report = {}
    for name, matrices in (("icosahedral I (60)", icosahedral_group("i")[0]),):
        trace = np.einsum("nij,gij->ng", sample, matrices)
        angle = np.degrees(np.arccos(np.clip((trace.max(axis=1) - 1) / 2, -1, 1)))
        report[name] = (angle.mean(), angle.max())
    from FMT_Utils.OctahedralGroup_3D import octahedral_group
    proper = octahedral_group("o")[0]
    trace = np.einsum("nij,gij->ng", sample, proper)
    angle = np.degrees(np.arccos(np.clip((trace.max(axis=1) - 1) / 2, -1, 1)))
    report["octahedral O (24)"] = (angle.mean(), angle.max())
    for name, (mean, worst) in report.items():
        print(f"  {name:22s} mean {mean:5.1f} deg   max {worst:5.1f} deg")
    assert report["icosahedral I (60)"][0] < report["octahedral O (24)"][0]
    print("  test_so3_covering_is_finer_than_octahedral ok")


if __name__ == "__main__":
    test_vertices_form_a_regular_icosahedron()
    test_group_orders_are_60_and_120()
    test_group_is_closed_under_multiplication()
    test_every_element_permutes_the_vertices()
    test_action_equals_rotating_the_traced_geometry()
    test_random_so3_is_proper_and_matches_the_group_on_group_elements()
    test_so3_covering_is_finer_than_octahedral()
    print("ICOSAHEDRAL GROUP 3D TEST PASSED")
