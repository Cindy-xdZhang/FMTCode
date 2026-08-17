"""Tests for the label-math and loader fixes of docs/code_review_2026-08-16.md.

Covers: signed IVD (A2), Amira payload offset (A3), Vatistas far-field softplus (A9),
FTLE baseline-sign invariance (B), AnalyticalFlowCreator local_dict freeze (B).

Run:  python tests/test_labels_and_loaders.py   (from the repo root)
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from FLowUtils.VectorField2d import UnsteadyVectorField2D  # noqa: E402
from FLowUtils.ScalarField2d import compute_ivd_2D, compute_vorticity_2D  # noqa: E402
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import AmiraLoader  # noqa: E402
from FLowUtils.AnalyticalFlowCreator import AnalyticalFlowCreator  # noqa: E402
from FMT_Utils.FTLE_fitting_utils import computeFTLEFromPathlineCrossPrimitive  # noqa: E402
from FittingVatistasParam import vatistas_mixture_velocity  # noqa: E402


def _make_field(u, v, dom_min, dom_max):
    T, Y, X = u.shape
    vf = UnsteadyVectorField2D(X, Y, T, list(dom_min), list(dom_max))
    vf.field = np.stack([u, v], axis=-1).astype(np.float64)
    return vf


def test_ivd_uses_signed_vorticity():
    """Half-plane omega=+1, half-plane omega=-1: signed IVD ~= 1 in the interior;
    the old |omega|-based version returned ~0 everywhere."""
    X = Y = 64; T = 2
    xs = np.linspace(-1, 1, X)
    v_prof = -np.abs(xs)                          # dv/dx = -sign(x) -> omega = -sign(x)
    u = np.zeros((T, Y, X)); v = np.tile(v_prof, (T, Y, 1))
    vf = _make_field(u, v, (-1, -1, 0.0), (1, 1, 1.0))

    vort = compute_vorticity_2D(vf)
    assert abs(float(np.mean(vort))) < 5e-2       # +-1 halves cancel
    ivd = compute_ivd_2D(vf)
    interior = ivd[:, :, 5:X // 2 - 3]            # away from the kink and edges
    med = float(np.median(interior))
    assert med > 0.9, f"signed IVD median {med} (old |omega| version gave ~0)"
    print(f"ok: signed IVD median {med:.3f} on counter-rotating halves (old ~0)")


def test_ivd_zero_for_rigid_rotation():
    X = Y = 48; T = 2
    ax = np.linspace(-1, 1, X)
    gx, gy = np.meshgrid(ax, ax)                  # (Y, X)
    u = np.tile(-gy, (T, 1, 1)); v = np.tile(gx, (T, 1, 1))   # omega = 2 everywhere
    vf = _make_field(u, v, (-1, -1, 0.0), (1, 1, 1.0))
    ivd = compute_ivd_2D(vf)
    assert float(np.abs(ivd[:, 2:-2, 2:-2]).max()) < 1e-6
    print("ok: IVD == 0 for rigid rotation")


def test_amira_real_format_and_roundtrip():
    T, Y, X = 3, 4, 5
    field = np.arange(T * Y * X * 2, dtype=np.float32).reshape(T, Y, X, 2)
    vf = _make_field(field[..., 0], field[..., 1], (0, 0, 0.0), (1, 1, 1.0))

    with tempfile.TemporaryDirectory() as td:
        # legacy format written by our own saver must still round-trip
        p1 = os.path.join(td, "legacy.am")
        AmiraLoader.save_vector_field2d(p1, vf)
        back = AmiraLoader.load_vector_field2d(p1)
        assert np.allclose(back.field, vf.getDataAsNumpy())

        # real AmiraMesh layout: "# Data section follows" line, then "@1", then binary.
        # The old regex left the "@1\n" bytes in the payload -> whole field shifted
        # by 3 bytes and silently reinterpreted as garbage.
        with open(p1, "rb") as f:
            raw = f.read()
        payload = vf.getDataAsNumpy().astype("<f4").tobytes(order="C")
        header = raw[: len(raw) - len(payload)].decode("ascii")
        header_real = header.replace("@1\n@1\n", "@1\n# Data section follows\n@1\n")
        assert "# Data section follows" in header_real
        p2 = os.path.join(td, "real_format.am")
        with open(p2, "wb") as f:
            f.write(header_real.encode("ascii")); f.write(payload)
        back2 = AmiraLoader.load_vector_field2d(p2)
        assert np.allclose(back2.field, vf.getDataAsNumpy()), \
            "real-format Amira payload misaligned (the 3-byte '@1' offset bug)"
    print("ok: Amira real format + legacy round-trip both exact")


def test_vatistas_far_field_decays():
    """Far field must decay ~1/(2*pi*r); the old p2n clamp made it grow linearly."""
    W = 200
    xs = np.linspace(0.1, 1.4, W)
    coords = torch.zeros(1, W, 2); coords[0, :, 0] = torch.from_numpy(xs)
    params = torch.zeros(1, 1, 7)
    params[0, 0, 1] = 1.0; params[0, 0, 2] = 1.0        # sx, sy
    params[0, 0, 3] = 0.05                              # rc (yaml lower bound)
    params[0, 0, 4] = 6.0                               # n  (yaml upper bound)
    shape_idx = torch.full((1, 1), 2, dtype=torch.long)  # ccw center
    bounds = {"rc": (0.05, 1.0), "n": (0.5, 6.0), "s": (0.05, 5.0)}
    vel = vatistas_mixture_velocity(coords, params, shape_idx, bounds)  # [1,1,W,2]
    speed = vel[0, 0].norm(dim=-1).detach().numpy()

    beyond = xs > 0.7   # old clamp kicked in at r ~ 0.61 for these params
    s = speed[beyond]
    assert np.all(np.diff(s) < 0), "far-field speed must be strictly decreasing"
    ratio = s[-1] / (1.0 / (2 * np.pi * xs[beyond][-1]))
    assert 0.8 < ratio < 1.2, f"far field should approach 1/(2 pi r), ratio={ratio}"
    print(f"ok: Vatistas far field decays ~1/(2 pi r) (ratio {ratio:.3f}; old grew linearly)")


def test_ftle_invariant_to_baseline_order():
    """C = J^T J is invariant to swapping x+/x- (or y+/y-); the old signed clamp_min
    turned the swapped case into ~1e12 garbage."""
    eps = 0.01
    A = np.diag([2.0, 0.5]); expected = np.log(2.0)
    starts = np.array([[0, 0], [eps, 0], [-eps, 0], [0, eps], [0, -eps]], np.float64)
    ends = starts @ A.T + np.array([0.3, -0.2])
    prim = np.zeros((1, 5, 2, 3))
    prim[0, :, 0, :2] = starts; prim[0, :, 1, :2] = ends
    prim[0, :, 0, 2] = 0.0;     prim[0, :, 1, 2] = 1.0

    def ftle_of(p):
        return float(computeFTLEFromPathlineCrossPrimitive(
            torch.from_numpy(p).float(), vectorfield_dt=0.05)[0])

    base = ftle_of(prim)
    assert abs(base - expected) < 1e-4

    swapped = prim.copy(); swapped[0, [1, 2]] = swapped[0, [2, 1]]  # swap x+ / x-
    assert abs(ftle_of(swapped) - expected) < 1e-4, \
        "swapped x+/x- must give the same FTLE (old clamp gave ~1e1 log of 1e12)"
    swapped2 = prim.copy(); swapped2[0, [3, 4]] = swapped2[0, [4, 3]]
    assert abs(ftle_of(swapped2) - expected) < 1e-4
    print(f"ok: FTLE {base:.4f} == ln 2, invariant to x+/x- and y+/y- order")


def test_analytical_creator_mixed_expressions_vary_in_time():
    """Callable x-expression + string y-expression: the old local_dict reuse froze t
    at the first frame for the y component."""
    creator = AnalyticalFlowCreator(lambda x, y, t: x * 0 + t, "x*0 + t", {})
    vf = creator.create_flow_field((8, 8), 4, [0, 0, 0.0], [1, 1, 3.0])
    data = np.asarray(vf.field)
    assert np.allclose(data[0, :, :, 1], 0.0)
    assert np.allclose(data[3, :, :, 1], 3.0), \
        "y component frozen at t=0 (local_dict reuse bug)"
    assert np.allclose(data[..., 0], data[..., 1])
    print("ok: mixed callable/string expressions vary in time")


if __name__ == "__main__":
    test_ivd_uses_signed_vorticity()
    test_ivd_zero_for_rigid_rotation()
    test_amira_real_format_and_roundtrip()
    test_vatistas_far_field_decays()
    test_ftle_invariant_to_baseline_order()
    test_analytical_creator_mixed_expressions_vary_in_time()
    print("ALL LABEL/LOADER TESTS PASSED")
