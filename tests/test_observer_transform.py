"""Property tests for the corrected observer transform (killing_abc_transform).

Pins the fix of docs/code_review_2026-08-16.md A1: the translation-rate term must be
the exact derivative of c(t) = Os - Q(t) p(t); the legacy constant (-a, -b) produced
a spurious uniform background flow of magnitude |(Q-I)(a,b)| (typ. ~10% of the field).

Run:  python tests/test_observer_transform.py   (from the repo root)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import VatistasFlowDatasetGenerator as vgen  # noqa: E402


def _own_killing_vel_eval(abc_const, center):
    a, b, c = abc_const
    cx, cy = center

    def vel_eval(pts):  # pts: [..., 2]
        ux = a - c * (pts[..., 1] - cy)
        uy = b + c * (pts[..., 0] - cx)
        return np.stack([ux, uy], axis=-1)

    return vel_eval


def test_observer_views_own_killing_field_as_zero():
    """A co-moving observer must see its own (steady, constant-abc) field as zero.

    With the legacy (-a,-b) term the residual is |(Q-I)(a,b)| ~ 0.56 here."""
    abc_const = (0.4, -0.3, 1.2)
    center = [0.2, -0.1]
    ocfg = {"timesteps": 101, "tmin": 0.0, "tmax": 1.0,
            "center": center, "start_pos": [0.3, 0.2]}
    G = 24
    ax = np.linspace(-1.0, 1.0, G)
    coords = np.stack(np.meshgrid(ax, ax, indexing="ij"), axis=-1)  # [G,G,2]

    abc_func = lambda t: np.array(abc_const, np.float64)  # noqa: E731
    field = vgen.killing_abc_transform(_own_killing_vel_eval(abc_const, center),
                                       abc_func, ocfg, coords)
    resid = float(np.abs(field).max())
    assert resid < 5e-4, f"observed own field residual {resid} (legacy bug ~0.56)"
    print(f"ok: own-Killing-field residual {resid:.2e} (legacy ~5.6e-1)")


def test_pathline_equivariance():
    """A steady-field pathline mapped by the frame must match a pathline integrated
    directly in the transformed field (endpoints compared)."""
    # steady, non-rigid analytic field
    def v_steady(p):
        x, y = p[..., 0], p[..., 1]
        return np.stack([-np.sin(np.pi * x) * np.cos(np.pi * y) * 0.4,
                         np.cos(np.pi * x) * np.sin(np.pi * y) * 0.4], axis=-1)

    a0, ad, b0, bd, c_const = 0.4, 0.5, -0.2, 0.3, 1.0
    abc_func = lambda t: np.array([a0 + ad * t, b0 + bd * t, c_const], np.float64)  # noqa: E731
    center = [0.0, 0.0]
    tmin, tmax, T = 0.0, 1.0, 161
    ocfg = {"timesteps": T, "tmin": tmin, "tmax": tmax,
            "center": center, "start_pos": [0.1, -0.2]}

    G = 100
    half = 2.5
    ax = np.linspace(-half, half, G)
    coords = np.stack(np.meshgrid(ax, ax, indexing="ij"), axis=-1)  # [G,G,2]
    field = vgen.killing_abc_transform(lambda pts: v_steady(pts), abc_func, ocfg, coords)

    # --- trajectory A: integrate in the steady field, then map through the frame ---
    n_fine = 1600
    dtf = (tmax - tmin) / n_fine
    x = np.array([0.3, 0.1])
    xs = [x.copy()]
    for k in range(n_fine):
        k1 = v_steady(x); k2 = v_steady(x + 0.5 * dtf * k1)
        k3 = v_steady(x + 0.5 * dtf * k2); k4 = v_steady(x + dtf * k3)
        x = x + (dtf / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x.copy())
    path_fine = vgen.integrate_observer_pathline(abc_func, ocfg["start_pos"], center,
                                                 tmin, tmax, n_fine + 1)
    Os = path_fine[0]

    def frame(pt, k):  # x* = Q x + c at fine step k (theta exact for constant c)
        th = c_const * (k * dtf)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Q = R.T
        c_t = Os - Q @ path_fine[k]
        return Q @ pt + c_t

    yA_end = frame(xs[-1], n_fine)
    assert np.allclose(frame(xs[0], 0), xs[0], atol=1e-12)  # t0 identity

    # --- trajectory B: integrate directly in the produced grid field ---
    dt_grid = (tmax - tmin) / (T - 1)
    dxg = ax[1] - ax[0]

    def sample(pt, t):
        fi = min(max((t - tmin) / dt_grid, 0.0), T - 1 - 1e-9)
        i0 = int(fi); w = fi - i0
        gx = min(max((pt[0] + half) / dxg, 0.0), G - 1 - 1e-9)
        gy = min(max((pt[1] + half) / dxg, 0.0), G - 1 - 1e-9)
        ix, iy = int(gx), int(gy); wx, wy = gx - ix, gy - iy

        def bil(sl):  # coords built with indexing='ij' -> axis0 = x, axis1 = y
            return ((1 - wx) * (1 - wy) * sl[ix, iy] + wx * (1 - wy) * sl[ix + 1, iy]
                    + (1 - wx) * wy * sl[ix, iy + 1] + wx * wy * sl[ix + 1, iy + 1])

        return (1 - w) * bil(field[i0]) + w * bil(field[i0 + 1])

    y = xs[0].copy()
    for k in range(n_fine):
        t = tmin + k * dtf
        k1 = sample(y, t); k2 = sample(y + 0.5 * dtf * k1, t + 0.5 * dtf)
        k3 = sample(y + 0.5 * dtf * k2, t + 0.5 * dtf); k4 = sample(y + dtf * k3, t + dtf)
        y = y + (dtf / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    err = float(np.linalg.norm(y - yA_end))
    assert err < 0.05, f"pathline equivariance endpoint error {err} (legacy bug ~0.2+)"
    print(f"ok: pathline equivariance endpoint error {err:.4f} (legacy ~0.2+)")


if __name__ == "__main__":
    test_observer_views_own_killing_field_as_zero()
    test_pathline_equivariance()
    print("ALL OBSERVER TRANSFORM TESTS PASSED")
