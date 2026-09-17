"""Discrete as-steady-as-possible material frame, approved on 2026-09-17.

Fit one proper rotation between consecutive synchronous particle sets. This
finite-step objective converges to the motion-minimizing angular-velocity
objective; it is not the Eulerian velocity-field steadiness objective.
"""
from __future__ import annotations

import numpy as np


def initial_frame(relative):
    """Covariant right-handed frame from the labeled x+/x- and y+/y- pairs."""
    a = relative[:, 1, 0] - relative[:, 2, 0]
    b = relative[:, 3, 0] - relative[:, 4, 0]
    length = np.linalg.norm(a, axis=-1)
    if np.any(length <= 1e-14):
        raise ValueError("Collapsed initial first neighbor pair.")
    e1 = a / length[:, None]
    b = b - np.sum(e1*b, axis=-1)[:, None]*e1
    second = np.linalg.norm(b, axis=-1)
    if np.any(second <= 1e-10*length):
        raise ValueError("Initial labeled pairs cannot define an orientation.")
    e2 = b / second[:, None]
    return np.stack([e1, e2, np.cross(e1, e2)], axis=-1)


def asap_frame(geometry, times, *, return_details=False):
    """Return invariant canonical coordinates [B,7,T,3], in float64.

    A_k minimizes sum_i ||r_i(k+1)-A_k r_i(k)||^2 / dt_k^2.
    One SVD per primitive and interval; no iterative nonlinear optimization.
    R_0 follows material labels, R_(k+1)=A_k R_k, y=R^T(x-center).
    Reject ambiguous proper-rotation fits instead of choosing a hidden axis.
    """
    x = np.asarray(geometry, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if x.ndim != 4 or x.shape[1] != 7 or x.shape[-1] != 3 or x.shape[2] < 2:
        raise ValueError("Expected seven synchronous lines [B,7,T,3].")
    if t.ndim == 1:
        t = np.broadcast_to(t, (len(x), x.shape[2]))
    if t.shape != (len(x), x.shape[2]) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Require increasing physical times [B,T].")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite material coordinates.")
    r = x-x[:, :1]
    radius = np.linalg.norm(r, axis=-1).max(axis=(1, 2))
    if np.any(radius <= 1e-14):
        raise ValueError("Collapsed primitive.")
    z = r/radius[:, None, None, None]
    # C = sum next @ previous.T; row-vector storage is kept explicit.
    covariance = np.einsum('bnki,bnkj->bkij', z[:, 1:, 1:], z[:, 1:, :-1])
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.linalg.det(u @ vt)
    if np.any(singular[..., 1] <= 1e-10*singular[..., 0]):
        raise ValueError("Ambiguous consecutive rotation: rank below two.")
    if np.any((sign < 0) & ((singular[..., 1]-singular[..., 2]) <= 1e-10*singular[..., 0])):
        raise ValueError("Ambiguous proper rotation: tied reflected singular values.")
    diagonal = np.ones((*sign.shape, 3)); diagonal[..., 2] = sign
    increments = (u*diagonal[..., None, :]) @ vt
    rotations = np.empty((len(x), x.shape[2], 3, 3))
    rotations[:, 0] = initial_frame(z)
    for k in range(x.shape[2]-1):
        rotations[:, k+1] = increments[:, k] @ rotations[:, k]
    observed = np.einsum('bnki,bkij->bnkj', r, rotations)
    # A mathematically stationary line can retain SVD roundoff. Downstream unit
    # tangents divide such noise by 1e-12. Resolve motion below 1e-12 radii as
    # stationary before that nonlinear operation; use a Euclidean criterion.
    motion_size = np.linalg.norm(observed-observed[:, :, :1], axis=-1).max(axis=-1)
    stationary = motion_size <= 1e-12*radius[:, None]
    observed = np.where(stationary[:, :, None, None], observed[:, :, :1], observed)
    # The camera center is stationary by construction.
    observed[:, 0] = 0.
    if not return_details:
        return observed
    residual = np.diff(observed[:, 1:], axis=2)
    motion = np.mean(np.sum(residual**2, axis=-1), axis=1)/np.diff(t)**2
    return observed, dict(rotation=rotations, increments=increments,
        minimum_rank_ratio=float(np.min(singular[..., 1]/singular[..., 0])),
        speed_rms=np.sqrt(motion), radius=radius)


def transformed_observer(geometry):
    """Adversarial test: unrelated deterministic proper rotation at every sample."""
    x = np.asarray(geometry, dtype=np.float64)
    rng = np.random.default_rng(910917)
    matrices = rng.normal(size=(x.shape[2], 3, 3))
    q, _ = np.linalg.qr(matrices)
    q[..., 2] *= np.linalg.det(q)[..., None]
    relative_scale = np.linalg.norm(x-x[:, :1], axis=-1).max(axis=(1, 2))
    translation = rng.normal(size=(x.shape[2], 3))
    return np.einsum('kij,bnkj->bnki', q, x) + relative_scale[:, None, None, None]*translation[None, None]
