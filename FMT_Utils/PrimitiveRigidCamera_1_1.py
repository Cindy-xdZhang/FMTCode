"""Center-following least-squares rigid camera for synchronous material paths.

Other_PrimitiveRigidCamera_1.1 minimizes particle motion, not the Eulerian
time derivative of a velocity field. Only NumPy is required.
"""
from __future__ import annotations

import numpy as np


def skew(vector):
    """Return the matrix such that skew(vector) @ p == cross(vector, p)."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def fit_angular_velocity(relative_positions, relative_velocities, weights=None):
    """Fit omega to r_dot = omega cross r; reject unidentifiable rotations."""
    r = np.asarray(relative_positions, dtype=float)
    v = np.asarray(relative_velocities, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3 or v.shape != r.shape:
        raise ValueError("Positions and velocities must both have shape [N, 3].")
    w = np.ones(len(r)) if weights is None else np.asarray(weights, dtype=float)
    if (w.shape != (len(r),) or not np.isfinite(w).all() or np.any(w <= 0)
            or not np.isfinite(r).all() or not np.isfinite(v).all()):
        raise ValueError("Require finite coordinates and positive fixed weights.")
    matrix = np.concatenate([-np.sqrt(wi) * skew(ri) for wi, ri in zip(w, r)])
    rhs = (np.sqrt(w)[:, None] * v).reshape(-1)
    omega, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=1e-10)
    if rank != 3:
        raise ValueError("Rotation is not identifiable: relative positions are collinear or collapsed.")
    residual = v - np.cross(omega, r)
    return omega, residual, float(singular[0] / singular[-1])


class SampledPrimitive:
    """Cubic Hermite interpolation in common physical time, never arc length.

    If velocities are absent, second-order finite differences estimate them.
    Finite sampling/interpolation is not exactly equivariant to arbitrary Q(t).
    """

    def __init__(self, times, positions, velocities=None):
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        if (self.times.ndim != 1 or len(self.times) < 3
                or self.positions.ndim != 3
                or self.positions.shape[0] != len(self.times)
                or self.positions.shape[2] != 3
                or not np.isfinite(self.times).all()
                or not np.isfinite(self.positions).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError("Require increasing physical times [T] and finite positions [T,N,3].")
        self.velocities = (np.gradient(self.positions, self.times, axis=0, edge_order=2)
                           if velocities is None else np.asarray(velocities, dtype=float))
        if self.velocities.shape != self.positions.shape or not np.isfinite(self.velocities).all():
            raise ValueError("Velocities must match positions and be finite.")

    def __call__(self, time):
        if not self.times[0] <= time <= self.times[-1]:
            raise ValueError("Query outside the sampled physical time interval.")
        index = np.clip(np.searchsorted(self.times, time, side="right") - 1, 0, len(self.times) - 2)
        dt = self.times[index + 1] - self.times[index]
        s = (time - self.times[index]) / dt
        p, q = self.positions[index:index + 2]
        a, b = self.velocities[index:index + 2]
        position = ((2*s**3 - 3*s*s + 1)*p + (s**3 - 2*s*s + s)*dt*a
                    + (-2*s**3 + 3*s*s)*q + (s**3 - s*s)*dt*b)
        velocity = ((6*s*s - 6*s)*p/dt + (3*s*s - 4*s + 1)*a
                    + (-6*s*s + 6*s)*q/dt + (3*s*s - 2*s)*b)
        return position, velocity


def solve_camera(trajectory, times, *, center_index=0, weights=None, substeps=4):
    """Integrate R_dot=skew(omega)R using fourth-order Runge-Kutta.

    trajectory(t) returns physical positions and velocities [N,3]. R maps
    observed coordinates into input coordinates. Polar projection removes
    numerical rotation drift only; it does not fit an independent pose.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or substeps < 1 or int(substeps) != substeps):
        raise ValueError("Require increasing times and a positive integer substep count.")
    initial, _ = trajectory(float(times[0]))
    if not 0 <= center_index < len(initial):
        raise ValueError("Invalid center particle index.")
    selected = np.arange(len(initial)) != center_index
    w = np.ones(len(initial) - 1) if weights is None else np.asarray(weights, dtype=float)

    def sample(time):
        x, v = map(lambda value: np.asarray(value, dtype=float), trajectory(float(time)))
        r = x - x[center_index]
        rv = v - v[center_index]
        omega, residual, condition = fit_angular_velocity(r[selected], rv[selected], w)
        return x, v, r, rv, omega, residual, condition

    rotation = np.eye(3)
    matrices = [rotation.copy()]
    for left, right in zip(times[:-1], times[1:]):
        dt = (right - left) / substeps
        for k in range(substeps):
            time = left + k * dt
            a = skew(sample(time)[4])
            b = skew(sample(time + dt / 2)[4])
            c = skew(sample(min(right, time + dt))[4])
            k1 = a @ rotation
            k2 = b @ (rotation + dt * k1 / 2)
            k3 = b @ (rotation + dt * k2 / 2)
            k4 = c @ (rotation + dt * k3)
            rotation = rotation + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            u, _, vt = np.linalg.svd(rotation)
            correction = np.diag([1., 1., np.linalg.det(u @ vt)])
            rotation = u @ correction @ vt
        matrices.append(rotation.copy())
    rows = [sample(t) for t in times]
    x, v, r, rv, omega, residual, condition = [np.array([row[j] for row in rows]) for j in range(7)]
    matrices = np.array(matrices)
    observed = np.einsum("tnj,tjk->tnk", r, matrices)
    observed_velocity = np.einsum("tnj,tjk->tnk", rv - np.cross(omega[:, None, :], r), matrices)
    centered_speed = np.sqrt(np.average(np.sum(rv[:, selected]**2, axis=-1), weights=w, axis=1))
    observed_speed = np.sqrt(np.average(np.sum(residual**2, axis=-1), weights=w, axis=1))
    return dict(times=times, positions=x, velocities=v, centered=r, observed=observed,
                rotation=matrices, omega=omega, observed_velocity=observed_velocity,
                centered_speed=centered_speed, observed_speed=observed_speed,
                condition=condition, weights=w)


def rotation_axis(axis, angle, rate):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    h = skew(axis)
    matrix = np.eye(3) + np.sin(angle)*h + (1-np.cos(angle))*(h @ h)
    return matrix, rate * h @ matrix


def analytic_motion(time, scene="deform"):
    """Invertible affine material flow; seven labeled trajectories, arbitrary units."""
    t = float(time)
    seeds = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0],
                      [0, .8, 0], [0, -.8, 0], [0, 0, .65], [0, 0, -.65]])
    c = np.array([.52*t, .38*np.sin(.9*t), .12*t + .14*np.sin(1.1*t)])
    cd = np.array([.52, .342*np.cos(.9*t), .12 + .154*np.cos(1.1*t)])
    if scene == "translation":
        g, gd = np.eye(3), np.zeros((3, 3))
    else:
        a, ad = rotation_axis([.2, .4, 1.], .9*t + .28*np.sin(.7*t), .9 + .196*np.cos(.7*t))
        b, bd = rotation_axis([1., -.3, .2], .38*np.sin(.8*t), .304*np.cos(.8*t))
        g, gd = a @ b, ad @ b + a @ bd
    s, sd = np.eye(3), np.zeros((3, 3))
    if scene == "deform":
        stretch = np.exp(.16*t)
        s[0, 0], sd[0, 0] = stretch, .16*stretch
        s[1, 1], sd[1, 1] = 1/np.sqrt(stretch), -.08/np.sqrt(stretch)
        s[2, 2], sd[2, 2] = 1/np.sqrt(stretch), -.08/np.sqrt(stretch)
        s[0, 1], sd[0, 1] = .45*np.sin(.6*t), .27*np.cos(.6*t)
        s[1, 2], sd[1, 2] = .20*np.sin(.9*t), .18*np.cos(.9*t)
    elif scene not in ("translation", "rigid"):
        raise ValueError("Unknown analytic scene.")
    matrix, matrix_dot = g @ s, gd @ s + g @ sd
    return seeds @ matrix.T + c, seeds @ matrix_dot.T + cd


def external_observer(time, initial_rotation=None):
    """Noncommuting time-dependent rigid observer, with analytic derivative."""
    t = float(time)
    a, ad = rotation_axis([-.3, 1, .2], .63*t+.2*np.sin(1.3*t), .63+.26*np.cos(1.3*t))
    b, bd = rotation_axis([.6, .1, 1], -.37*t, -.37)
    q0 = np.eye(3) if initial_rotation is None else np.asarray(initial_rotation)
    q, qd = a @ b @ q0, (ad @ b + a @ bd) @ q0
    shift = np.array([.22*np.sin(t), -.25*t, .25*np.sin(.8*t)])
    shift_dot = np.array([.22*np.cos(t), -.25, .2*np.cos(.8*t)])
    return q, qd, shift, shift_dot


def change_observer(trajectory, initial_rotation=None):
    def transformed(time):
        x, v = trajectory(time)
        q, qd, shift, shift_dot = external_observer(time, initial_rotation)
        return x @ q.T + shift, v @ q.T + x @ qd.T + shift_dot
    return transformed
