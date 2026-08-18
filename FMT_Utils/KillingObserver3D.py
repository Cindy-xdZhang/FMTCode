"""Validated 3D Killing-observer frame and steady-to-unsteady pushforward."""

from __future__ import annotations

import numpy as np


def hat(vector):
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_exp(angular_velocity, dt):
    angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
    angle = float(np.linalg.norm(angular_velocity)) * float(dt)
    matrix = hat(angular_velocity) * float(dt)
    if angle < 1e-12:
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    matrix /= angle
    return np.eye(3) + np.sin(angle) * matrix + (1.0 - np.cos(angle)) * matrix @ matrix


def integrate_killing_frame(parameters, dt):
    """Integrate q=(translation, angular velocity); return observed->lab R and D."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 6:
        raise ValueError("parameters must have shape [T,6]")
    count = len(parameters)
    rotation = np.empty((count, 3, 3), dtype=np.float64); rotation[0] = np.eye(3)
    for index in range(1, count):
        average = 0.5 * (parameters[index - 1, 3:] + parameters[index, 3:])
        rotation[index] = rotation_exp(average, dt) @ rotation[index - 1]
    integrand = np.einsum("tji,tj->ti", rotation, parameters[:, :3])
    displacement = np.zeros((count, 3), dtype=np.float64)
    if count > 1:
        displacement[1:] = np.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * float(dt), axis=0
        )
    return rotation, displacement


def smooth_channel_observer(times, domain_min, domain_max):
    """Deterministic small-amplitude time-varying rigid observer for channel data."""
    times = np.asarray(times, dtype=np.float64)
    lower = np.asarray(domain_min, dtype=np.float64)
    upper = np.asarray(domain_max, dtype=np.float64)
    center = 0.5 * (lower + upper); span = upper - lower
    axis = np.array([0.35, 0.55, 0.76], dtype=np.float64); axis /= np.linalg.norm(axis)
    phase = 2.0 * np.pi * times
    # theta(t)=A sin(2pi t)+0.35A sin(4pi t), hence w=dtheta/dt * axis.
    amplitude = 0.035
    theta_dot = amplitude * 2.0 * np.pi * np.cos(phase) + \
        0.35 * amplitude * 4.0 * np.pi * np.cos(2.0 * phase)
    angular = theta_dot[:, None] * axis[None, :]
    translation_amplitude = np.array([0.012, 0.018, 0.012]) * span
    moving_center = center + translation_amplitude * np.stack(
        (np.sin(phase), np.sin(phase + 1.1), np.sin(phase + 2.0)), axis=-1
    )
    center_velocity = translation_amplitude * 2.0 * np.pi * np.stack(
        (np.cos(phase), np.cos(phase + 1.1), np.cos(phase + 2.0)), axis=-1
    )
    translation = center_velocity - np.cross(angular, moving_center)
    return np.concatenate((translation, angular), axis=1)


def compose_steady_to_unsteady(points_xyz, steady_interpolator, parameters,
                               rotation, displacement, bounds_min=None, bounds_max=None):
    """Evaluate v(x,t)=R s(R^T x-D)+t_vec+w cross x on fixed lab points."""
    points = np.asarray(points_xyz, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    output = np.empty((len(parameters), len(points), 3), dtype=np.float32)
    for index, (q, matrix, shift) in enumerate(zip(parameters, rotation, displacement)):
        observed = points @ matrix - shift
        if bounds_min is not None:
            outside = ((observed < np.asarray(bounds_min)) |
                       (observed > np.asarray(bounds_max))).any(axis=1)
            if outside.any():
                raise ValueError(
                    f"observer inverse map leaves steady domain at t-index {index}: "
                    f"{outside.sum()}/{len(points)} points"
                )
        steady = np.asarray(steady_interpolator(observed), dtype=np.float64)
        observer_velocity = q[:3] + np.cross(np.broadcast_to(q[3:], points.shape), points)
        output[index] = (steady @ matrix.T + observer_velocity).astype(np.float32)
    return output
