"""Explicit observer worldline -> reference transformation -> observed field.

Pure translating Killing fields only. Coordinates follow ReferenceFrame3d.cpp:
lab = a(t) + R(t) @ (observed - a(t_ref)), with R=I here.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


class TranslatingObserverField:
    def __init__(self, times, velocities):
        self.times = np.asarray(times, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        if (self.times.ndim != 1 or len(self.times) < 2
                or self.velocities.shape != (len(self.times), 3)
                or not np.isfinite(self.times).all()
                or not np.isfinite(self.velocities).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Require increasing times and finite velocity samples [T,3].')
        self.spline = CubicSpline(self.times, self.velocities, axis=0,
                                  bc_type='natural', extrapolate=False)

    def __call__(self, lab_positions, time):
        if not self.times[0] <= time <= self.times[-1]:
            raise ValueError('Observer query outside schedule.')
        return np.broadcast_to(self.spline(time), np.asarray(lab_positions).shape)


class ReferenceFrameFromWorldline:
    def __init__(self, observer_field, reference_time, camera_start=(0., 0., 0.)):
        self.observer = observer_field
        self.t0 = float(reference_time)
        self.start = np.asarray(camera_start, dtype=np.float64)
        if self.start.shape != (3,) or not np.isfinite(self.start).all():
            raise ValueError('Invalid camera start.')
        if not observer_field.times[0] <= self.t0 <= observer_field.times[-1]:
            raise ValueError('Reference time outside schedule.')
        self.forward = self._integrate(observer_field.times[-1])
        self.backward = self._integrate(observer_field.times[0])

    def _integrate(self, end):
        if end == self.t0:
            return None
        result = solve_ivp(lambda t, a: self.observer(a, t), (self.t0, end),
                           self.start, method='DOP853', rtol=1e-12, atol=1e-13,
                           max_step=float(np.diff(self.observer.times).min()) / 4,
                           dense_output=True)
        if not result.success:
            raise RuntimeError(result.message)
        return result.sol

    def camera_position(self, time):
        if not self.observer.times[0] <= time <= self.observer.times[-1]:
            raise ValueError('Transformation outside worldline.')
        if time == self.t0:
            return self.start.copy()
        return (self.forward if time > self.t0 else self.backward)(time)

    def observed_to_lab(self, positions, time):
        return np.asarray(positions) - self.start + self.camera_position(time)

    def lab_to_observed(self, positions, time):
        return np.asarray(positions) - self.camera_position(time) + self.start


class ObservedVectorField:
    def __init__(self, lab_field, observer_field, transformation):
        self.lab_field = lab_field
        self.observer_field = observer_field
        self.transformation = transformation

    def __call__(self, observed_positions, time):
        # Both fields are sampled at the inverse-transformed LAB position.
        lab_positions = self.transformation.observed_to_lab(observed_positions, time)
        relative = self.lab_field(lab_positions, time) - self.observer_field(lab_positions, time)
        # R(t)^T is identity for this pure-translation verification.
        return relative
