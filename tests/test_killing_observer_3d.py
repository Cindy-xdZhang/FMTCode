import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.KillingObserver3D import (
    compose_steady_to_unsteady, integrate_killing_frame,
)


def test_pushforward_recovers_steady_function():
    times = np.linspace(0, 1, 17); dt = times[1] - times[0]
    parameters = np.zeros((len(times), 6))
    parameters[:, :3] = np.stack((0.1 * np.cos(times), -0.05 * np.sin(times),
                                  0.03 * np.cos(2 * times)), axis=-1)
    parameters[:, 3:] = np.stack((0.2 * np.sin(times), 0.1 * np.cos(times),
                                  -0.15 * np.sin(2 * times)), axis=-1)
    rotation, displacement = integrate_killing_frame(parameters, dt)
    rng = np.random.default_rng(4); points = rng.uniform(-0.3, 0.3, size=(100, 3))

    def steady(x):
        return np.stack((x[:, 1] + 0.2 * x[:, 2], -x[:, 0], 0.3 * x[:, 2]), axis=-1)

    lab = compose_steady_to_unsteady(points, steady, parameters, rotation, displacement)
    for index, q in enumerate(parameters):
        observed = points @ rotation[index] - displacement[index]
        observer_velocity = q[:3] + np.cross(np.broadcast_to(q[3:], points.shape), points)
        pushed = (lab[index] - observer_velocity) @ rotation[index]
        np.testing.assert_allclose(pushed, steady(observed), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.einsum("tij,tkj->tik", rotation, rotation),
        np.broadcast_to(np.eye(3), (len(times), 3, 3)), atol=1e-12,
    )


if __name__ == "__main__":
    test_pushforward_recovers_steady_function()
    print("KILLING OBSERVER 3D TEST PASSED")
