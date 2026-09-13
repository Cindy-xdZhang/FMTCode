"""Mechanistic checks of the existing AIVD recipe, without changing it."""
import unittest
import numpy as np
import torch
from FMT_Utils.DFT_FMT_3D import (
    pathline_anchored_kinematic_dft_features_3d,
    pathline_velocity_gradient_scalar_sequences_3d,
)
from FMT_Utils.Task12Data_3D import _anchored_recipe, feature_matrix


def fixture(length=12):
    t = np.linspace(0, 1, length)
    rng = np.random.default_rng(7107)
    a = rng.normal(size=(9, 3, 3)) * .13
    d = np.eye(3)[None, None] + t[None, :, None, None] * a[:, None]
    centre = rng.normal(size=(9, 1, 3)) + t[None, :, None] * np.array([.3, -.2, .1])
    x = np.repeat(centre[:, None], 7, axis=1)
    for axis in range(3):
        x[:, 1+2*axis] += .5*d[..., axis]
        x[:, 2+2*axis] -= .5*d[..., axis]
    return x, t


def scalar(x):
    return pathline_anchored_kinematic_dft_features_3d(
        torch.as_tensor(x), **_anchored_recipe('aivd1w3_dft'))


def reference_numpy(x):
    d = np.stack((x[:, 1]-x[:, 2], x[:, 3]-x[:, 4], x[:, 5]-x[:, 6]), axis=-1)
    derivative = np.empty_like(d)
    derivative[:, 0], derivative[:, -1] = d[:, 1]-d[:, 0], d[:, -1]-d[:, -2]
    derivative[:, 1:-1] = .5*(d[:, 2:]-d[:, :-2])
    g = derivative @ np.linalg.pinv(d, rcond=1e-6)
    omega = np.stack((g[..., 2, 1]-g[..., 1, 2], g[..., 0, 2]-g[..., 2, 0],
                      g[..., 1, 0]-g[..., 0, 1]), axis=-1)
    ivd = np.linalg.norm(omega-omega.mean(axis=0, keepdims=True), axis=-1)
    return ivd[:, :3].sum(axis=1, keepdims=True)


class AIVDRecipeTests(unittest.TestCase):
    def test_exact_recipe_and_independent_numpy(self):
        recipe = _anchored_recipe('aivd1w3_dft')
        self.assertEqual(recipe['channels'], (0,))
        self.assertEqual(recipe['window'], 3)
        self.assertEqual(recipe['num_freq'], 1)
        self.assertEqual(recipe['anchor_names'], ())
        x, _ = fixture()
        actual = scalar(x)
        self.assertEqual(actual.shape, (9, 1))
        np.testing.assert_allclose(actual, reference_numpy(x), rtol=1e-11, atol=1e-12)
        record = {'raw': x.astype(np.float32).reshape(9, -1), 'features': {}}
        np.testing.assert_array_equal(feature_matrix(record, 'aivd1w3_dft'), scalar(x.astype(np.float32)))

    def test_centre_is_not_an_input_feature(self):
        x, _ = fixture()
        changed = x.copy()
        changed[:, 0] += 100.0
        np.testing.assert_array_equal(scalar(x), scalar(changed))

    def test_time_dependent_translation_and_constant_rotation(self):
        x, t = fixture()
        q, _ = np.linalg.qr(np.random.default_rng(10).normal(size=(3, 3)))
        q[:, 0] *= np.linalg.det(q)
        changed = np.einsum('ij,nktj->nkti', q, x)
        changed += np.stack((2*t*t, -np.sin(t), 3*t), axis=-1)[None, None]
        np.testing.assert_allclose(scalar(x), scalar(changed), rtol=1e-10, atol=1e-12)

    def test_finite_step_time_dependent_rotation_is_not_exact(self):
        errors = []
        for length in (12, 120):
            x, t = fixture(length)
            c, s = np.cos(.8*t), np.sin(.8*t)
            q = np.zeros((length, 3, 3))
            q[:, 0, 0] = q[:, 1, 1] = c
            q[:, 0, 1], q[:, 1, 0], q[:, 2, 2] = -s, s, 1
            changed = np.einsum('tij,nktj->nkti', q, x)
            if length == 12:
                self.assertGreater(float(np.max(np.abs(scalar(x)-scalar(changed)))), 1e-5)
            before = pathline_velocity_gradient_scalar_sequences_3d(torch.from_numpy(x), sample_times=torch.from_numpy(t))
            after = pathline_velocity_gradient_scalar_sequences_3d(torch.from_numpy(changed), sample_times=torch.from_numpy(t))
            errors.append(float(torch.max(torch.abs(before[:, 1:-1, 0]-after[:, 1:-1, 0]))))
        self.assertGreater(errors[0], 20*errors[1])

    def test_scalar_depends_on_slice_context_and_fourth_coordinate_frame(self):
        x, _ = fixture()
        self.assertEqual(float(scalar(x[:1])[0, 0]), 0.0)
        self.assertGreater(float(np.max(np.abs(scalar(x)[:4]-scalar(x[:4])))), 1e-4)
        changed = x.copy()
        changed[0, 1, 3, 1] += .07
        self.assertGreater(float(np.max(np.abs(scalar(x)-scalar(changed)))), 1e-4)
        changed = x.copy()
        changed[:, :, 4:] += 7
        np.testing.assert_array_equal(scalar(x), scalar(changed))


if __name__ == '__main__':
    unittest.main()
