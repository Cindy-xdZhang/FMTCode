import numpy as np
from experiments.Visualize_JHTDB_Structure import q_criterion


def test_rotation_and_strain_on_physical_noncubic_grid():
    axes = [np.linspace(3, 3.6, 9), np.linspace(-.9, -.3, 11), np.linspace(.2, .8, 13)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    rotation = np.stack([-2*y, 2*x, z*0], axis=-1)
    np.testing.assert_allclose(q_criterion(rotation, axes), 4., atol=1e-11)
    strain = np.stack([3*x, -3*y, z*0], axis=-1)
    np.testing.assert_allclose(q_criterion(strain, axes), -9., atol=1e-11)
