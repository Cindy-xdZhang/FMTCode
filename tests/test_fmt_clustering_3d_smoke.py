from pathlib import Path
import sys
import tempfile

import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DeepUtils.utils import EasyConfig
from FMT_Clustering_3D import run
from FMT_Utils.FMT_3D_pipeline import load_vector_field_3d


def make_synthetic_field(path):
    t_count, z_count, y_count, x_count = 12, 8, 10, 10
    x = np.linspace(-1, 1, x_count); y = np.linspace(-1, 1, y_count)
    z = np.linspace(-1, 1, z_count); times = np.linspace(0, 1, t_count)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    field = np.empty((t_count, z_count, y_count, x_count, 3), np.float32)
    for index, time in enumerate(times):
        strength = np.exp(-4 * (xx * xx + yy * yy)) * (1 + 0.15 * np.sin(2 * np.pi * time))
        field[index, ..., 0] = -strength * yy
        field[index, ..., 1] = strength * xx
        field[index, ..., 2] = 0.03 * np.sin(np.pi * zz)
    np.savez(path, field=field, domain_min=[-1, -1, -1], domain_max=[1, 1, 1], tmin=0, tmax=1)


def test_netcdf_axis_reordering(path):
    # Deliberately store components as [time, xdim, zdim, ydim].
    with nc.Dataset(path, "w") as ds:
        for name, size in (("time", 3), ("xdim", 4), ("ydim", 5), ("zdim", 2)):
            ds.createDimension(name, size)
        for name, dim, values in (
            ("time", "time", np.linspace(2, 4, 3)),
            ("x", "xdim", np.linspace(-2, 2, 4)),
            ("y", "ydim", np.linspace(-3, 3, 5)),
            ("z", "zdim", np.linspace(-1, 1, 2)),
        ):
            ds.createVariable(name, "f4", (dim,))[:] = values
        shape = (3, 4, 2, 5)
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        for component, shift in (("u", 0), ("v", 1000), ("w", 2000)):
            ds.createVariable(component, "f4", ("time", "xdim", "zdim", "ydim"))[:] = base + shift
    field = load_vector_field_3d(path)
    assert field.field.shape == (3, 2, 5, 4, 3)
    assert field.field[1, 0, 3, 2, 1] == base[1, 2, 0, 3] + 1000
    np.testing.assert_allclose(field.domainMinBoundary, [-2, -3, -1])
    np.testing.assert_allclose(field.domainMaxBoundary, [2, 3, 1])
    assert field.tmin == 2 and field.tmax == 4


def smoke_test():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        field_path = tmp / "synthetic_vortex.npz"
        make_synthetic_field(field_path)
        test_netcdf_axis_reordering(tmp / "axis_order.nc")
        cfg = EasyConfig()
        cfg.update({
            "seed": 7,
            "dataset": {"path": str(field_path), "grid_shape": [6, 6, 5],
                        "boundary_fraction": 0.12, "seed_time_ratio": 0.1},
            "pathlines": {"method": "RK4", "dt_scale": 0.4, "integration_steps": 8,
                          "sampled_steps": 7, "offset_grid_scale": 0.4, "chunk_size": 256},
            "encoder": {"num_freq": 3, "mode": "gram", "include_chirality": True,
                        "neighbor_pool": "sort",
                        "neighbor_weight": 0.5},
            "clustering": {"classes": 2, "n_init": 5},
            "visualization": {"max_pathlines": 8},
            "output": {"dir": str(tmp / "outputs")},
        })
        output = run(cfg)
        with np.load(output / "clustering_result.npz") as data:
            assert data["features"].shape[0] == data["labels"].shape[0] > 10
            assert np.unique(data["labels"]).size == 2
        for name in ("clusters_3d.png", "clusters_projections.png", "clusters_xy_slices.png",
                     "cluster_pathlines.png"):
            assert (output / name).stat().st_size > 1000


if __name__ == "__main__":
    smoke_test()
    print("3D CLUSTERING SMOKE TEST PASSED")
