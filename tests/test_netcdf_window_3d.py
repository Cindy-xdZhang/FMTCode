import sys
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.NetCDF_window_3D import (
    inspect_netcdf_3d, interior_time_indices, load_netcdf_window_3d, resolve_time_indices,
)


def make_field(path):
    with nc.Dataset(path, "w") as dataset:
        for name, size in (("tdim", 20), ("xdim", 8), ("ydim", 6), ("zdim", 4)):
            dataset.createDimension(name, size)
        for name, dim, values in (
            ("x", "xdim", np.linspace(-2, 2, 8)),
            ("y", "ydim", np.linspace(-1, 1, 6)),
            ("z", "zdim", np.linspace(0, 3, 4)),
        ):
            dataset.createVariable(name, "f4", (dim,))[:] = values
        # Deliberately leave tdim coordinate unwritten/masked to test index fallback.
        dataset.createVariable("tdim", "f4", ("tdim",), fill_value=-999.0)
        shape = (20, 8, 4, 6)  # non-canonical [t,x,z,y]
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        for name, shift in (("u", 0), ("v", 10000), ("w", 20000)):
            dataset.createVariable(name, "f4", ("tdim", "xdim", "zdim", "ydim"))[:] = base + shift
    return base


def test_window_and_schedule():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "field.nc"
        base = make_field(path)
        info = inspect_netcdf_3d(path)
        assert info["time_source"] == "index"
        np.testing.assert_array_equal(interior_time_indices(100, 10, required_future_frames=5),
                                      [20, 28, 35, 43, 51, 58, 66, 74, 81, 89])
        field, metadata = load_netcdf_window_3d(path, 5, 4, max_spatial_dim=4)
        assert field.field.shape == (4, 4, 3, 4, 3)
        # Loaded x index 4, z index 2, y index 2 at source time 6.
        assert field.field[1, 2, 1, 2, 1] == base[6, 4, 2, 2] + 10000
        assert metadata["spatial_strides"] == {"x": 2, "y": 2, "z": 1}
        assert field.tmin == 0 and field.tmax == 3


def test_frozen_time_schedule_validation():
    expected = np.array([20, 28, 35, 43, 51, 58, 66, 74, 81, 89])
    np.testing.assert_array_equal(
        resolve_time_indices(100, 10, 0.2, 0.9, 5, fixed_indices=expected), expected
    )
    for invalid in ([20] * 10, [20, 28, 35, 43, 51, 58, 66, 74, 81, 96],
                    [20, 28, 35, 43, 51, 58, 66, 74, 81, 89.5]):
        try:
            resolve_time_indices(100, 10, 0.2, 0.9, 5, fixed_indices=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid fixed schedule was accepted: {invalid}")


if __name__ == "__main__":
    test_window_and_schedule()
    test_frozen_time_schedule_validation()
    print("NETCDF 3D WINDOW TEST PASSED")
