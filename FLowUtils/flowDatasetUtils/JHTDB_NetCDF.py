"""Read fixed, uniformly sampled JHTDB getData output into an FMT field."""
from pathlib import Path

import netCDF4
import numpy as np


def load_jhtdb_netcdf(path: str | Path, start: int = 0, stop: int | None = None):
    """Load [start, stop); default includes every frame, including the last.

    This reader is for Download_JHTDB_Channel output, not native channel cutouts
    with a nonuniform wall-normal grid or a time-dependent spatial frame.
    """
    from FLowUtils.VectorField3d import UnsteadyVectorField3D

    with netCDF4.Dataset(path) as ds:
        n = len(ds.dimensions['time'])
        stop = n if stop is None else stop
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= n:
            raise ValueError('Require 0 <= start < stop <= number of frames')
        axes = [np.asarray(ds.variables[a][:], dtype=np.float64) for a in 'xyz']
        times = np.asarray(ds.variables['time'][start:stop], dtype=np.float64)
        for a in [*axes, times]:
            if not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
                raise ValueError('Coordinates must be finite and strictly increasing')
            if len(a) > 2 and not np.allclose(np.diff(a), a[1]-a[0], rtol=1e-8, atol=1e-12):
                raise ValueError('FMT fields require uniform coordinates and times')
        for name in 'uvw':
            if ds.variables[name].dimensions != ('time', 'z', 'y', 'x'):
                raise ValueError('Expected velocity dimensions (time,z,y,x)')
        arr = np.stack([np.ma.filled(ds.variables[a][start:stop], np.nan) for a in 'uvw'], axis=-1).astype(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError('Missing or invalid saved velocities')
    obj = UnsteadyVectorField3D(len(axes[0]), len(axes[1]), len(axes[2]), len(times),
                               [float(a[0]) for a in axes], [float(a[-1]) for a in axes],
                               float(times[0]), float(times[-1]))
    obj.field = arr
    obj.jhtdb_times = times
    obj.jhtdb_coordinates = dict(zip('xyz', axes))
    return obj
