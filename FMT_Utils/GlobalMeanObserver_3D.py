"""Volume-averaged translating observer from the complete source grid."""
import numpy as np


def nodal_weights(coordinates):
    x = np.asarray(coordinates, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError('Coordinates must be finite and strictly increasing.')
    widths = np.diff(x)
    return np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]


def volume_mean(values, coordinates_zyx):
    """Tensor-product trapezoidal volume mean, excluding joint missing nodes."""
    values = np.ma.asarray(values)
    valid = ~np.ma.getmaskarray(values).any(axis=-1)
    data = np.asarray(values.filled(np.nan), dtype=float)
    valid &= np.isfinite(data).all(axis=-1)
    wz, wy, wx = [nodal_weights(c) for c in coordinates_zyx]
    weights = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    if data.shape != weights.shape + (3,):
        raise ValueError('Expected a complete Z,Y,X,3 grid.')
    denominator = weights[valid].sum()
    if denominator <= 0:
        raise ValueError('No valid spatial volume.')
    mean = (data[valid] * weights[valid, None]).sum(axis=0) / denominator
    return mean, int((~valid).sum())


def source_mean_schedule(path, start_index, frame_count):
    import netCDF4 as nc
    from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate
    with nc.Dataset(path) as ds:
        dims = {a: _axis_dimension(ds, a) for a in 'tzyx'}
        coords = [_coordinate(ds, dims[a], np.arange(len(ds.dimensions[dims[a]])))
                  for a in 'zyx']
        names = next(n for n in [('u', 'v', 'w'), ('velocity_x', 'velocity_y', 'velocity_z'),
                                ('Component1', 'Component2', 'Component3')]
                     if all(k in ds.variables for k in n))
        indices = np.arange(start_index, start_index + frame_count)
        times = _coordinate(ds, dims['t'], indices)
        means, missing = [], []
        for index in indices:
            components = []
            for name in names:
                var = ds.variables[name]
                spatial = [d for d in var.dimensions if d != dims['t']]
                if set(spatial) != {dims[a] for a in 'zyx'}:
                    raise ValueError('Unexpected velocity variable dimensions.')
                data = var[tuple(int(index) if d == dims['t'] else slice(None) for d in var.dimensions)]
                components.append(np.ma.transpose(data, [spatial.index(dims[a]) for a in 'zyx']))
            mean, excluded = volume_mean(np.ma.stack(components, axis=-1), coords)
            means.append(mean); missing.append(excluded)
        info = {'mean_definition': 'Full source-grid spatial volume mean at each physical time; tensor-product trapezoidal weights',
                'mean_grid_shape_zyx': [len(c) for c in coords],
                'missing_nodes_per_frame': missing,
                'zero_velocity_nodes': 'Included; no obstacle mask inferred from zero velocity',
                'velocity_units': [getattr(ds.variables[n], 'units', 'source units (unspecified)') for n in names],
                'mean_time_indices': indices.tolist()}
    return times, np.asarray(means), info
