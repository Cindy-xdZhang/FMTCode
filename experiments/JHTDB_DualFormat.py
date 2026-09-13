"""Export and verify one complete, lossless NetCDF time series from fresh VTK frames."""
from pathlib import Path

import netCDF4
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from FLowUtils.flowDatasetUtils.JHTDB_VTK import read_frame, check_geometry
from experiments.Download_JHTDB_Channel import sha
from experiments.Download_JHTDB_VTK import axes_for, array_sha


def export_netcdf(config, directory, manifest):
    directory = Path(directory)
    axes = axes_for(config)
    records = sorted(manifest['frames'], key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(config['time_res'])):
        raise ValueError('A complete time series is required')
    times = np.linspace(config['time_start'], config['time_end'], config['time_res'])
    np.testing.assert_array_equal([r['time'] for r in records], times)
    path = directory / f'{config["name"]}.nc'
    temporary = path.with_suffix('.partial.nc')
    with netCDF4.Dataset(temporary, 'w', format='NETCDF4') as ds:
        ds.setncatts({'title': f'JHTDB {config["dataset_name"]} velocity time series',
                     'dataset': config['dataset_name'], 'version': manifest['version'],
                     'coordinate_frame': 'Fixed physical xyz; velocity components u,v,w',
                     'spatial_interpolation': config['spatial_method'],
                     'temporal_interpolation': config['temporal_method'],
                     'velocity_axis_order': 'time,z,y,x',
                     'source': 'https://turbulence.idies.jhu.edu/',
                     'Conventions': 'CF-1.10'})
        for name, values in zip(('time', 'x', 'y', 'z'), (times, *axes)):
            ds.createDimension(name, len(values))
            var = ds.createVariable(name, 'f8', (name,))
            var[:] = values
            var.axis = {'time': 'T', 'x': 'X', 'y': 'Y', 'z': 'Z'}[name]
            var.long_name = 'simulation time' if name == 'time' else f'physical {name} coordinate'
            var.units = '1'
        chunks = (1, min(16, len(axes[2])), len(axes[1]), len(axes[0]))
        velocity = [ds.createVariable(name, 'f4', ('time', 'z', 'y', 'x'),
                                     zlib=True, complevel=1, shuffle=True, chunksizes=chunks)
                    for name in 'uvw']
        for name, var in zip('xyz', velocity):
            var.long_name = f'{name} velocity component'
            var.units = '1'
        for r in records:
            source = directory / r['file']
            assert sha(source) == r['sha256']
            arr, grid = read_frame(source)
            check_geometry(grid, axes)
            assert array_sha(arr) == r['velocity_sha256']
            assert float(vtk_to_numpy(grid.GetFieldData().GetArray('TimeValue'))[0]) == r['time']
            for component, var in enumerate(velocity):
                var[r['index']] = arr[..., component]
    verify_netcdf(temporary, config, directory, records)
    temporary.replace(path)
    return {'file': path.name, 'sha256': sha(path), 'frames': len(records),
            'dimensions': ['time', 'z', 'y', 'x'], 'velocity_dtype': 'float32',
            'all_values_equal_vtk': True, 'compression': 'lossless zlib level 1'}


def verify_netcdf(path, config, directory, records):
    axes = axes_for(config)
    times = np.linspace(config['time_start'], config['time_end'], config['time_res'])
    with netCDF4.Dataset(path) as ds:
        np.testing.assert_array_equal(ds['time'][:], times)
        for name, axis in zip('xyz', axes):
            np.testing.assert_array_equal(ds[name][:], axis)
        for name in 'uvw':
            assert ds[name].dimensions == ('time', 'z', 'y', 'x')
            assert ds[name].shape == (len(times), len(axes[2]), len(axes[1]), len(axes[0]))
            assert ds[name].dtype == np.dtype('float32')
        for r in records:
            arr, _ = read_frame(Path(directory) / r['file'])
            for component, name in enumerate('uvw'):
                actual = np.ma.filled(ds[name][r['index']], np.nan)
                assert np.isfinite(actual).all()
                np.testing.assert_array_equal(actual, arr[..., component])

