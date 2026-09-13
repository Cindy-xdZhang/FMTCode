import netCDF4
import numpy as np

from experiments.JHTDB_DualFormat import export_netcdf
from experiments.Download_JHTDB_Channel import sha
from experiments.Download_JHTDB_VTK import array_sha, axes_for
from FLowUtils.flowDatasetUtils.JHTDB_VTK import write_frame


def test_entire_noncubic_unsteady_netcdf_matches_vtk(tmp_path):
    c = dict(name='channel', dataset_name='channel', spatial_method='lag8', temporal_method='pchip',
             x_range=[3., 3.6], y_range=[-.9, -.3], z_range=[.2, .8],
             x_res=3, y_res=4, z_res=5, time_start=1., time_end=1.013, time_res=3)
    axes = axes_for(c)
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    records = []
    for i, t in enumerate(np.linspace(1., 1.013, 3)):
        a = np.stack([x+t, y+10*t, z-20*t], axis=-1).astype(np.float32)
        path = tmp_path / f'channel_{i:03d}.vtk'
        write_frame(path, a, axes, t)
        records.append(dict(index=i, time=float(t), file=path.name, sha256=sha(path), velocity_sha256=array_sha(a)))
    result = export_netcdf(c, tmp_path, dict(version='test', frames=records))
    assert result['all_values_equal_vtk'] and result['frames'] == 3
    with netCDF4.Dataset(tmp_path / result['file']) as ds:
        assert ds['u'].shape == (3, 5, 4, 3)
        assert ds['u'].filters()['zlib']
        np.testing.assert_array_equal(ds['w'][2], (z-20*1.013).astype(np.float32))
