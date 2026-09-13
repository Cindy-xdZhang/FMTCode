import numpy as np
import netCDF4
import pytest
from FLowUtils.flowDatasetUtils.JHTDB_NetCDF import load_jhtdb_netcdf


@pytest.fixture
def saved(tmp_path):
    file=tmp_path/'velocity.nc'
    with netCDF4.Dataset(file,'w') as ds:
        for key,coords in dict(time=[1.,1.1,1.2],x=[3.,3.3],y=[-.9,-.8,-.7],z=[.2,.3,.4,.5]).items():
            ds.createDimension(key,len(coords))
            ds.createVariable(key,'f8',(key,))[:]=coords
        for i,a in enumerate('uvw'):
            v=ds.createVariable(a,'f4',('time','z','y','x'))
            for t in range(3):
                v[t]=10*t+i
    return file


def test_default_all_and_single_frame(saved):
    pytest.importorskip('torch')
    pytest.importorskip('numba')
    field=load_jhtdb_netcdf(saved)
    assert field.field.shape==(3,4,3,2,3)
    np.testing.assert_array_equal(field.field[-1,0,0,0],[20,21,22])
    np.testing.assert_allclose(field.jhtdb_times,[1.,1.1,1.2])
    one=load_jhtdb_netcdf(saved,1,2)
    assert one.field.shape==(1,4,3,2,3)
    np.testing.assert_array_equal(one.field[0,0,0,0],[10,11,12])


def test_nonuniform_rejected(saved):
    pytest.importorskip('torch')
    pytest.importorskip('numba')
    with netCDF4.Dataset(saved,'a') as ds:
        ds.variables['y'][:]=[-.9,-.89,-.7]
    with pytest.raises(ValueError,match='uniform'):
        load_jhtdb_netcdf(saved)
