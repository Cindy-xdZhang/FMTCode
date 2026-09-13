"""Coordinate-coded responses detect permutations independently of packing code."""
import numpy as np
import pytest
from FLowUtils.flowDatasetUtils.JHTDB_Lodader import JHTDBLoader


@pytest.fixture
def loader(monkeypatch):
    import givernylocal.turbulence_toolkit as toolkit
    obj = JHTDBLoader(auth_token='unit-test', max_query_points=5, max_retries=1)
    monkeypatch.setattr(obj, '_get_or_create_dataset', lambda _: object())
    def response(dataset, variable, time, temporal, spatial, operator, points, **kwargs):
        return [points + np.array([10, 20, 30]) * time]
    monkeypatch.setattr(toolkit, 'getData', response)
    return obj


def test_3d_non_cubic_order(loader):
    a = loader.load_3d_unsteadyFlow('channel', 'velocity', 'pchip', 'lag8', 'field',
                                   2, 3, 4, 1., 2, 1.1, (3, 3.3), (-.9, -.6), (.2, .5), return_array=True)
    assert a.shape == (2, 4, 3, 2, 3)
    for t, time in enumerate([1., 1.1]):
        for iz, z in enumerate(np.linspace(.2, .5, 4)):
            for iy, y in enumerate(np.linspace(-.9, -.6, 3)):
                for ix, x in enumerate([3., 3.3]):
                    np.testing.assert_allclose(a[t, iz, iy, ix], np.array([x,y,z])+np.array([10,20,30])*time)


@pytest.mark.parametrize('plane,components', [('xy',[0,1]), ('xz',[0,2]), ('yz',[1,2])])
def test_2d_components_and_order(loader, plane, components):
    a = loader.load_2d_unsteadyFlow('synthetic', plane, 2, 3, .25, (0., .1), (.5, .7), 1., 1.1, 2, return_array=True)
    assert a.shape == (2,3,2,2)
    for t,time in enumerate([1.,1.1]):
        for j,b in enumerate([.5,.6,.7]):
            for i,x in enumerate([0.,.1]):
                expected=np.array([x,b])+np.array([10,20,30])[components]*time
                np.testing.assert_allclose(a[t,j,i],expected)


def test_time_none_and_integer_count():
    times,_=JHTDBLoader._generate_times('none', 0, 32, 31*.0065)
    assert len(times)==32
    np.testing.assert_allclose(np.diff(times),.0065)
    times,_=JHTDBLoader._generate_times('none',1,3,5,True)
    np.testing.assert_array_equal(times,[1,3,5])


@pytest.mark.parametrize('args', [('pchip',0,0,1),('pchip',0,2,None),('none',0,1,1),('none',0,3,1,True)])
def test_invalid_times(args):
    with pytest.raises(ValueError):
        JHTDBLoader._generate_times(*args)


@pytest.mark.parametrize('response', [np.zeros((1,6)), np.full((2,3),np.nan), np.full((2,3),-999.9)])
def test_invalid_response(loader,monkeypatch,response):
    import givernylocal.turbulence_toolkit as toolkit
    monkeypatch.setattr(toolkit,'getData',lambda *a,**k:[response])
    with pytest.raises(RuntimeError):
        loader._query_getData('channel','velocity',[0.],'none','lag8','field',np.zeros((2,3)))


def test_channel_off_grid_requires_interpolation(loader):
    with pytest.raises(ValueError,match='pchip'):
        loader._query_getData('channel','velocity',[1.],'none','lag8','field',np.zeros((2,3)))


def test_reject_scalar(loader):
    with pytest.raises(ValueError,match='velocity'):
        loader._query_getData('channel','pressure',[0.],'none','lag8','field',np.zeros((2,3)))


def test_3d_fmt_object(loader):
    pytest.importorskip('torch')
    pytest.importorskip('numba')
    obj=loader.load_3d_unsteadyFlow('channel','velocity','pchip','lag8','field',
                                   2,3,4,1.,2,1.1,(3.,3.3),(-.9,-.6),(.2,.5))
    assert obj.field.shape==(2,4,3,2,3)
    assert obj.time_steps==2
    np.testing.assert_allclose(obj.jhtdb_times,[1.,1.1])
    np.testing.assert_allclose(obj.get_vector(3.15,-.75,.35,1.05),
                               np.array([3.15,-.75,.35])+np.array([10,20,30])*1.05,rtol=1e-6)


def test_2d_fmt_object(loader):
    pytest.importorskip('torch')
    pytest.importorskip('numba')
    pytest.importorskip('typeguard')
    obj=loader.load_2d_unsteadyFlow('channel','xz',2,3,-.75,(3.,3.3),(.2,.5),1.,1.1,2)
    assert obj.field.shape==(2,3,2,2)
    assert obj.jhtdb_plane=='xz'
    np.testing.assert_allclose(obj.get_vector(3.15,.35,1.05),[3.15+10*1.05,.35+30*1.05],rtol=1e-6)


def test_single_snapshot_default_and_channel_offgrid(loader,monkeypatch):
    import givernylocal.turbulence_toolkit as toolkit
    seen=[]
    def response(dataset,variable,time,temporal,spatial,operator,points,**kwargs):
        seen.append(temporal)
        return [np.zeros((len(points),3))]
    monkeypatch.setattr(toolkit,'getData',response)
    for dataset in ['channel5200','channel']:
        loader.load_2d_unsteadyFlow(dataset,'xz',1,1,-.7,(3.,3.),(.2,.2),1.,1.,1,return_array=True)
    assert seen==['none','pchip']
