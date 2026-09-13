import json
from pathlib import Path
import time

import numpy as np
import pytest

from FMT_Utils.Task2GeometryRecompute import integration_window, RecomputeJobs


def test_window_accepts_arbitrary_start_crosses_old_split_and_clips_at_source_end():
    times = np.linspace(0, 15, 151).astype(np.float32)
    start, count = integration_window(times, 12.25, .1, 40)
    assert times[start] <= 12.25 < times[start+1]
    assert times[start+count-1] == 15
    assert integration_window(times, 14.5, .1, 100)[1] == 6
    with pytest.raises(ValueError, match="起始"):
        integration_window(times, 15, .1, 2)


def test_float32_window_uses_real_source_endpoint():
    times = np.linspace(7.5, 15., 76).astype(np.float32)
    start, count = integration_window(times, 14.23, .0125, 640)
    assert times[start] <= 14.23 and times[start+count-1] == 15


def test_clipped_integrator_has_exact_short_tail_and_time_resampling():
    from FMT_Utils.Task2GeometryRecompute import integrate_clipped_primitives
    from FLowUtils.VectorField3d import UnsteadyVectorField3D
    field = UnsteadyVectorField3D(4,4,4,6,np.zeros(3),np.ones(3)*10,14.,15.)
    field.field = np.zeros((6,4,4,4,3),dtype=np.float32)
    field.field[...,0] = 1.
    seeds = np.array([[2.,2.,2.],[3.,3.,3.]])
    for start, dt in [(14.23,.2),(14.99,.2),(14.1,.025)]:
        paths, valid = integrate_clipped_primitives(field,seeds,start,15.,dt,32,.1)
        assert valid.all() and paths.shape == (2,7,32,4)
        assert np.allclose(paths[:,0,-1,0],seeds[:,0]+15-start,atol=1e-4)
        assert np.allclose(paths[:,:,0,3],start) and np.all(paths[:,:,-1,3]==15.)
        assert np.allclose(np.diff(paths[0,0,:,3]),(15-start)/31,atol=2e-6)


@pytest.mark.parametrize("dt,steps", [(0, 48), (-1, 48), (float('nan'), 48), (.1, 2.5)])
def test_window_rejects_invalid_dt_and_steps(dt, steps):
    with pytest.raises(ValueError):
        integration_window(np.arange(20), 2, dt, steps)


def test_real_integrator_changes_geometry_with_dt_and_steps():
    from FLowUtils.VectorField3d import UnsteadyVectorField3D
    from FMT_Utils.FMT_3D_pipeline import integrate_cross_primitives_3d
    field = UnsteadyVectorField3D(4, 4, 4, 5, np.zeros(3), np.ones(3) * 10, 0., 4.)
    field.field = np.zeros((5, 4, 4, 4, 3), dtype=np.float32)
    field.field[..., 0] = 1.
    seeds = np.array([[2., 2., 2.], [3., 3., 3.]])
    for dt, steps in ((.02, 32), (.04, 32), (.02, 64)):
        paths, valid, _ = integrate_cross_primitives_3d(field, seeds, 0., dt, steps, 32, .1, method="RK4")
        assert paths.shape == (2, 7, 32, 4) and valid.all()
        assert np.allclose(paths[:, 0, -1, 0], seeds[:, 0] + dt * steps, atol=1e-4)
        assert np.allclose(paths[:, 0, -1, 3], dt * steps, atol=1e-5)


def test_failed_job_is_recorded_and_next_job_can_start(tmp_path, monkeypatch):
    import FMT_Utils.Task2GeometryRecompute as module
    monkeypatch.setattr(module, "catalog", lambda c: {"test": {"available": True}})
    def fail(*args):
        raise ValueError("source window unavailable")
    monkeypatch.setattr(module, "recompute", fail)
    manager = RecomputeJobs({"output_root": str(tmp_path), "recompute": {"max_integration_steps": 100}})
    try:
        key = manager.start("test", .1, 48)
        for _ in range(100):
            if manager.snapshot(key)["status"] != "running":
                break
            time.sleep(.01)
        job = manager.snapshot(key)
        assert job["status"] == "failed" and "source window unavailable" in job["message"]
        assert json.loads((Path(job["run_dir"]) / "job.json").read_text())["status"] == "failed"
        next_key = manager.start("test", .1, 48)
        assert next_key != key
    finally:
        manager.executor.shutdown(wait=True)


def test_non_frame_start_interpolates_time_dependent_velocity():
    from FMT_Utils.Task2GeometryRecompute import integrate_clipped_primitives
    from FLowUtils.VectorField3d import UnsteadyVectorField3D
    field = UnsteadyVectorField3D(4,4,4,6,np.zeros(3),np.ones(3)*10,0.,1.5)
    field.field = np.zeros((6,4,4,4,3),dtype=np.float32)
    field.field[...,0] = np.linspace(0,1.5,6)[:,None,None,None]
    paths, valid = integrate_clipped_primitives(field,np.array([[2.,2.,2.]]),.23,1.5,.2,32,.1)
    assert valid.all()
    assert np.isclose(paths[0,0,-1,0],2+.5*(1.5**2-.23**2),atol=1e-5)
