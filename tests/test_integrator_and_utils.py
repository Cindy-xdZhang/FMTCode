"""Tests for the CPU integrator semantics and small utility fixes
(docs/code_review_2026-08-16.md A6 / B).

Run:  python tests/test_integrator_and_utils.py   (from the repo root)
"""
import logging
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import FLowUtils.flowlineIntegral as fli  # noqa: E402
from FLowUtils.VectorField2d import UnsteadyVectorField2D  # noqa: E402
from DeepUtils.utils.stable_hash import stable_hash  # noqa: E402
from DeepUtils.utils.config import EasyConfig  # noqa: E402

# Force the CPU backend so these tests pin the CPU-fallback semantics even on GPU boxes.
fli.USE_CUDA_PATHLINE = False


def _uniform_field(u_val=1.0, v_val=0.0, T=5, Y=16, X=16,
                   dom_min=(0.0, 0.0, 0.0), dom_max=(1.0, 1.0, 10.0)):
    vf = UnsteadyVectorField2D(X, Y, T, list(dom_min), list(dom_max))
    f = np.zeros((T, Y, X, 2)); f[..., 0] = u_val; f[..., 1] = v_val
    vf.field = f
    return vf


def test_method_name_case_insensitive():
    vf = _uniform_field()
    for m in ["euler", "Euler", "EULER", "rk4", "RK4"]:
        path = fli.pathline_integration_one_direction_2D(
            vf, np.array([0.2, 0.5, 0.0]), 0.0, 0.5, 0.05, 100, m)
        assert len(path) > 1, m
    pos, valid = fli.batch_pathlineCross_integration_2D_auto(
        np.array([[0.5, 0.5]]), vf, 0.0, 0.5, 0.05, 12, 0.01, method="euler")
    assert int(valid.min()) > 1
    print("ok: euler/Euler/EULER all accepted on the CPU path (used to raise ValueError)")


def test_no_out_of_domain_points_recorded():
    """u=1 pushes the particle out of the right edge; every recorded point must stay
    inside the domain (the old code appended one out-of-domain point)."""
    vf = _uniform_field()
    pos, valid = fli.batch_pathlineCross_integration_2D_auto(
        np.array([[0.9, 0.5]]), vf, 0.0, 5.0, 0.01, 100, 0.005, method="rk4")
    for i in range(pos.shape[0]):
        n = int(valid[i])
        assert n < 100  # line truncated by the boundary, not by max_steps
        xs = pos[i, :n, 0]
        assert float(xs.max()) <= 1.0 + 1e-9, \
            f"recorded out-of-domain x={float(xs.max())}"
    print("ok: truncated lines record no out-of-domain point (CPU matches CUDA rule)")


def test_t_target_clipped_with_warning():
    vf = _uniform_field(u_val=0.01)
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _H(); logging.getLogger().addHandler(h)
    try:
        pos, valid = fli.batch_pathlineCross_integration_2D_auto(
            np.array([[0.5, 0.5]]), vf, 9.5, 99.0, 0.05, 200, 0.005, method="rk4")
    finally:
        logging.getLogger().removeHandler(h)
    assert any("clipped" in m for m in records), "expected a loud t_target clip warning"
    n = int(valid[0])
    assert float(pos[0, :n, 2].max()) <= 10.0 + 1e-6, "recorded time beyond tmax"
    print("ok: out-of-range t_target clipped for the CPU backend, with warning")


def test_stable_hash_ndarray_and_set():
    a = np.arange(20000, dtype=np.float64)
    b = a.copy(); b[5000] += 1e-9
    assert stable_hash(a) != stable_hash(b), \
        "large arrays differing mid-way must hash differently (old repr truncation)"
    assert stable_hash(a) == stable_hash(a.copy())
    assert stable_hash(a) != stable_hash(a.astype(np.float32))
    s1 = set(["alpha", "beta", "gamma", "delta"])
    s2 = set(list(s1)[::-1])
    assert stable_hash(s1) == stable_hash(s2)
    print("ok: stable_hash handles ndarray content and set order")


def test_easyconfig_empty_yaml():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "empty.yaml")
        open(p, "w").close()
        cfg = EasyConfig(); cfg.load(p)
        assert len(cfg) == 0
    print("ok: EasyConfig tolerates an empty yaml file")


def test_pose_runs_on_cpu():
    from pnn.models.point_nn import PosE_Initial, PosE_Geo
    out = PosE_Initial(3, 24, 1000, 19)(torch.randn(2, 3, 10))
    assert out.device.type == "cpu" and out.shape == (2, 24, 10)
    out2 = PosE_Geo(3, 24, 1000, 19)(torch.randn(2, 3, 10, 5), torch.randn(2, 24, 10, 5))
    assert out2.device.type == "cpu" and out2.shape == (2, 24, 10, 5)
    print("ok: pnn PosE modules run on CPU (was hardcoded .cuda())")


def test_gen_starts_unpacks_3elem_boundary():
    from pnn.libs.parallel_flows import gen_starts
    vf = _uniform_field()
    starts = gen_starts(vf, 4.0, device="cpu")
    assert starts.ndim == 2 and starts.shape[1] == 2 and starts.shape[0] > 0
    print("ok: gen_starts accepts the length-3 domain boundary")


if __name__ == "__main__":
    test_method_name_case_insensitive()
    test_no_out_of_domain_points_recorded()
    test_t_target_clipped_with_warning()
    test_stable_hash_ndarray_and_set()
    test_easyconfig_empty_yaml()
    test_pose_runs_on_cpu()
    test_gen_starts_unpacks_3elem_boundary()
    print("ALL INTEGRATOR/UTILS TESTS PASSED")
