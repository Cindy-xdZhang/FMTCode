"""Analytic physical checks, leakage checks and all-arm executable smoke test."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FMT_Utils.FlowMapData_3D import (select_windows, material_offsets, query_offsets,
    integrate, build_window, affine_query, trajectory_metrics, sha256, write_json, randomized_queries)
from FMT_Utils.FlowMapModels_3D import raw_features, FlowMapDecoder, compose
from experiments.Build_Task678_FlowMap_1_1 import make_features, DEFAULT_CONFIG, separated_seed_grid


def analytic_field(shift=0.):
    f = UnsteadyVectorField3D(17, 17, 17, 9, [-4, -4, -4], [4, 4, 4], 0., 1.)
    f.field = np.zeros((9, 17, 17, 17, 3), np.float32)
    for i, t in enumerate(np.linspace(0, 1, 9)):
        f.field[i, ..., 0] = .1 + .04 * (t + shift)
        f.field[i, ..., 1] = -.02 * (t + shift)
    return f


class FlowMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.field = analytic_field()
        cls.centers = np.random.default_rng(17).uniform(-1, 1, (12, 3))
        cls.data, cls.audit = build_window(cls.field, cls.centers, .1, .5)

    def test_material_layout_and_no_hidden_duplicates(self):
        o = material_offsets()
        self.assertEqual(o.shape, (31, 3))
        self.assertEqual(len(np.unique(o, axis=0)), 31)
        self.assertGreater(np.linalg.norm(o[:, None] - query_offsets()[None], axis=-1).min(), .1)
        self.assertGreater(self.audit["task7_minimum_seed_gap"], .1)
        q = randomized_queries(10, 91)
        self.assertTrue((np.abs(q).sum(-1) < 1).all())
        self.assertFalse(np.array_equal(q[0], q[1]))
        self.assertFalse(np.array_equal(q, randomized_queries(10, 92)))

    def test_missing_frame_and_split_isolation(self):
        t = np.arange(120, dtype=float)
        available = set(range(120)) - set(range(23, 46))
        starts = select_windows(t, available, count=8)
        intervals = [set(range(s, s + 9)) for s in starts]
        self.assertTrue(all(a <= available for a in intervals))
        self.assertTrue(all(not intervals[i] & intervals[j] for i in range(8) for j in range(i)))
        with self.assertRaises(ValueError):
            select_windows(t, available, count=8, minimum_time=100.)

    def test_thin_axis_grid_retains_context_separation(self):
        grid, counts, separation = separated_seed_grid(np.array([0., 0., 0.]), np.array([50., 20., 3.]), .1, 8)
        self.assertEqual(counts, [8, 8, 3])
        self.assertGreater(separation, 1.)
        self.assertEqual(len(grid), 192)

    def test_unsteady_shear_deformation_and_composition(self):
        from experiments.Run_Task678_FlowMap_1_1 import interpolation
        f = analytic_field()
        xx = np.linspace(-4, 4, 17)
        f.field[:] = 0.
        for i, t in enumerate(np.linspace(0, 1, 9)):
            f.field[i, ..., 1] = (.1 + .1*t) * xx[None, None, :]
        p, valid = integrate(f, np.array([[.4, .2, 0.]]), 0., 1., 124)
        self.assertTrue(valid.all())
        t = np.linspace(0, 1, 125)
        np.testing.assert_allclose(p[0, :, 1], .2 + .4*(.1*t + .05*t*t), atol=2e-8)
        data, _ = build_window(f, self.centers[:4], .1, .5)
        pred = interpolation(data, "Task8")
        error = trajectory_metrics(pred, data["target_long"], data["radius0"])
        self.assertLess(error["position_nrmse"], 1e-4)
        self.assertLess(error["pair_distance_error"], 1e-5)

    def test_physical_unsteady_rk4_and_full_time(self):
        seeds = np.array([[.1, .2, .3], [-.1, .5, 1.]])
        paths, valid = integrate(self.field, seeds, 0., 1., 124)
        t = np.linspace(0, 1, 125)
        expected = seeds[:, None] + np.stack([.1*t + .02*t*t, -.01*t*t, np.zeros_like(t)], -1)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(paths, expected, atol=2e-8)
        self.assertEqual(self.data["support0"].shape[2], 32)
        self.assertEqual(self.data["target_long"].shape[2], 125)

    def test_internal_boundary_crossing_is_rejected(self):
        p, valid = integrate(self.field, np.array([[3.99, 0., 0.]]), 0., 1., 20)
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(p).any())

    def test_affine_interpolation_and_metrics(self):
        data = self.data
        queries = (data["target0"][:, :, 0] - data["origin0"][:, None]) / data["radius0"][:, None, None]
        p = affine_query(data["support0"], data["origin0"], data["radius0"], queries, np.linspace(0, 1, 63))
        # Temporal interpolation only has second-order error between 32 support times.
        self.assertLess(trajectory_metrics(p, data["target0"], data["radius0"])["position_nrmse"], 2e-5)
        self.assertEqual(trajectory_metrics(data["target_long"], data["target_long"], data["radius0"])["position_nrmse"], 0.)
        swapped = data["target0"][:, ::-1]
        self.assertGreater(trajectory_metrics(swapped, data["target0"], data["radius0"])["position_nrmse"], .1)

    def test_hidden_targets_do_not_enter_features(self):
        arms = ["aivd1w3_dft", "fmt_all", "largeNeighbor"]
        original = make_features(self.data, arms)
        changed = {k: v.copy() for k, v in self.data.items()}
        for key in ("target0", "target1", "target_long"):
            changed[key][:] = 987654.
        again = make_features(changed, arms)
        for key in original:
            np.testing.assert_array_equal(original[key], again[key])
        changed["support0"] += 12.
        # Task7 IVD context cannot use the hidden-center support group.
        context = make_features(changed, ["aivd1w3_dft"])["aivd1w3_dft__context"]
        np.testing.assert_array_equal(original["aivd1w3_dft__context"], context)

    def test_capacity_and_initial_condition_and_context_permutation(self):
        model = FlowMapDecoder(8, 16)
        token, geometry = torch.randn(3, 6, 8), torch.randn(3, 6, 4)
        q, scale = torch.randn(3, 3), torch.ones(3, 2)
        torch.testing.assert_close(model(token, geometry, q, torch.zeros(3), scale), q)
        a = model(token, geometry, q, torch.ones(3), scale)
        b = model(token[:, [5, 0, 4, 1, 3, 2]], geometry[:, [5, 0, 4, 1, 3, 2]], q, torch.ones(3), scale)
        torch.testing.assert_close(a, b)

    def test_composition_uses_predicted_arrival_and_second_origin(self):
        class Translation(torch.nn.Module):
            def forward(self, tokens, geometry, query, tau, scales):
                v = torch.zeros_like(query)
                v[:, 0] = tokens[:, 0, 0]
                return query + tau[:, None] * v
        arrays = dict(tokens=np.array([[[.1]], [[.2]]], np.float32), geometry=np.zeros((2, 1, 4), np.float32),
                      query=np.zeros((2, 8, 3), np.float32), scales=np.zeros((2, 2), np.float32),
                      target=np.full((2, 8, 5, 3), 999., np.float32),
                      origin=np.array([[0., 0., 0.], [.7, 0., 0.]], np.float32),
                      radius=np.ones(2, np.float32), duration=np.ones(2, np.float32))
        prediction, info = compose(Translation(), arrays, "cpu")
        np.testing.assert_allclose(prediction[:, :, -1, 0], .3, atol=1e-6)
        self.assertEqual(info["second_query_uses"], "predicted_first_stage_endpoint")
        self.assertEqual(prediction.shape, (1, 8, 9, 3))


def run_smoke(output):
    """Exercise all 12 feature methods, both learned tasks and composition."""
    from experiments.Run_Task678_FlowMap_1_1 import run
    output = Path(output)
    spec = json.loads(Path(DEFAULT_CONFIG).read_text())
    spec.update(experiment="Verify_Task678_LocalSmoke_1.1", output_root=str(output),
                datasets=["analytic_unsteady"], seeds=[9100], token_dimensions=[8])
    spec["decoder"]["optimizer_steps"] = 3
    spec["decoder"]["batch_size"] = 16
    spec["vae"]["optimizer_steps"] = 3
    spec["vae"]["batch_size"] = 16
    cfg = output / "smoke_config.json"
    write_json(cfg, spec)
    h = sha256(cfg)
    for i in range(8):
        rng = np.random.default_rng(9106 + i)
        data, audit = build_window(analytic_field(i / 8), rng.uniform(-1, 1, (12, 3)), .1, .5, query_seed=9106+i)
        role = next(k for k, ids in spec["splits"].items() if i in ids)
        path = output / "cache" / "analytic_unsteady" / f"window_{i:02d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(ordinal=i, role=role, config_sha256=h, start_index=i * 9, end_index=i * 9 + 8,
                    time_start=float(i), time_end=float(i+1), **audit)
        np.savez_compressed(path, **data, metadata_json=np.asarray(json.dumps(meta)))
        meta["cache_sha256"] = sha256(path)
        write_json(path.with_suffix(".json"), meta)
        fp = path.with_name(path.stem + "_features.npz")
        np.savez_compressed(fp, **make_features(data, spec["arms"]))
        write_json(fp.with_suffix(".json"), dict(config_sha256=h, cache_sha256=sha256(path), feature_sha256=sha256(fp)))
    write_json(output / "build" / "analytic_unsteady.json", dict(status="PASS", config_sha256=h))
    run(spec, str(cfg), "analytic_unsteady", 9100, 8, "cpu")
    completed = json.loads((output / "shards/analytic_unsteady/seed9100_width8/attempt_0/completed.json").read_text())
    assert completed["metric_records"] == 39
    assert not any(output.rglob("*.pt")) and not any(output.rglob("*.pth"))
    print("ALL-ARM SMOKE PASS: 39 evaluations; no model files", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-output":
        run_smoke(sys.argv[2])
    else:
        unittest.main()
