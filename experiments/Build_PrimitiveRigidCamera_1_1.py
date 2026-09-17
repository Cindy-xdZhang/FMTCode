"""Build and audit the analytic rigid-camera demo without training models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from FMT_Utils.PrimitiveRigidCamera_1_1 import (
    SampledPrimitive, analytic_motion, change_observer, external_observer,
    fit_angular_velocity, rotation_axis, skew, solve_camera,
)

VERSION = "Other_PrimitiveRigidCamera_1.1"


def cross_time_error(a, b):
    p, q = a.reshape(-1, 3), b.reshape(-1, 3)
    dp = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    dq = np.linalg.norm(q[:, None] - q[None, :], axis=-1)
    return float(np.max(np.abs(dp - dq)))


def observed_field_integration_check(trajectory, times, reference):
    """Independently integrate particles in the inverse-queried observed field."""
    state = np.concatenate([np.eye(3).ravel(), (trajectory(times[0])[0] - trajectory(times[0])[0][0]).ravel()])
    output = [state[9:].reshape(7, 3).copy()]

    def rhs(t, s):
        rotation, y = s[:9].reshape(3, 3), s[9:].reshape(7, 3)
        x, v = trajectory(t)
        r, rv = x - x[0], v - v[0]
        omega = fit_angular_velocity(r[1:], rv[1:])[0]
        # This analytic primitive lies in an invertible affine flow.
        matrix = np.stack([r[1], r[3]/.8, r[5]/.65], axis=1)
        matrix_dot = np.stack([rv[1], rv[3]/.8, rv[5]/.65], axis=1)
        jacobian = matrix_dot @ np.linalg.inv(matrix)
        query_lab = y @ rotation.T + x[0]
        velocity_lab = (query_lab - x[0]) @ jacobian.T + v[0]
        observer_lab = v[0] + np.cross(omega, query_lab - x[0])
        ydot = (velocity_lab - observer_lab) @ rotation
        return np.concatenate([(skew(omega) @ rotation).ravel(), ydot.ravel()])

    for left, right in zip(times[:-1], times[1:]):
        dt = (right-left)/4
        for k in range(4):
            t = left+k*dt
            k1 = rhs(t, state)
            k2 = rhs(t+dt/2, state+dt*k1/2)
            k3 = rhs(t+dt/2, state+dt*k2/2)
            k4 = rhs(t+dt, state+dt*k3)
            state += dt*(k1+2*k2+2*k3+k4)/6
        output.append(state[9:].reshape(7, 3).copy())
    return float(np.max(np.abs(np.array(output) - reference)))


def audit_case(base, changed, trajectory):
    times = base["times"]
    report = {
        "max_center_error": float(np.max(np.abs(base["observed"][:, 0]))),
        "max_orthogonality_error": float(np.max(np.abs(np.swapaxes(base["rotation"], 1, 2) @ base["rotation"] - np.eye(3)))),
        "max_determinant_error": float(np.max(np.abs(np.linalg.det(base["rotation"]) - 1))),
        "max_observer_coordinate_error_Q0_identity": float(np.max(np.abs(base["observed"] - changed["observed"]))),
        "max_cross_time_distance_error": cross_time_error(base["observed"], changed["observed"]),
        "max_same_time_distance_error": float(np.max(np.abs(
            np.linalg.norm(base["positions"][:, :, None] - base["positions"][:, None, :], axis=-1)
            - np.linalg.norm(base["observed"][:, :, None] - base["observed"][:, None, :], axis=-1)))),
        "max_design_matrix_condition": float(np.max(base["condition"])),
        "max_speed_increase": float(np.max(base["observed_speed"] - base["centered_speed"])),
        "observed_path_displacement_max": float(np.max(np.linalg.norm(base["observed"] - base["observed"][0], axis=-1))),
        "max_observed_speed": float(np.max(base["observed_speed"])),
        "max_angular_covariance_error": 0.,
    }
    errors = []
    for t, omega, other in zip(times, base["omega"], changed["omega"]):
        q, qd, _, _ = external_observer(t)
        errors.append(np.max(np.abs(skew(other) - q @ skew(omega) @ q.T - qd @ q.T)))
    report["max_angular_covariance_error"] = float(max(errors))
    report["independent_observed_field_integration_error"] = observed_field_integration_check(trajectory, times, base["observed"])
    for key in ("max_center_error", "max_orthogonality_error", "max_determinant_error",
                "max_observer_coordinate_error_Q0_identity", "max_cross_time_distance_error",
                "max_same_time_distance_error", "max_angular_covariance_error",
                "independent_observed_field_integration_error"):
        assert report[key] < 2e-7, (key, report[key])
    assert report["max_speed_increase"] < 1e-12
    return report


def json_arrays(result):
    keys = ("positions", "centered", "observed", "rotation", "omega", "centered_speed", "observed_speed")
    return {key: np.round(result[key], 7).tolist() for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / VERSION)
    parser.add_argument("--inline", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0., 4., 121)
    scenes, evidence = {}, {}
    names = {"translation": "纯平移", "rigid": "平移＋旋转", "deform": "平移＋旋转＋拉伸与剪切"}
    for scene, name in names.items():
        trajectory = lambda t, scene=scene: analytic_motion(t, scene)
        base = solve_camera(trajectory, times)
        changed = solve_camera(change_observer(trajectory), times)
        evidence[scene] = audit_case(base, changed, trajectory)
        if scene != "deform":
            assert evidence[scene]["observed_path_displacement_max"] < 2e-7
            assert evidence[scene]["max_observed_speed"] < 1e-12
        else:
            assert evidence[scene]["observed_path_displacement_max"] > .1
        scenes[scene] = {"name": name, "base": json_arrays(base), "changed": json_arrays(changed),
                         "distance_error": evidence[scene]["max_cross_time_distance_error"]}
        print(scene, json.dumps(evidence[scene]), flush=True)
    trajectory = lambda t: analytic_motion(t, "deform")
    q0 = rotation_axis([1., 2., -.5], .8, 0)[0]
    nonidentity = solve_camera(change_observer(trajectory, q0), times)
    original = solve_camera(trajectory, times)
    gauge_error = float(np.max(np.abs(nonidentity["observed"] - original["observed"] @ q0.T)))
    assert gauge_error < 2e-7
    # Finite-difference convergence is reported separately from analytic derivatives.
    finite = []
    for count in (33, 65, 129):
        nodes = np.linspace(0, 4, count)
        x = np.array([trajectory(t)[0] for t in nodes])
        xstar = np.array([change_observer(trajectory)(t)[0] for t in nodes])
        solved = solve_camera(SampledPrimitive(nodes, x), times)
        star = solve_camera(SampledPrimitive(nodes, xstar), times)
        error = float(np.max(np.abs(solved["observed"] - star["observed"])))
        finite.append(dict(samples=count, max_coordinate_error=error))
    assert finite[-1]["max_coordinate_error"] < finite[0]["max_coordinate_error"]/5
    rejected = False
    try:
        fit_angular_velocity(np.array([[1., 0, 0], [-1., 0, 0]]), np.zeros((2, 3)))
    except ValueError:
        rejected = True
    assert rejected
    sources = [ROOT / "FMT_Utils" / "PrimitiveRigidCamera_1_1.py", Path(__file__),
               ROOT / "assets" / "primitive-rigid-camera-1.1.html"]
    audit = {"version": VERSION, "status": "PASS", "data_kind": "analytic_synthetic_material_paths",
             "physical_time": [0., 4.], "display_samples": 121, "particles": 7,
             "rotation_integrator": "RK4, 4 substeps/output interval, polar roundoff correction",
             "objective": "sum over six neighbors of squared observed material speed",
             "cases": evidence, "nonidentity_initial_gauge_error": gauge_error,
             "finite_difference_observer_convergence": finite, "collinear_input_rejected": rejected,
             "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    payload = {"times": times.tolist(), "scenes": scenes, "version": VERSION}
    template = sources[-1].read_text(encoding="utf-8")
    fragment = template.replace("/*__PRIMITIVE_DATA__*/null", json.dumps(payload, separators=(",", ":")))
    assert "/*__PRIMITIVE_DATA__*/" not in fragment
    target = args.output / "primitive-rigid-camera.html"
    target.write_text(fragment, encoding="utf-8")
    if args.inline:
        args.inline.parent.mkdir(parents=True, exist_ok=True)
        args.inline.write_text(fragment, encoding="utf-8")
    print(json.dumps({"status": "PASS", "gauge_error": gauge_error, "finite_difference": finite,
                      "fragment_bytes": target.stat().st_size, "output": str(target)}), flush=True)


if __name__ == "__main__":
    main()
