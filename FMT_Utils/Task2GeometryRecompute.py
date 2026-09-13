"""Versioned pathline regeneration with a single background worker.

The requested slice is fitted and displayed without a data split. Each
request has its own directory; existing caches and VAE recipes are read-only.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from threading import Lock
import traceback
from uuid import uuid4

import numpy as np


def integration_window(times, start_time, dt, steps):
    """Bracket any requested start and clip only at the source time endpoint."""
    times = np.asarray(times, dtype=float)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt 必须为正数")
    if not np.isfinite(steps) or int(steps) != steps or steps < 1:
        raise ValueError("积分步数 N 必须为正整数")
    differences = np.diff(times)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(differences <= 0) or not np.allclose(differences, differences.mean(), rtol=1e-4, atol=1e-7):
        raise ValueError("需要均匀、递增的源时间坐标")
    if not np.isfinite(start_time) or not times[0] <= start_time < times[-1]:
        raise ValueError(f"起始时间必须满足 {times[0]:g} ≤ t0 < {times[-1]:g}")
    end = min(float(start_time) + float(dt) * int(steps), float(times[-1]))
    start = int(np.searchsorted(times, start_time, side="right") - 1)
    stop = min(int(np.searchsorted(times, end)), len(times) - 1)
    return start, max(stop, start + 1) - start + 1


def integration_schedule(duration, dt):
    ratio = duration / dt
    full = int(np.floor(ratio))
    if abs(ratio - round(ratio)) < 1e-10:
        full = int(round(ratio))
    return full, max(0., duration - full*dt)


def integrate_clipped_primitives(field, seeds, start_time, end_time, dt, sampled_steps, offset, chunk_size=2048):
    """RK4 with requested full steps and a final short step, then time resampling."""
    from FLowUtils.flowlineIntegral import compute_pathlines_3D_batch
    if not field.tmin <= start_time < end_time <= field.tmax:
        raise ValueError("积分时间超出实际流场范围")
    duration = end_time - start_time
    full, tail = integration_schedule(duration, dt)
    offsets = np.array([[0,0,0],[offset,0,0],[-offset,0,0],[0,offset,0],[0,-offset,0],[0,0,offset],[0,0,-offset]])
    expanded = (np.asarray(seeds)[:,None,:] + offsets).reshape(-1,3)
    chunks, masks = [], []
    target_times = np.linspace(start_time, end_time, sampled_steps)
    for first in range(0, len(expanded), chunk_size):
        xyz = expanded[first:first+chunk_size]
        seed = np.column_stack((xyz, np.full(len(xyz), start_time)))
        if full:
            result, lengths = compute_pathlines_3D_batch(field, seed, start_time, start_time+full*dt, dt, full, 'RK4')
            positions = result[:, :full+1]
            valid = lengths == full+1
        else:
            positions, valid = seed[:,None,:], np.ones(len(seed), dtype=bool)
        times = start_time + np.arange(full+1)*dt
        if tail > 1e-10:
            last = positions[:,-1].astype(float)
            last[:,3] = start_time+full*dt
            final, lengths = compute_pathlines_3D_batch(field, last, last[0,3], end_time, tail, 1, 'RK4')
            positions = np.concatenate((positions, final[:,1:2]),axis=1)
            valid &= lengths == 2
            times = np.append(times, end_time)
        xyz = positions[...,:3]
        valid &= np.isfinite(xyz).all(axis=(1,2)) & ((xyz >= field.domainMinBoundary) & (xyz <= field.domainMaxBoundary)).all(axis=(1,2))
        right = np.clip(np.searchsorted(times,target_times,side='right'),1,len(times)-1)
        left = right-1
        weight = (target_times-times[left])/(times[right]-times[left])
        sampled = positions[:,left,:3]*(1-weight)[None,:,None]+positions[:,right,:3]*weight[None,:,None]
        chunks.append(np.concatenate((sampled,np.broadcast_to(target_times[None,:,None],(len(seed),sampled_steps,1))),axis=2))
        masks.append(valid)
    paths = np.concatenate(chunks).reshape(-1,7,sampled_steps,4).astype(np.float32)
    valid = np.concatenate(masks).reshape(-1,7).all(axis=1)
    return paths[valid], valid


def catalog(config):
    from experiments.Task2_Visual_Analysis import frozen_recipe, root_path
    import yaml
    spec, source, _, _, _ = frozen_recipe(config)
    policy = json.loads(root_path(config["cylinder_time_policy"]).read_text(encoding="utf-8"))
    result = {}
    for group in source["groups"].values():
        source_config = root_path(group["source_config"])
        source_settings = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        for dataset in group["datasets"]:
            try:
                manifest_path = root_path(group["development_cache"]) / dataset / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                cutoff = None
                if dataset in policy["simulation_time_ranges"]:
                    low, high = policy["simulation_time_ranges"][dataset]
                    cutoff = max(policy["minimum_physical_start"], low + .5*(high-low))
                eligible = [e for e in manifest["slices"] if cutoff is None or e["source_time"] >= cutoff]
                display_id = int(min(eligible, key=lambda e:e["source_time"])["ordinal"])
                train_ids = []
                declared = manifest["source"]
                path = Path(declared["path"] if isinstance(declared, dict) else declared)
                entries = {int(e["ordinal"]): e for e in manifest["slices"]}
                display = entries[display_id]
                step = float(display["source_time_step"])
                source_range = None
                if path.is_file():
                    if dataset == "channel":
                        source_range = [0., float(source_settings["channel_observer"]["duration"])]
                    else:
                        import netCDF4 as nc
                        from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate
                        with nc.Dataset(path) as data:
                            dimension = _axis_dimension(data, "t")
                            clock = _coordinate(data, dimension, np.arange(len(data.dimensions[dimension])))
                        source_range = [float(clock[0]), float(clock[-1])]
                result[dataset] = dict(dataset=dataset, source_path=str(path), source_time_range=source_range,
                    available=path.is_file(), source_config=str(source_config),
                    settings=source_settings, manifest_path=str(manifest_path), manifest=manifest,
                    train_ids=train_ids, display_id=display_id, cutoff=cutoff, entries=entries,
                    default_dt=step * float(source_settings["pathlines"]["dt_scale"]),
                    default_steps=int(source_settings["pathlines"]["integration_steps"]),
                    spec=spec)
            except (OSError, ValueError, KeyError) as exc:
                result[dataset] = {"dataset": dataset, "available": False, "error": str(exc)}
    return result


def _channel_loader(entry, path):
    from experiments.Build_Channel_Killing_Cache import load_channel_vtk
    from FMT_Utils.KillingObserver3D import integrate_killing_frame, compose_steady_to_unsteady
    from FLowUtils.VectorField3d import UnsteadyVectorField3D
    settings = entry["settings"]
    interpolator, points, axes, lower, upper, vtk_meta = load_channel_vtk(
        path, int(settings["sampling"]["max_spatial_dim"]),
        float(settings["channel_observer"]["output_crop_fraction"]))
    times = np.linspace(0, settings["channel_observer"]["duration"], settings["channel_observer"]["total_frames"])
    parameters = np.asarray(entry["manifest"]["observer_parameters"])
    if len(parameters) != len(times):
        raise ValueError("Channel 观察者参数与源时间长度不一致")
    rotation, displacement = integrate_killing_frame(parameters, times[1] - times[0])
    def load(start, count):
        selection = slice(start, start + count)
        data = compose_steady_to_unsteady(points, interpolator, parameters[selection],
            rotation[selection], displacement[selection], bounds_min=lower, bounds_max=upper)
        x, y, z = axes
        field = UnsteadyVectorField3D(len(x), len(y), len(z), count,
            np.array([x[0], y[0], z[0]]), np.array([x[-1], y[-1], z[-1]]),
            0., float((times[1] - times[0]) * (count - 1)))
        field.field = data.reshape(count, len(z), len(y), len(x), 3)
        return field, {"source_time": float(times[start]), "source_start_index": start,
            "source_time_step": float(times[1] - times[0]), "source_path": str(path),
            "frame_count": count, "loaded_shape_TZYXC": list(field.field.shape),
            "channel_observer": "frozen manifest parameters", "vtk": vtk_meta}
    return times, load


def recompute(config, entry, dt, steps, run_dir, progress, start_time=None):
    import torch
    import netCDF4 as nc
    from DeepUtils.utils import EasyConfig
    from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate, load_netcdf_window_3d
    from FMT_Utils.FMT_3D_pipeline import generate_seeding_grid_3d, integrate_cross_primitives_3d
    from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
    from FMT_Utils.Task2VisualAnalysis import digest
    from experiments.Build_Task2_Universality_Cache import resolve_primitive_offset
    from experiments.Run_Task2_3D_Main import _prepare_inputs
    from experiments.Verify_HighReVAE import _train
    from experiments.Task2_Visual_Analysis import frozen_recipe, versions, code_identity

    if steps is None or not np.isfinite(steps) or int(steps) != steps:
        raise ValueError("N 必须为整数")
    dt, steps = float(dt), int(steps)
    spec, _, winner, architecture, frozen_path = frozen_recipe(config)
    settings = entry["settings"]
    sampled_steps = int(settings["pathlines"]["sampled_steps"])
    if not 1 <= steps <= config["recompute"]["max_integration_steps"]:
        raise ValueError(f"N 必须介于1和{config['recompute']['max_integration_steps']}")
    source_path = Path(entry["source_path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"原始流场不可用：{source_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if entry["dataset"] == "channel":
        progress("读取 Channel VTK 与冻结观察者参数")
        times, loader = _channel_loader(entry, source_path)
        spatial_counts = entry["manifest"]["vtk"]["output_counts_xyz"]
    else:
        with nc.Dataset(source_path) as data:
            tdim = _axis_dimension(data, "t")
            times = _coordinate(data, tdim, np.arange(len(data.dimensions[tdim])))
            sizes = [len(data.dimensions[_axis_dimension(data, axis)]) for axis in "xyz"]
        maximum = int(settings["sampling"]["max_spatial_dim"])
        spatial_counts = [int(np.ceil(n / max(1, np.ceil(n / maximum)))) for n in sizes]
        def loader(start, count):
            return load_netcdf_window_3d(source_path, start, count, maximum)
    if start_time is None:
        start_time = float(entry["entries"][entry["display_id"]]["source_time"])
    start_time = float(start_time)
    entries = {0: {"source_time": start_time}}
    ordinals = [0]
    windows = {0: integration_window(times, start_time, dt, steps)}
    end_time = min(start_time + dt*steps, float(times[-1]))
    estimated = windows[0][1] * int(np.prod(spatial_counts)) * 3 * 4
    if estimated > config["recompute"]["max_window_bytes"]:
        raise ValueError("所需流场窗口超过配置内存上限，请减小 dt 或 N")
    records, evidence = {}, []
    for j, ordinal in enumerate(ordinals):
        progress(f"重新积分时间片 {j+1}/{len(ordinals)}，t={entries[ordinal]['source_time']:.6g}")
        field, meta = loader(*windows[ordinal])
        meta["loaded_velocity_sha256"] = hashlib.sha256(memoryview(np.ascontiguousarray(field.field)).cast("B")).hexdigest()
        first, count = windows[ordinal]
        # Use the two actual source endpoints for the local uniform clock;
        # do not accumulate a noisy float32 first interval past tmax.
        field.tmin = float(times[first])
        field.tmax = float(times[first+count-1])
        field.timeInterval = (field.tmax-field.tmin)/(count-1)
        meta.update(source_window_start=field.tmin, source_window_end=field.tmax,
                    source_time=start_time, source_time_step=field.timeInterval)
        offset = resolve_primitive_offset(field.gridInterval, settings["pathlines"]["offset_grid_scale"],
                                          settings["pathlines"].get("offset_mode", "min"))
        seeds, _ = generate_seeding_grid_3d(field, settings["sampling"]["seed_grid_shape"],
            settings["sampling"]["boundary_fraction"], offset, grid_phase=settings["sampling"].get("seed_grid_phase"))
        primitives, valid = integrate_clipped_primitives(field, seeds, start_time, end_time, dt,
            sampled_steps, offset, chunk_size=settings["pathlines"]["chunk_size"])
        if len(primitives) < max(100, int(config["analysis"]["perplexity"]) + 1):
            raise ValueError(f"t={meta['source_time']}: 完整有效样本仅 {len(primitives)}；请缩短积分时长")
        xyz = primitives[..., :3].astype(np.float32)
        raw = (xyz - xyz[:, :1, :1]).reshape(len(xyz), -1)
        fmt = pathline_dft_features_3d(torch.from_numpy(primitives).to(device),
            num_freq=int(settings["encoder"]["num_freq"]), neighbor_weight=1., neighbor_scale=1.,
            neighbor_pool=settings["encoder"]["neighbor_pool"], mode=settings["encoder"]["mode"],
            include_chirality=bool(settings["encoder"]["include_chirality"])).astype(np.float32)
        meta.update(dataset=entry["dataset"], ordinal=ordinal, valid_primitives=len(xyz), total_primitives=len(seeds),
                    integration_dt_requested=dt, integration_dt_effective=dt, integration_steps=steps,
                    integration_duration=end_time-start_time, integration_end_time=end_time,
                    requested_end_time=start_time+dt*steps, truncated_at_tmax=end_time < start_time+dt*steps,
                    sampled_steps=sampled_steps, primitive_offset=offset,
                    full_dt_steps=integration_schedule(end_time-start_time,dt)[0],
                    final_step_dt=integration_schedule(end_time-start_time,dt)[1],
                    resampling="linear interpolation at 32 uniform times")
        records[ordinal] = {"raw": raw, "fmt": fmt, "features": {}, "metadata": meta}
        path = run_dir / f"geometry_slice_{ordinal:02d}.npz"
        np.savez_compressed(path, raw_features=raw, fmt_features=fmt,
                            sample_ids=np.flatnonzero(valid), metadata_json=json.dumps(meta))
        evidence.append({"path": str(path), "sha256": digest(path), **meta})
        if ordinal == 0:
            display_paths, display_ids = xyz, np.flatnonzero(valid)
        del field, primitives
    train = [records[0]]
    evaluation = train
    latent, losses = {}, {}
    source_config = EasyConfig(entry["source_config"])
    for arm in ("raw", "fmt"):
        progress(f"训练 {arm.upper()} + VAE，固定 {architecture['optimizer_steps']} 步；两臂使用同一配方")
        tx, ex = _prepare_inputs(train, evaluation, arm, winner["fmt_feature"], device)
        _, latent[f"{arm}_mu"], losses[arm] = _train(tx, ex, architecture, source_config,
            int(config["training_seed"]), device)
        del tx, ex
    meta = {"schema": 1, "experiment": config["experiment"], "config": config,
        "dataset": entry["dataset"], "display_split": config["display_split"],
        "display_ordinal": None, "display_metadata": evaluation[0]["metadata"],
        "train_ordinals": [], "fit_scope": "all primitives at requested start time; no data split",
        "integration_dt": dt, "integration_steps": steps, "integration_duration": end_time-start_time, "start_time": start_time, "end_time": end_time,
        "requested_end_time": start_time+dt*steps, "source_time_range": [float(times[0]),float(times[-1])],
        "sampled_steps": sampled_steps, "architecture": architecture, "fmt_feature": winner["fmt_feature"],
        "training_seed": config["training_seed"], "losses": losses, "device": str(device),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "versions": versions(), "code": code_identity(), "recompute_source_sha256": digest(Path(__file__)),
        "source_config_sha256": digest(entry["source_config"]),
        "source_file": {"path": str(source_path), "size": source_path.stat().st_size, "mtime_ns": source_path.stat().st_mtime_ns},
        "source_manifest_sha256": digest(entry["manifest_path"]), "frozen_recipe_sha256": digest(frozen_path),
        "geometry_evidence": evidence, "reference_labels_used": False, "confirmation_opened": None, "split_policy": "none; unrestricted interactive exploration",
        "sample_id_definition": "original seed-grid row before invalid-path filtering",
        "checkpoint_policy": "in-memory training only; no checkpoints",
        "scope": "interactive in-sample exploration; no independent evaluation"}
    target = run_dir / "latent_bundle.npz"
    np.savez_compressed(target, paths=display_paths, sample_ids=display_ids, **latent, metadata_json=json.dumps(meta))
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return target


class RecomputeJobs:
    """One worker prevents simultaneous training from sharing RNG/device state."""
    def __init__(self, config):
        self.config = config
        self.catalog = catalog(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task2-geometry")
        self.lock = Lock()
        self.jobs = {}

    def start(self, dataset, dt, steps, start_time=None):
        from experiments.Task2_Visual_Analysis import root_path
        if start_time is not None and not np.isfinite(start_time):
            raise ValueError("起始时间必须为有限数值")
        entry = self.catalog.get(dataset)
        if not entry or not entry["available"]:
            raise ValueError("所选流场的原始数据不可用")
        if dt is None or not np.isfinite(dt) or dt <= 0:
            raise ValueError("请填写正数 dt")
        if steps is None or not np.isfinite(steps) or int(steps) != steps or not 1 <= steps <= self.config["recompute"]["max_integration_steps"]:
            raise ValueError("N 必须是允许范围内的整数，且至少为1")
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("已有几何重算正在运行，请等待完成")
            key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
            directory = root_path(self.config["output_root"]) / dataset / key
            directory.mkdir(parents=True, exist_ok=False)
            job = {"id": key, "status": "running", "message": "等待读取原始流场", "dataset": dataset,
                   "dt": float(dt), "steps": int(steps), "start_time": start_time, "run_dir": str(directory),
                   "created_utc": datetime.now(timezone.utc).isoformat()}
            self.jobs[key] = job
            self._save(job)
        self.executor.submit(self._run, key, entry)
        return key

    def _save(self, job):
        (Path(job["run_dir"]) / "job.json").write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")

    def snapshot(self, key):
        with self.lock:
            return dict(self.jobs[key])

    def _run(self, key, entry):
        def update(message):
            with self.lock:
                self.jobs[key]["message"] = message
                self._save(self.jobs[key])
        job = self.snapshot(key)
        try:
            target = recompute(self.config, entry, job["dt"], job["steps"], Path(job["run_dir"]), update, job.get("start_time"))
            from experiments.Task2_Visual_Analysis import Analysis
            update("计算两臂 t-SNE 投影和默认聚类")
            analysis = Analysis(target, self.config["analysis"])
            for arm in ("raw", "fmt"):
                settings = analysis.settings
                analysis.write_report(Path(job["run_dir"]) / "reports" / arm, arm,
                    settings["cluster_space"], settings["eps"], settings["min_samples"])
            outcome = dict(status="complete", message="重算完成", bundle=str(target))
        except Exception as exc:
            outcome = dict(status="failed", message=str(exc), traceback=traceback.format_exc())
        finally:
            with self.lock:
                self.jobs[key].update(outcome)
                self.jobs[key]["finished_utc"] = datetime.now(timezone.utc).isoformat()
                self._save(self.jobs[key])
