"""Build, audit and train the frozen single-flow primitive reconstruction experiment."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import netCDF4 as nc
import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import cross_offsets, integrate, sha256, write_json
from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate, load_netcdf_window_3d
from FMT_Utils.PrimitiveVAE_3D import fmt_tokens, resample_time, time_plan, PrimitiveVAE, vae_loss, geometry_metrics

DEFAULT_CONFIG = "config/mainExp_Task6_PrimitiveVAE_2.1.json"


def provenance(config):
    return dict(config_sha256=sha256(config),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        node=socket.gethostname(), job_id=os.getenv("SLURM_JOB_ID"),
        array_id=os.getenv("SLURM_ARRAY_TASK_ID"),
        time=datetime.datetime.now().astimezone().isoformat())


def assert_encoder(spec):
    if sha256("FMT_Utils/DFT_FMT_3D.py") != spec["frozen_encoder_sha256"]:
        raise ValueError("Frozen original FMT encoder changed")


def inspect_source(spec, dataset):
    path = Path(spec["source_fields"][dataset])
    with nc.Dataset(path) as source:
        td = _axis_dimension(source, "t")
        times = _coordinate(source, td, np.arange(len(source.dimensions[td])))
        available = list(range(len(times)))
        if "selected_original_indices_json" in source.ncattrs():
            available = json.loads(source.getncattr("selected_original_indices_json"))
        if dataset in spec.get("available_frame_ranges", {}):
            lo, hi = spec["available_frame_ranges"][dataset]
            available = sorted(set(available).intersection(range(lo, hi + 1)))
        extraction = str(getattr(source, "extraction", "complete supplied source timeline"))
        if "missing" in extraction.lower() and len(available) == len(times):
            raise ValueError("Missing-frame export lacks an availability map")
        original = spec.get("original_time_ranges", {}).get(dataset, [times[0], times[-1]])
        plan = time_plan(times, available, original, dataset in spec["cylinder_datasets"])
    return dict(dataset=dataset, source=str(path), source_bytes=path.stat().st_size,
        source_mtime_ns=path.stat().st_mtime_ns, available_frame_indices=available,
        extraction=extraction, plan=plan)


def strict_window(path, start, frames, max_spatial_dim):
    """Reuse coordinates/interpolation conventions but never turn masked voxels into zero flow."""
    field, meta = load_netcdf_window_3d(path, start, frames, max_spatial_dim)
    with nc.Dataset(path) as source:
        dims = {axis: _axis_dimension(source, axis) for axis in "tzyx"}
        canonical = [dims[a] for a in "tzyx"]
        slices = {dims["t"]: slice(start, start + frames),
            **{dims[a]: slice(None, None, meta["spatial_strides"][a]) for a in "xyz"}}
        components = next(names for names in (("u", "v", "w"), ("velocity_x", "velocity_y", "velocity_z"),
            ("Component1", "Component2", "Component3")) if all(n in source.variables for n in names))
        for c, name in enumerate(components):
            variable = source.variables[name]
            if len(variable.dimensions) != 4:
                raise ValueError("Only four-dimensional velocity arrays are supported")
            a = np.ma.asarray(variable[tuple(slices[d] for d in variable.dimensions)])
            a = np.asarray(a.filled(np.nan), np.float32)
            a = a.transpose([variable.dimensions.index(d) for d in canonical])
            if not np.isfinite(a).reshape(frames, -1).any(1).all():
                raise ValueError(f"Missing full velocity frame in {path}/{name}")
            field.field[..., c] = a
    # Float32 source time coordinates can differ by a few ulps. Use the physical
    # duration over the complete window, not an endpoint-clamping correction.
    dt = (meta["frame_count"] - 1)
    with nc.Dataset(path) as source:
        td = _axis_dimension(source, "t")
        t = _coordinate(source, td, np.arange(start, start + frames))
    field.timeInterval = float((t[-1] - t[0]) / dt)
    field.tmax = float(t[-1] - t[0])
    meta["source_time_step"] = field.timeInterval
    meta["masked_voxels"] = "NaN; reject any path encountering nonfinite interpolation"
    return field, meta


def preflight(spec, config):
    assert_encoder(spec)
    root = Path(spec["output_root"])
    if (root / "preflight.json").exists():
        raise FileExistsError("Refuse to replace preflight")
    sources = [inspect_source(spec, d) for d in spec["datasets"]]
    write_json(root / "preflight.json", dict(provenance=provenance(config), sources=sources))
    for s in sources:
        print(s["dataset"], {r:len(v) for r,v in s["plan"]["starts"].items()}, flush=True)


def build(spec, config, dataset):
    assert_encoder(spec)
    root = Path(spec["output_root"])
    frozen = json.loads((root / "preflight.json").read_text())
    if frozen["provenance"]["config_sha256"] != sha256(config):
        raise ValueError("Config differs from preflight")
    source = next(s for s in frozen["sources"] if s["dataset"] == dataset)
    if source != inspect_source(spec, dataset):
        raise ValueError("Source metadata changed after preflight")
    folder = root / "data" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.iterdir()):
        raise FileExistsError(f"Refuse to overwrite scientific cache {folder}")
    counts = spec["sampling"]["retained_samples"]
    rows, files, serial = [], {}, 0
    all_scales = spec["scales"] + spec["unseen_scales"]
    for role in ("train", "validation", "test", "unseen_scale"):
        starts = source["plan"]["starts"]["test" if role == "unseen_scale" else role]
        scale_ids = list(range(len(spec["scales"]))) if role != "unseen_scale" else list(range(len(spec["scales"]), len(all_scales)))
        target = int(counts[role])
        chunks = {k: [] for k in ("geometry", "token", "origin", "radius", "seed_time", "physical_dt", "integration_steps", "horizon", "scale_id", "source_start", "source_end", "primitive_id")}
        quotas = np.full((len(starts), len(scale_ids)), target // (len(starts) * len(scale_ids)), dtype=int)
        # Balanced exact quotas, randomized cell priority independent of geometry.
        rng = np.random.default_rng(spec["sampling"]["data_seed"] + spec["datasets"].index(dataset) * 100 + list(counts).index(role))
        flat = quotas.ravel()
        flat[rng.permutation(len(flat))[:target % len(flat)]] += 1
        for time_id, start in enumerate(starts):
            if not quotas[time_id].any():
                continue
            field, window = strict_window(source["source"], start, 13, spec["sampling"]["max_spatial_dim"])
            spacing = float(np.min(field.gridInterval))
            for scale_slot, sid in enumerate(scale_ids):
                wanted = int(quotas[time_id, scale_slot])
                if wanted == 0:
                    continue
                scale = all_scales[sid]
                radius = spacing * scale["offset_grid_scale"]
                h = float(field.timeInterval * scale["dt_scale"])
                steps = int(scale["integration_steps"])
                duration = h * steps
                lo = np.asarray(field.domainMinBoundary, float) + radius
                hi = np.asarray(field.domainMaxBoundary, float) - radius
                if np.any(lo >= hi):
                    raise ValueError("Radius does not fit loaded spatial domain")
                accepted, attempted, rejected = 0, 0, 0
                # Retry count depends only on physical validity, never model/labels.
                while accepted < wanted and attempted < wanted * spec["sampling"]["max_candidate_multiplier"]:
                    batch = min(max(32, 2 * (wanted - accepted)), 512,
                                wanted * spec["sampling"]["max_candidate_multiplier"] - attempted)
                    centers = rng.uniform(lo, hi, (batch, 3))
                    paths, valid = integrate(field, centers[:, None] + radius * cross_offsets(), 0., duration, steps)
                    good = valid.all(1) & np.isfinite(paths).all((1, 2, 3))
                    rejected += int((~good).sum())
                    attempted += batch
                    indices = np.flatnonzero(good)[:wanted - accepted]
                    if not len(indices):
                        continue
                    geometry = ((resample_time(paths[indices]) - centers[indices, None, None]) / radius).astype(np.float32)
                    n = len(indices)
                    chunks["geometry"].append(geometry)
                    chunks["token"].append(fmt_tokens(geometry))
                    chunks["origin"].append(centers[indices])
                    for key, value in dict(radius=radius, seed_time=window["source_time"], physical_dt=h,
                        integration_steps=steps, horizon=duration, scale_id=sid, source_start=start,
                        source_end=start + 12).items():
                        chunks[key].append(np.full(n, value))
                    chunks["primitive_id"].append(np.arange(serial, serial + n, dtype=np.int64))
                    serial += n
                    accepted += n
                rows.append(dict(role=role, source_start=start, seed_time=window["source_time"], scale_id=sid,
                    requested=wanted, retained=accepted, candidates=attempted, rejected_boundary_or_nonfinite=rejected,
                    valid_surplus_discarded=attempted-rejected-accepted, window=window))
                if accepted != wanted:
                    write_json(folder / "failed_sampling.json", dict(rows=rows, provenance=provenance(config)))
                    raise ValueError(f"Insufficient valid primitives {dataset}/{role}/{start}/{sid}: {accepted}/{wanted}")
            print(dataset, role, start, "complete", flush=True)
        arrays = {k: np.concatenate(v) for k, v in chunks.items()}
        for key in ("integration_steps", "scale_id", "source_start", "source_end", "primitive_id"):
            arrays[key] = arrays[key].astype(np.int64)
        path = folder / f"{role}.npz"
        np.savez(path, **arrays)
        files[role] = dict(path=str(path), sha256=sha256(path), samples=len(arrays["geometry"]))
    write_json(folder / "manifest.json", dict(provenance=provenance(config), source=source, files=files,
        sampling=rows, coordinate_convention="(x(t)-center(t0))/initial_radius", scales=all_scales))


def audit_data(spec, config, dataset):
    folder = Path(spec["output_root"]) / "data" / dataset
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["provenance"]["config_sha256"] == sha256(config)
    frames, ids, counts = {}, set(), {}
    for role, file in manifest["files"].items():
        path = folder / f"{role}.npz"
        assert sha256(path) == file["sha256"]
        with np.load(path) as data:
            x, token = data["geometry"], data["token"]
            assert x.shape == (spec["sampling"]["retained_samples"][role], 7, 32, 3)
            assert np.isfinite(x).all() and np.isfinite(token).all()
            np.testing.assert_allclose(x[:, :, 0], np.broadcast_to(cross_offsets(), x[:, :, 0].shape), atol=2e-5)
            chosen = np.unique(np.linspace(0, len(x)-1, 97).astype(int))
            np.testing.assert_allclose(token[chosen], fmt_tokens(x[chosen]), rtol=1e-5, atol=1e-5)
            assert not ids.intersection(data["primitive_id"].tolist())
            ids.update(data["primitive_id"].tolist())
            tlo, thi = manifest["source"]["plan"]["seed_time_bounds"]
            assert np.all((data["seed_time"] >= tlo-1e-6) & (data["seed_time"] <= thi+1e-6))
            np.testing.assert_allclose(data["horizon"], data["physical_dt"] * data["integration_steps"])
            assert (data["radius"] > 0).all()
            used = set()
            for lo, hi in set(zip(data["source_start"], data["source_end"])):
                used.update(range(int(lo), int(hi)+1))
            assert used <= set(manifest["source"]["available_frame_indices"])
            frames[role] = used
            counts[role] = len(x)
    assert not frames["train"].intersection(frames["validation"] | frames["test"] | frames["unseen_scale"])
    assert not frames["validation"].intersection(frames["test"] | frames["unseen_scale"])
    result = dict(dataset=dataset, passed=True, counts=counts, provenance=provenance(config),
        frame_sets={k:sorted(v) for k,v in frames.items()}, manifest_sha256=sha256(folder / "manifest.json"))
    write_json(folder / "audit.json", result)
    return result


def predict(model, x, device, batch=1024):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            p, _, _ = model(torch.as_tensor(x[start:start+batch], device=device), sample=False)
            values.append(p.cpu().numpy())
    return np.concatenate(values)


def fit(spec, x, y, vx, vy, seed, device, epochs=None, fixed_steps=None, folder=None):
    cfg = spec["vae"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = x.std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.
    x = ((x - mean) / std).astype(np.float32)
    vx = ((vx - mean) / std).astype(np.float32)
    model = PrimitiveVAE(x.shape[1], cfg["width"], cfg["latent_dim"], cfg["blocks"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.)
    xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    batch = cfg["batch_size"]
    per_epoch = math.ceil(len(x) / batch)
    updates = fixed_steps if fixed_steps is not None else epochs * per_epoch
    rng = np.random.default_rng(seed)
    probe = np.sort(rng.choice(len(x), min(4096, len(x)), replace=False))
    records, started, order = [], time.monotonic(), None
    def record(step, rec=None, kl=None):
        train_metric = geometry_metrics(predict(model, x[probe], device), y[probe])["position_rmse_r"]
        val_metric = geometry_metrics(predict(model, vx, device), vy)["position_rmse_r"]
        row = dict(step=step, train_probe_rmse_r=train_metric, validation_rmse_r=val_metric,
            sampled_reconstruction=rec, kl=kl, elapsed_seconds=time.monotonic()-started)
        records.append(row)
        if folder is not None:
            with (folder / "curve.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps(row), flush=True)
    record(0)
    for step in range(updates):
        if step % per_epoch == 0:
            order = rng.permutation(len(x))
        selected = order[(step % per_epoch)*batch:((step % per_epoch)+1)*batch]
        ids = torch.as_tensor(selected, device=device)
        lr = cfg["learning_rate"] * (.05 + .95 * .5 * (1 + math.cos(math.pi*step/max(1, updates-1))))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output, mu, logvar = model(xt[ids], sample=True)
        loss, rec, kl = vae_loss(output, yt[ids], mu, logvar, cfg["beta"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite VAE loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if (step+1) % cfg["probe_every"] == 0 or step+1 == updates:
            record(step+1, float(rec.detach()), float(kl.detach()))
    return model, mean, std, dict(curve=records, updates=updates, parameter_count=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.monotonic()-started, train_samples=len(x), train_examples_exposed=
        (updates//per_epoch)*len(x)+min(len(x), (updates%per_epoch)*batch), normalization="train-input-only mean/std")


def train(spec, config, dataset, arm, seed):
    assert_encoder(spec)
    root = Path(spec["output_root"])
    data_dir = root / "data" / dataset
    data_audit = json.loads((data_dir / "audit.json").read_text())
    assert data_audit["passed"] and data_audit["provenance"]["config_sha256"] == sha256(config)
    assert data_audit["manifest_sha256"] == sha256(data_dir / "manifest.json")
    folder = root / "runs" / dataset / arm / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.iterdir()):
        raise FileExistsError("Refuse to overwrite a training run")
    if not torch.cuda.is_available():
        raise RuntimeError("Ibex main experiment must use a GPU")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = dict(provenance=provenance(config), dataset=dataset, arm=arm, seed=seed,
        device=torch.cuda.get_device_name(0), torch_version=torch.__version__, vae=spec["vae"])
    write_json(folder / "started.json", info)
    def load(role):
        with np.load(data_dir / f"{role}.npz") as d:
            geometry = d["geometry"]
            inputs = d["token"] if arm == "fmt_all_vae" else geometry.reshape(len(geometry), -1)
            return inputs, geometry, d["scale_id"]
    x, y, _ = load("train")
    vx, vy, _ = load("validation")
    subset = np.sort(np.random.default_rng(spec["sampling"]["data_seed"]).choice(len(x), spec["vae"]["fit_samples"], replace=False))
    fit_dir = folder / "fit_check"
    fit_dir.mkdir()
    model, mean, std, fitting = fit(spec, x[subset], y[subset], x[subset], y[subset], seed, device,
        fixed_steps=spec["vae"]["fit_steps"], folder=fit_dir)
    fitting["metrics"] = geometry_metrics(predict(model, (x[subset]-mean)/std, device), y[subset])
    # The fitting check uses only training primitives; main training starts fresh.
    write_json(fit_dir / "result.json", {**info, **fitting, "validation_role": "same training subset; not held out"})
    del model
    torch.cuda.empty_cache()
    model, mean, std, training = fit(spec, x, y, vx, vy, seed, device, epochs=spec["vae"]["epochs"], folder=folder)
    np.savez(folder / "input_statistics.npz", mean=mean, std=std)
    metrics = []
    for role in ("test", "unseen_scale"):
        tx, ty, scale_ids = load(role)
        pred = predict(model, (tx-mean)/std, device)
        np.savez(folder / f"{role}_predictions.npz", prediction=pred)
        for sid in [-1] + sorted(np.unique(scale_ids).tolist()):
            mask = np.ones(len(ty), bool) if sid == -1 else scale_ids == sid
            metrics.append(dict(role=role, scale_id=sid, **geometry_metrics(pred[mask], ty[mask])))
    write_json(folder / "result.json", {**info, "training": training, "fit_check": fitting["metrics"],
        "metrics": metrics, "data_manifest_sha256":sha256(data_dir / "manifest.json"),
        "model_selection":"none; fixed final epoch", "checkpoints":"none; model in memory only"})


def audit_results(spec, config):
    root = Path(spec["output_root"])
    rows = []
    for dataset in spec["datasets"]:
        for arm in spec["arms"]:
            for seed in spec["seeds"]:
                folder = root / "runs" / dataset / arm / str(seed)
                result = json.loads((folder / "result.json").read_text())
                assert result["provenance"]["config_sha256"] == sha256(config)
                assert result["data_manifest_sha256"] == sha256(root/"data"/dataset/"manifest.json")
                for role in ("test", "unseen_scale"):
                    with np.load(root/"data"/dataset/f"{role}.npz") as d:
                        truth, scales = d["geometry"], d["scale_id"]
                    with np.load(folder/f"{role}_predictions.npz") as d:
                        pred = d["prediction"]
                    for metric in (m for m in result["metrics"] if m["role"]==role):
                        sid = metric["scale_id"]
                        mask = np.ones(len(truth), bool) if sid == -1 else scales==sid
                        fresh = geometry_metrics(pred[mask], truth[mask])
                        for key, value in fresh.items():
                            np.testing.assert_allclose(value, metric[key], rtol=1e-8, atol=1e-9)
                        rows.append(dict(dataset=dataset, arm=arm, seed=seed, **{k:v for k,v in metric.items() if not isinstance(v,list)}))
    assert not any(root.rglob("*.pt")) and not any(root.rglob("*.pth")) and not any(root.rglob("*.ckpt"))
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(root / "final_audit.json", dict(passed=True, rows=len(rows), provenance=provenance(config)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "build", "audit-data", "train", "audit-results"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if args.phase == "preflight": preflight(spec, args.config)
    elif args.phase == "build": build(spec, args.config, spec["datasets"][args.index])
    elif args.phase == "audit-data":
        for d in spec["datasets"]: audit_data(spec, args.config, d)
    elif args.phase == "train":
        jobs = [(d,a,s) for d in spec["datasets"] for a in spec["arms"] for s in spec["seeds"]]
        train(spec, args.config, *jobs[args.index])
    else: audit_results(spec, args.config)


if __name__ == "__main__":
    main()
