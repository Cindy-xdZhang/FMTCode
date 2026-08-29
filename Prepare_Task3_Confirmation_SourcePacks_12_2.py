"""Create exact pre-strided temporal source packs for Task3 confirmation.

The packer runs the production ``load_netcdf_window_3d`` reader on each frozen
original source index, preserves its float32 velocity samples and physical
bounds exactly, and concatenates only those four independent windows.  The
result removes unused times and spatial samples; it does not interpolate,
filter, or change the field consumed by the pathline integrator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import netCDF4 as nc
import numpy as np

import Build_Task3_SpatialRobust_Confirmation_5_2 as spatial
from DeepUtils.utils import EasyConfig
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d


ROOT = Path(__file__).resolve().parent
DEFAULT_REMOTE_DATA_ROOT = Path("/home/zhanx0o/DeepVortex/FLowDataFolder")
DEFAULT_REMOTE_PACK_ROOT = Path(
    "/ibex/scratch/zhanx0o/FMT_Task3_Confirmation_SourcePacks_12_2"
)
PACK_DATASETS = (
    "halfcylinderRe6400", "deltaWing_LBM", "f22raptor", "boeing747",
)
REMOTE_ORIGINAL_NAMES = {
    "cylinder3d": "halfcylinderRe160Resampled.nc",
    "halfcylinderRe640": "halfcylinderRe640resampled.nc",
    "tangaroa": "tangaroa.nc",
    "deltaWing_resampled": "deltaWing_mag0_3reesampled.nc",
    "smokeBuoyancy": "SmokeBuoyancy80_239.nc",
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _remote_posix_path(value: str | Path) -> PurePosixPath:
    """Normalize an Ibex path without applying the Windows host syntax."""
    path = PurePosixPath(str(value).replace("\\", "/"))
    if not path.is_absolute():
        raise ValueError(f"remote path must be absolute POSIX path: {value}")
    return path


def rewrite_manifest_remote_paths(
    manifest_path: str | Path,
    remote_pack_root: str | Path = DEFAULT_REMOTE_PACK_ROOT,
    remote_data_root: str | Path = DEFAULT_REMOTE_DATA_ROOT,
    remote_channel_path: str | Path | None = None,
) -> Path:
    """Repair only host-dependent path strings in an existing manifest.

    Source-window hashes, original/effective indices, the seed-grid phase and
    every equivalence field remain byte-for-byte identical as JSON values.
    This is intentionally separate from pack construction so a path-format
    repair never rereads or overwrites the already verified source packs.
    """
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_datasets = {
        dataset
        for settings in spatial.SETTINGS.values()
        for dataset in settings["indices"]
    }
    if set(payload.get("datasets", {})) != expected_datasets:
        raise RuntimeError("staging manifest dataset set changed")
    if list(payload.get("seed_grid_phase", [])) != list(
        spatial.SEED_GRID_PHASE
    ):
        raise RuntimeError("staging manifest seed-grid phase changed")
    if not bool(payload.get("scientific_protocol_unchanged", False)):
        raise RuntimeError("staging manifest equivalence declaration missing")

    pack_root = _remote_posix_path(remote_pack_root)
    data_root = _remote_posix_path(remote_data_root)
    channel_path = _remote_posix_path(
        remote_channel_path
        if remote_channel_path is not None else pack_root / "channel.vtk"
    )
    for dataset, entry in payload["datasets"].items():
        kind = str(entry.get("kind", ""))
        if kind == "exact_prestrided_temporal_window_pack":
            remote = pack_root / f"{dataset}_task3_confirmation_windows.nc"
        elif dataset == "channel" and kind == "original_channel_vtk":
            remote = channel_path
        elif kind == "remote_original_netcdf":
            if dataset not in REMOTE_ORIGINAL_NAMES:
                raise RuntimeError(
                    f"{dataset}: no frozen remote original filename"
                )
            remote = data_root / REMOTE_ORIGINAL_NAMES[dataset]
        else:
            raise RuntimeError(
                f"{dataset}: unsupported staging source kind {kind!r}"
            )
        entry["path"] = str(remote)

    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(manifest_path)
    return manifest_path


def _base_config_for_dataset(dataset: str) -> tuple[str, EasyConfig, dict]:
    for group, settings in spatial.SETTINGS.items():
        if dataset in settings["indices"]:
            return group, EasyConfig(settings["base"]), settings
    raise KeyError(dataset)


def _entry(config: EasyConfig, dataset: str) -> dict:
    matches = [dict(item) for item in config.datasets if str(item["id"]) == dataset]
    if len(matches) != 1:
        raise RuntimeError(f"{dataset}: expected exactly one source entry")
    return matches[0]


def _frame_count(config: EasyConfig) -> int:
    return int(np.ceil(
        float(config.pathlines.dt_scale)
        * int(config.pathlines.integration_steps)
    )) + 2


def _write_pack(
    dataset: str,
    source_path: Path,
    output_path: Path,
    original_indices: list[int],
    frame_count: int,
    max_spatial_dim: int,
    overwrite: bool = False,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to replace existing source pack without --overwrite: "
            f"{output_path}"
        )
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        raise FileExistsError(f"temporary source-pack path already exists: {temporary}")
    effective_indices = [i * frame_count for i in range(len(original_indices))]
    source_windows = []
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as target:
            variables = None
            time_values = np.empty(
                len(original_indices) * frame_count, dtype=np.float64
            )
            for ordinal, (original_index, effective_index) in enumerate(zip(
                original_indices, effective_indices
            )):
                field, metadata = load_netcdf_window_3d(
                    source_path, original_index, frame_count, max_spatial_dim
                )
                values = np.ascontiguousarray(field.field, dtype=np.float32)
                if variables is None:
                    _, z_count, y_count, x_count, component_count = values.shape
                    if component_count != 3:
                        raise RuntimeError("packed source must have three components")
                    target.createDimension(
                        "tdim", len(original_indices) * frame_count
                    )
                    target.createDimension("zdim", z_count)
                    target.createDimension("ydim", y_count)
                    target.createDimension("xdim", x_count)
                    axes = {
                        "xdim": np.linspace(
                            field.domainMinBoundary[0],
                            field.domainMaxBoundary[0], x_count,
                        ),
                        "ydim": np.linspace(
                            field.domainMinBoundary[1],
                            field.domainMaxBoundary[1], y_count,
                        ),
                        "zdim": np.linspace(
                            field.domainMinBoundary[2],
                            field.domainMaxBoundary[2], z_count,
                        ),
                    }
                    for name, axis in axes.items():
                        coordinate = target.createVariable(name, "f8", (name,))
                        coordinate[:] = axis
                    target.createVariable("tdim", "f8", ("tdim",))
                    chunks = (
                        1, min(z_count, 24), min(y_count, 24), min(x_count, 24)
                    )
                    variables = [
                        target.createVariable(
                            name, "f4", ("tdim", "zdim", "ydim", "xdim"),
                            zlib=True, complevel=1, shuffle=True,
                            chunksizes=chunks,
                        )
                        for name in ("u", "v", "w")
                    ]
                    target.setncattr(
                        "pack_kind", "exact_prestrided_temporal_windows"
                    )
                    target.setncattr("source_dataset", dataset)
                    target.setncattr(
                        "original_source_path", str(source_path.resolve())
                    )
                    target.setncattr(
                        "original_fixed_indices_json",
                        json.dumps(original_indices),
                    )
                    target.setncattr(
                        "effective_fixed_indices_json",
                        json.dumps(effective_indices),
                    )
                    target.setncattr("frame_count", frame_count)
                    target.setncattr("max_spatial_dim", max_spatial_dim)
                expected_shape = variables[0].shape[1:]
                if tuple(values.shape[1:4]) != tuple(expected_shape):
                    raise RuntimeError(
                        f"{dataset}: source-window spatial shape changed"
                    )
                selection = slice(effective_index, effective_index + frame_count)
                for component, variable in enumerate(variables):
                    variable[selection] = values[..., component]
                segment_time = (
                    float(metadata["source_time"])
                    + np.arange(frame_count, dtype=np.float64)
                    * float(metadata["source_time_step"])
                )
                time_values[selection] = segment_time
                source_windows.append({
                    "ordinal": ordinal,
                    "original_source_start_index": int(original_index),
                    "effective_source_start_index": int(effective_index),
                    "source_time": float(metadata["source_time"]),
                    "source_time_step": float(metadata["source_time_step"]),
                    "loaded_shape_TZYXC": list(values.shape),
                    "velocity_sha256": _array_sha256(values),
                    "domain_min": [
                        float(value) for value in field.domainMinBoundary
                    ],
                    "domain_max": [
                        float(value) for value in field.domainMaxBoundary
                    ],
                })
            target.variables["tdim"][:] = time_values

        # Verify the complete temporary artifact before it can replace a good
        # existing pack.  This makes --overwrite recoverable from any failed
        # build or equivalence check.
        verified = []
        for window in source_windows:
            field, metadata = load_netcdf_window_3d(
                temporary,
                window["effective_source_start_index"],
                frame_count,
                max_spatial_dim,
            )
            observed_hash = _array_sha256(field.field)
            if observed_hash != window["velocity_sha256"]:
                raise RuntimeError(f"{dataset}: packed velocity values changed")
            if not np.array_equal(
                np.asarray(field.domainMinBoundary, dtype=np.float32),
                np.asarray(window["domain_min"], dtype=np.float32),
            ) or not np.array_equal(
                np.asarray(field.domainMaxBoundary, dtype=np.float32),
                np.asarray(window["domain_max"], dtype=np.float32),
            ):
                raise RuntimeError(f"{dataset}: packed physical bounds changed")
            if not np.isclose(metadata["source_time"], window["source_time"]):
                raise RuntimeError(f"{dataset}: packed source time changed")
            if not np.isclose(
                metadata["source_time_step"], window["source_time_step"]
            ):
                raise RuntimeError(
                    f"{dataset}: packed source time step changed"
                )
            verified.append(True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"source pack appeared during build: {output_path}"
            )
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "size_bytes": output_path.stat().st_size,
        "original_fixed_indices": original_indices,
        "effective_fixed_indices": effective_indices,
        "frame_count": frame_count,
        "max_spatial_dim": max_spatial_dim,
        "windows": source_windows,
        "all_windows_verified_exact": bool(all(verified)),
    }


def build(
    output_root: str | Path,
    manifest_path: str | Path,
    remote_pack_root: str | Path = DEFAULT_REMOTE_PACK_ROOT,
    remote_data_root: str | Path = DEFAULT_REMOTE_DATA_ROOT,
    remote_channel_path: str | Path | None = None,
    datasets: tuple[str, ...] = PACK_DATASETS,
    overwrite: bool = False,
) -> Path:
    output_root = Path(output_root)
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to replace existing staging manifest without "
            f"--overwrite: {manifest_path}"
        )
    remote_pack_root = _remote_posix_path(remote_pack_root)
    remote_data_root = _remote_posix_path(remote_data_root)
    if remote_channel_path is not None:
        remote_channel_path = _remote_posix_path(remote_channel_path)
    selected = set(datasets)
    unknown = sorted(selected - set(PACK_DATASETS))
    if unknown:
        raise ValueError(f"unsupported source-pack datasets: {unknown}")
    packed = {}
    for dataset in PACK_DATASETS:
        if dataset not in selected:
            continue
        _, config, settings = _base_config_for_dataset(dataset)
        source_path = Path(_entry(config, dataset)["path"])
        target = output_root / f"{dataset}_task3_confirmation_windows.nc"
        packed[dataset] = _write_pack(
            dataset,
            source_path,
            target,
            [int(value) for value in settings["indices"][dataset]],
            _frame_count(config),
            int(config.sampling.max_spatial_dim),
            overwrite=overwrite,
        )

    entries = {}
    for group, settings in spatial.SETTINGS.items():
        config = EasyConfig(settings["base"])
        for dataset, original_indices in settings["indices"].items():
            if dataset in PACK_DATASETS:
                details = packed.get(dataset)
                if details is None:
                    raise RuntimeError(f"{dataset}: source pack was not built")
                entries[dataset] = {
                    "kind": "exact_prestrided_temporal_window_pack",
                    "path": str(
                        remote_pack_root
                        / f"{dataset}_task3_confirmation_windows.nc"
                    ),
                    "sha256": details["sha256"],
                    "original_fixed_indices": details["original_fixed_indices"],
                    "effective_fixed_indices": details["effective_fixed_indices"],
                    "all_windows_verified_exact": True,
                }
            elif dataset == "channel":
                if remote_channel_path is None:
                    raise ValueError("remote_channel_path is required")
                entries[dataset] = {
                    "kind": "original_channel_vtk",
                    "path": str(remote_channel_path),
                    "original_fixed_indices": list(original_indices),
                    "effective_fixed_indices": list(original_indices),
                }
            else:
                entries[dataset] = {
                    "kind": "remote_original_netcdf",
                    "path": str(
                        remote_data_root / REMOTE_ORIGINAL_NAMES[dataset]
                    ),
                    "original_fixed_indices": list(original_indices),
                    "effective_fixed_indices": list(original_indices),
                }
    manifest = {
        "experiment": "Confirm_Task3_CombinedOptimization_12.2_source_staging",
        "scientific_protocol_unchanged": True,
        "equivalence": (
            "packed NetCDF values are exact float32 outputs of the production "
            "strided source-window loader; no interpolation or filtering"
        ),
        "seed_grid_phase": list(spatial.SEED_GRID_PHASE),
        "datasets": entries,
        "local_packs": packed,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    if manifest_temporary.exists():
        raise FileExistsError(
            f"temporary staging-manifest path already exists: "
            f"{manifest_temporary}"
        )
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if manifest_path.exists() and not overwrite:
        manifest_temporary.unlink(missing_ok=True)
        raise FileExistsError(
            f"staging manifest appeared during build: {manifest_path}"
        )
    manifest_temporary.replace(manifest_path)
    print(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/Confirm_Task3_CombinedOptimization_12.2/source_packs",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/Confirm_Task3_CombinedOptimization_12.2/"
            "source_staging_manifest.json"
        ),
    )
    parser.add_argument("--remote-pack-root", default=str(DEFAULT_REMOTE_PACK_ROOT))
    parser.add_argument("--remote-data-root", default=str(DEFAULT_REMOTE_DATA_ROOT))
    parser.add_argument(
        "--remote-channel-path",
        default=str(DEFAULT_REMOTE_PACK_ROOT / "channel.vtk"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(PACK_DATASETS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rewrite-manifest-paths-only", action="store_true",
        help=(
            "normalize only the existing manifest's remote paths; do not "
            "rebuild or overwrite any verified source pack"
        ),
    )
    args = parser.parse_args()
    if args.rewrite_manifest_paths_only:
        rewrite_manifest_remote_paths(
            args.manifest,
            args.remote_pack_root,
            args.remote_data_root,
            args.remote_channel_path,
        )
        return
    build(
        args.output_root,
        args.manifest,
        args.remote_pack_root,
        args.remote_data_root,
        args.remote_channel_path,
        tuple(args.datasets),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
