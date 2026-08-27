"""Read cached 3D pathline primitives and derive label-free feature blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from FMT_Utils.DFT_FMT_3D import (
    fmt_feature_indices_3d,
    pathline_velocity_gradient_dft_features_3d,
    time_local_gram_dft_features_3d,
)


def load_cache_records(cache_dir, expected_count=None, ordinals=None):
    """Load selected cache records without opening held-out slices.

    ``expected_count`` validates the complete directory before selection.
    Supplying ``ordinals`` is therefore suitable for development-only search:
    files outside the requested ordinal set are enumerated, but never opened.
    """
    records = []
    paths = sorted(Path(cache_dir).glob("slice_*.npz"))
    if expected_count is not None and len(paths) != int(expected_count):
        raise RuntimeError(
            f"expected {expected_count} slices in {cache_dir}, found {len(paths)}"
        )
    if ordinals is None:
        selected = list(enumerate(paths))
    else:
        requested = [int(value) for value in ordinals]
        if len(requested) != len(set(requested)):
            raise ValueError(f"duplicate cache ordinals requested: {requested}")
        invalid = [value for value in requested if not 0 <= value < len(paths)]
        if invalid:
            raise IndexError(
                f"cache ordinals {invalid} outside [0, {len(paths)}) in {cache_dir}"
            )
        selected = [(ordinal, paths[ordinal]) for ordinal in requested]
    for ordinal, path in selected:
        with np.load(path) as data:
            raw = np.asarray(data["raw_features"], dtype=np.float32)
            if raw.shape[1] % (7 * 3):
                raise ValueError(f"unexpected raw feature width in {path}: {raw.shape}")
            records.append({
                "path": path,
                "ordinal": ordinal,
                "raw": raw,
                "fmt": np.asarray(data["fmt_features"], dtype=np.float32),
                "reference": np.asarray(data["reference"], dtype=bool),
                "metadata": json.loads(str(data["metadata_json"])),
                "features": {},
            })
    if not records:
        raise RuntimeError(f"no cached slices found in {cache_dir}")
    return records


def _extended_feature(primitives, name, device):
    tensor = torch.from_numpy(primitives).to(device)
    if name.startswith("gram"):
        return time_local_gram_dft_features_3d(
            tensor, num_freq=int(name.removeprefix("gram"))
        ).astype(np.float32)
    if name.startswith("kin"):
        return pathline_velocity_gradient_dft_features_3d(
            tensor, num_freq=int(name.removeprefix("kin"))
        ).astype(np.float32)
    raise ValueError(f"unknown extended FMT block: {name}")


def feature_matrix(record, name, device="cpu"):
    """Return one named raw/FMT representation for a cache record."""
    cache = record["features"]
    if name in cache:
        return cache[name]
    if "+" in name:
        value = np.concatenate(
            [feature_matrix(record, part, device) for part in name.split("+")], axis=1
        ).astype(np.float32)
    elif name == "raw":
        value = record["raw"]
    elif name == "fmt_all":
        value = record["fmt"]
    elif name.startswith("fmt_"):
        indices = fmt_feature_indices_3d(name.removeprefix("fmt_"))
        value = record["fmt"][:, indices]
    elif name.startswith("gram") or name.startswith("kin"):
        length = record["raw"].shape[1] // (7 * 3)
        primitives = record["raw"].reshape(-1, 7, length, 3)
        value = _extended_feature(primitives, name, torch.device(device))
    else:
        raise ValueError(f"unknown feature representation: {name}")
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite values in feature block {name}")
    cache[name] = np.asarray(value, dtype=np.float32)
    return cache[name]


def stack_features(records, name, device="cpu"):
    return np.concatenate([feature_matrix(record, name, device) for record in records])


def stack_reference(records):
    return np.concatenate([record["reference"] for record in records])
