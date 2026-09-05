"""Fail-closed preflight for the frozen Task4-b supervised experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
import yaml


EXPECTED_EXPERIMENT = "mainExp_Task4B_1.2"
EXPECTED_VARIANTS = ["raw", "raw_wide", "fmt_only", "raw_fmt"]
EXPECTED_SEEDS = [7068, 7069, 7070]
EXPECTED_RAW_SHAPE = (20641, 693)
EXPECTED_FMT_SHAPE = (20641, 161)
EXPECTED_PROXY_LABEL_SHA256 = (
    "f545c46427991842c05814d8df71bfa09b3ea11e33d3f240e18fccee0d8383e8"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(config_path: Path, *, require_cuda: bool = False) -> dict:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if spec.get("experiment") != EXPECTED_EXPERIMENT:
        raise RuntimeError(f"unexpected experiment: {spec.get('experiment')!r}")
    if list(spec.get("variants", [])) != EXPECTED_VARIANTS:
        raise RuntimeError("the frozen four-variant comparison changed")
    if list(spec.get("training", {}).get("seeds", [])) != EXPECTED_SEEDS:
        raise RuntimeError("the frozen three-seed population changed")

    output_dir = Path(spec["output_dir"])
    occupied = [
        str(path)
        for path in (output_dir / "summary.json", output_dir / "per_run_metrics.csv")
        if path.exists()
    ]
    if occupied:
        raise RuntimeError(f"formal output already exists: {occupied}")

    cache_path = Path(spec["cache"])
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    actual_sha256 = _sha256(cache_path)
    expected_sha256 = str(spec.get("cache_sha256", "")).lower()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"cache SHA-256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )

    with np.load(cache_path, allow_pickle=False) as cache:
        raw_shape = tuple(cache["raw_features"].shape)
        fmt_shape = tuple(cache["fmt_features"].shape)
        labels = np.asarray(cache["labels"], dtype=np.int8)
        split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
        metadata = json.loads(str(cache["metadata_json"]))
    if raw_shape != EXPECTED_RAW_SHAPE or fmt_shape != EXPECTED_FMT_SHAPE:
        raise RuntimeError(
            f"unexpected cache shapes: raw={raw_shape}, fmt={fmt_shape}"
        )
    if set(np.unique(labels).tolist()) != {0, 1, 2, 3}:
        raise RuntimeError("cache labels are not exactly the four frozen classes")
    if set(np.unique(split_codes).tolist()) != {0, 1, 2}:
        raise RuntimeError("cache split codes are not exactly train/validation/test")
    split_counts = {}
    for split_code, split_name in enumerate(("train", "validation", "test")):
        counts = [
            int(np.count_nonzero((split_codes == split_code) & (labels == class_id)))
            for class_id in range(4)
        ]
        if any(count == 0 for count in counts):
            raise RuntimeError(f"{split_name} lacks a frozen class: {counts}")
        split_counts[split_name] = counts

    certificate = metadata.get("split_nonoverlap_certificate", {})
    minimum_gap = float(certificate.get("minimum_cyclic_x_gap", np.nan))
    required_gap = float(certificate.get("required_minimum_gap", np.nan))
    if (
        certificate.get("certified") is not True
        or not np.isfinite(minimum_gap)
        or not np.isfinite(required_gap)
        or minimum_gap <= required_gap
    ):
        raise RuntimeError(f"invalid spatial split certificate: {certificate}")
    if metadata.get("proxy_label_sha256") != EXPECTED_PROXY_LABEL_SHA256:
        raise RuntimeError("cache does not reference the frozen proxy-label artifact")

    cache_encoder = metadata.get("encoder", {})
    encoder_fields = (
        "num_freq",
        "mode",
        "include_chirality",
        "neighbor_pool",
        "neighbor_scale",
        "neighbor_weight_after_train_standardization",
    )
    for field in encoder_fields:
        if cache_encoder.get(field) != spec["encoder"].get(field):
            raise RuntimeError(f"encoder contract mismatch for {field}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the formal Ibex job")

    return {
        "status": "PASS",
        "experiment": EXPECTED_EXPERIMENT,
        "cache": str(cache_path.resolve()),
        "cache_sha256": actual_sha256,
        "proxy_label_sha256": metadata["proxy_label_sha256"],
        "raw_shape": list(raw_shape),
        "fmt_shape": list(fmt_shape),
        "split_counts_by_class": split_counts,
        "split_nonoverlap_certificate": certificate,
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mainExp_Task4B_1.2.yaml")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        json.dumps(
            preflight(Path(args.config), require_cuda=args.require_cuda),
            indent=2,
        )
    )
