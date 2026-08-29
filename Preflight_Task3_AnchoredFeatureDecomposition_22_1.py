"""Static and real-cache preflight for Task3 anchored feature search 22.1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from FMT_Utils.PathlineClassifier_3D import (
    PathlineBinaryClassifier3D,
    PathlineFMTResidualClassifier3D,
    residual_model_kwargs,
    trainable_parameter_count,
)
from FMT_Utils.Task12Data_3D import feature_matrix
from Search_Task3_FMTResidual_3D import (
    _candidate_spec,
    _decode_job,
    _frozen_raw_normalization,
    _group_for_dataset,
    _load_search_splits,
    _load_spec,
    _normalize_train_only,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_record() -> dict:
    generator = torch.Generator().manual_seed(7068)
    velocity = torch.randn(4, 7, 31, 3, generator=generator)
    primitive = torch.cat((torch.zeros(4, 7, 1, 3), velocity), dim=2)
    primitive = primitive.cumsum(dim=2).numpy().astype(np.float32)
    return {
        "raw": primitive.reshape(4, -1),
        "fmt": np.linspace(-1.0, 1.0, 4 * 161, dtype=np.float32).reshape(4, 161),
        "reference": np.asarray([0, 1, 0, 1], dtype=bool),
        "features": {},
    }


def _model_audit(spec: dict, candidate: dict, fmt_dim: int) -> dict:
    group_name, group = _group_for_dataset(spec, spec["datasets"][0])
    run_spec = _candidate_spec(
        spec, group, candidate, spec["datasets"][0],
        int(spec["screen_seeds"][0]), "fmt", Path("."), int(fmt_dim),
    )
    kwargs = residual_model_kwargs(run_spec["model"])

    def build():
        raw = PathlineBinaryClassifier3D(
            "raw", fmt_dim=int(fmt_dim),
            embedding_dim=int(run_spec["model"]["embedding_dim"]),
        )
        return PathlineFMTResidualClassifier3D(
            raw, fmt_dim=int(fmt_dim), **kwargs
        )

    fmt_model = build()
    raw_pca_model = build()
    fmt_trainable = trainable_parameter_count(fmt_model)
    raw_pca_trainable = trainable_parameter_count(raw_pca_model)
    fmt_total = sum(parameter.numel() for parameter in fmt_model.parameters())
    raw_pca_total = sum(
        parameter.numel() for parameter in raw_pca_model.parameters()
    )
    ceiling = int(spec["raw_wide_parameter_count"])
    if fmt_trainable != raw_pca_trainable or fmt_total != raw_pca_total:
        raise RuntimeError(f"{candidate['id']}: paired models differ in capacity")
    if fmt_total >= ceiling:
        raise RuntimeError(
            f"{candidate['id']}: total parameters {fmt_total} reach Raw-wide {ceiling}"
        )
    return {
        "candidate_id": str(candidate["id"]),
        "fmt_feature": str(candidate["fmt_feature"]),
        "feature_width": int(fmt_dim),
        "paired_trainable_parameter_count": int(fmt_trainable),
        "paired_total_parameter_count": int(fmt_total),
        "raw_wide_parameter_ceiling": ceiling,
        "group_probe": group_name,
    }


def _preflight_checkpoint_group(group: dict, dataset: str, seed: int) -> dict:
    """Resolve the read-only local checkpoint mirror without changing config."""
    filename = f"{dataset}_raw_seed{int(seed)}.pt"
    primary = Path(group["raw_checkpoint_dir"]) / filename
    if primary.exists():
        return dict(group)
    mirror_root = Path(".deploy/task3_dense_checkpoints")
    mirror = mirror_root / Path(group["raw_checkpoint_dir"]) / filename
    if not mirror.exists():
        raise FileNotFoundError(
            f"missing frozen Raw checkpoint at both {primary} and {mirror}"
        )
    resolved = dict(group)
    resolved["raw_checkpoint_dir"] = str(mirror.parent)
    return resolved


def _preflight_spec_with_label_mirror(spec: dict, dataset: str) -> dict:
    """Use an equivalent local p95 mirror when the canonical label is absent."""
    group_name, group = _group_for_dataset(spec, dataset)
    primary = Path(group["label_cache_root"]) / dataset
    if primary.is_dir():
        return spec
    mirror = (
        Path("outputs/Ablation_Task23IVDPercentile_1.1/labels/p95")
        / "development_new2/labels" / dataset
    )
    if not mirror.is_dir():
        raise FileNotFoundError(
            f"missing development labels at both {primary} and {mirror}"
        )
    local_spec = copy.deepcopy(spec)
    local_spec["groups"][group_name]["label_cache_root"] = str(mirror.parent)
    return local_spec


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    spec = _load_spec(config_path)
    job_count = len(spec["datasets"]) * len(spec["candidates"])
    if job_count != 160:
        raise RuntimeError(f"expected 160 array jobs, found {job_count}")
    if _decode_job(spec, job_count - 1) != ("smokeBuoyancy", 15):
        raise RuntimeError("last array mapping changed")
    training_count = job_count * len(spec["screen_seeds"]) * 2
    if training_count != 640:
        raise RuntimeError(f"expected 640 paired trainings, found {training_count}")

    synthetic = _synthetic_record()
    raw_width = int(synthetic["raw"].shape[1])
    candidate_audit = []
    for candidate in spec["candidates"]:
        values = feature_matrix(
            {
                "raw": synthetic["raw"],
                "fmt": synthetic["fmt"],
                "reference": synthetic["reference"],
                "features": {},
            },
            str(candidate["fmt_feature"]),
            "cpu",
        )
        if values.ndim != 2 or values.shape[0] != 4:
            raise RuntimeError(f"{candidate['id']}: invalid feature shape {values.shape}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"{candidate['id']}: non-finite synthetic features")
        if values.shape[1] > raw_width:
            raise RuntimeError(
                f"{candidate['id']}: Raw-PCA width {raw_width} cannot match "
                f"FMT width {values.shape[1]}"
            )
        candidate_audit.append(_model_audit(spec, candidate, values.shape[1]))

    controls_by_feature = {
        str(candidate["fmt_feature"]): candidate
        for candidate in spec["candidates"]
    }
    real_cache_audit = []
    for dataset in spec["datasets"]:
        group_name, group = _group_for_dataset(spec, dataset)
        control_feature = str(
            spec["selection"]["absolute_fmt_guard"]["by_group"][group_name][
                "feature"
            ]
        )
        candidate = controls_by_feature[control_feature]
        local_spec = _preflight_spec_with_label_mirror(spec, dataset)
        train, validation = _load_search_splits(
            local_spec, dataset, candidate, torch.device("cpu")
        )
        checkpoint_group = _preflight_checkpoint_group(
            group, dataset, int(spec["screen_seeds"][0])
        )
        raw_stats = _frozen_raw_normalization(
            checkpoint_group, dataset, int(spec["screen_seeds"][0])
        )
        train, validation, _, stats = _normalize_train_only(
            train, validation, raw_stats=raw_stats
        )
        for split_name, split in (("train", train), ("validation", validation)):
            split_lengths = {len(value) for value in split}
            if not len(split[0]) or len(split_lengths) != 1:
                raise RuntimeError(f"{dataset}: malformed {split_name} split")
            if not np.isfinite(split[0]).all() or not np.isfinite(split[1]).all():
                raise RuntimeError(f"{dataset}: non-finite {split_name} inputs")
            if not np.all(np.isin(split[2], (0.0, 1.0))):
                raise RuntimeError(f"{dataset}: non-binary {split_name} labels")
        if train[1].shape[1] > train[0].reshape(len(train[0]), -1).shape[1]:
            raise RuntimeError(f"{dataset}: Raw-PCA cannot match control width")
        if not all(np.isfinite(value).all() for value in stats.values()):
            raise RuntimeError(f"{dataset}: non-finite normalization statistics")
        real_cache_audit.append({
            "dataset": dataset,
            "group": group_name,
            "control_feature": control_feature,
            "feature_width": int(train[1].shape[1]),
            "training_samples": int(len(train[0])),
            "validation_samples": int(len(validation[0])),
            "training_positive_fraction": float(np.mean(train[2])),
            "validation_positive_fraction": float(np.mean(validation[2])),
            "raw_checkpoint_root": str(
                checkpoint_group["raw_checkpoint_dir"]
            ),
            "base_label_root": str(
                _group_for_dataset(local_spec, dataset)[1]["label_cache_root"]
            ),
        })
        del train, validation, stats

    payload = {
        "experiment": spec["experiment"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_manifests": {
            name: {
                "path": str(Path(spec[name]["source_manifest"])),
                "sha256": _sha256(Path(spec[name]["source_manifest"])),
            }
            for name in ("exposed_training", "robust_validation")
        },
        "dataset_count": len(spec["datasets"]),
        "candidate_count": len(spec["candidates"]),
        "array_job_count": job_count,
        "paired_training_count": training_count,
        "screen_seeds": [int(value) for value in spec["screen_seeds"]],
        "candidate_audit": candidate_audit,
        "real_cache_audit": real_cache_audit,
        "absolute_fmt_guard": spec["selection"]["absolute_fmt_guard"],
        "confirmation_opened": False,
        "outer_ordinals_opened": False,
    }
    target = Path(spec["output_root"]) / "preflight_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/Verify_Task3_AnchoredFeatureDecomposition_22.1.yaml",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
