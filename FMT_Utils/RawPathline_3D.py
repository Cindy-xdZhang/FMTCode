"""Label-free representations derived directly from 3D pathline primitives."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def raw_pathline_representation(flat, name="positions"):
    """Transform flattened local 7-line primitives without Fourier encoding."""
    pathlines = np.asarray(flat, dtype=np.float32).reshape(len(flat), 7, -1, 3)
    center = pathlines[:, :1]
    relative = pathlines[:, 1:] - center
    center_delta = np.diff(center, axis=2)
    relative_delta = np.diff(relative, axis=2)
    if name == "positions":
        values = pathlines
    elif name == "center":
        values = center
    elif name == "center_delta":
        values = center_delta
    elif name == "relative":
        values = relative
    elif name == "relative_delta":
        values = relative_delta
    elif name == "center_relative":
        return np.concatenate(
            (center.reshape(len(flat), -1), relative.reshape(len(flat), -1)), axis=1
        ).astype(np.float32)
    elif name == "center_delta_relative":
        return np.concatenate(
            (center_delta.reshape(len(flat), -1), relative.reshape(len(flat), -1)), axis=1
        ).astype(np.float32)
    elif name == "center_relative_delta":
        return np.concatenate(
            (center.reshape(len(flat), -1), relative_delta.reshape(len(flat), -1)), axis=1
        ).astype(np.float32)
    elif name == "dynamics":
        return np.concatenate(
            (center_delta.reshape(len(flat), -1), relative_delta.reshape(len(flat), -1)), axis=1
        ).astype(np.float32)
    elif name == "relative_distance":
        values = np.linalg.norm(relative, axis=-1)
    elif name == "relative_distance_delta":
        values = np.diff(np.linalg.norm(relative, axis=-1), axis=2)
    elif name in {"pair_distance", "pair_distance_delta", "invariant_dynamics"}:
        pair_index = np.asarray([(a, b) for a in range(7) for b in range(a + 1, 7)])
        pair_distance = np.linalg.norm(
            pathlines[:, pair_index[:, 0]] - pathlines[:, pair_index[:, 1]], axis=-1
        )
        if name == "pair_distance":
            values = pair_distance
        elif name == "pair_distance_delta":
            values = np.diff(pair_distance, axis=2)
        else:
            center_speed = np.linalg.norm(center_delta[:, 0], axis=-1)
            return np.concatenate(
                (center_speed, np.diff(pair_distance, axis=2).reshape(len(flat), -1)), axis=1
            ).astype(np.float32)
    else:
        raise ValueError(f"unknown representation: {name}")
    return values.reshape(len(flat), -1).astype(np.float32)


def raw_representation_group_split(name, sampled_steps=32):
    """Return the first/second group boundary used for post-scale weighting."""
    length = int(sampled_steps)
    if name in {"positions", "center_relative"}:
        return length * 3
    if name in {"center_delta_relative", "dynamics"}:
        return (length - 1) * 3
    if name == "center_relative_delta":
        return length * 3
    if name == "pair_distance":
        return 6 * length
    if name == "pair_distance_delta":
        return 6 * (length - 1)
    raise ValueError(f"representation {name!r} has no two-group weighting contract")


def normalize_raw_train_eval(train, evaluate, representation, sampled_steps,
                             mode="standard"):
    """Fit label-free raw-feature normalization on train data and transform both sets."""
    train = np.asarray(train, dtype=np.float32)
    evaluate = np.asarray(evaluate, dtype=np.float32)
    split = raw_representation_group_split(representation, sampled_steps)

    def rms(values):
        scale = np.sqrt(np.square(values).mean(axis=1, keepdims=True)).clip(1e-8)
        return values / scale

    if mode == "pre_sample_rms":
        train, evaluate = rms(train), rms(evaluate)
    elif mode == "pre_group_rms":
        train = np.concatenate((rms(train[:, :split]), rms(train[:, split:])), axis=1)
        evaluate = np.concatenate(
            (rms(evaluate[:, :split]), rms(evaluate[:, split:])), axis=1
        )
    elif mode not in {"standard", "post_sample_rms", "post_group_rms"}:
        raise ValueError(f"unknown normalization: {mode}")

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train).astype(np.float32)
    evaluate = scaler.transform(evaluate).astype(np.float32)
    if mode == "post_sample_rms":
        train, evaluate = rms(train), rms(evaluate)
    elif mode == "post_group_rms":
        train = np.concatenate((rms(train[:, :split]), rms(train[:, split:])), axis=1)
        evaluate = np.concatenate(
            (rms(evaluate[:, :split]), rms(evaluate[:, split:])), axis=1
        )
    return train.astype(np.float32), evaluate.astype(np.float32)
