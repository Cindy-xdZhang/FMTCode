"""Display-only diagnostics of frozen, signed hairpin intervention scores."""
from __future__ import annotations
import numpy as np


def summarize_bundle(values, count):
    indices = np.asarray(values['patch_indices'], dtype=np.int64)
    delta = np.asarray(values['patch_margin_delta'], dtype=np.float64)
    p = float(values['probability'])
    margin = float(values['margin'])
    changed_p = np.asarray(values['patch_probability'], dtype=np.float64)
    changed_margin = margin - delta
    exp = np.exp(-np.abs(changed_margin))
    sigmoid = np.where(changed_margin >= 0, 1 / (1 + exp), exp / (1 + exp))
    if not np.allclose(sigmoid, changed_p, atol=2e-7, rtol=0):
        raise ValueError('Saved intervention probabilities do not match the signed class margin')
    if not np.allclose(p - changed_p, values['patch_probability_delta'], atol=1e-12, rtol=0):
        raise ValueError('Probability delta has the wrong sign or source')
    positive = np.flatnonzero(delta > 0)
    positive = positive[np.argsort(-delta[positive], kind='stable')]
    negative = np.flatnonzero(delta < 0)
    negative = negative[np.argsort(delta[negative], kind='stable')]
    touched = np.zeros((count, 32), dtype=bool)
    for i in positive:
        line, start, stop = indices[i]
        touched[line, start + 1:stop] = True
    previous = np.asarray(values['local_shape_delta'])[:count]
    cancellation = float(np.mean(previous[touched] <= 0)) if touched.any() else 0.
    return dict(positive_order=positive.tolist(), negative_order=negative.tolist(),
        strongest_support=int(positive[0]) if len(positive) else None,
        strongest_support_delta=float(delta[positive[0]]) if len(positive) else 0.,
        positive_windows=int(len(positive)), negative_windows=int(len(negative)),
        previous_absolute_max_was_negative=bool(delta[np.argmax(np.abs(delta))] < 0),
        positive_window_covered_points_with_nonpositive_old_color_fraction=cancellation,
        changed_probability_max_formula_error=float(np.max(np.abs(sigmoid - changed_p))),
        windows=len(delta))
