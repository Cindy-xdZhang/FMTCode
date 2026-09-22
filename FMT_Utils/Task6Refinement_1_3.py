"""Reject any failed refinement level; preserve the original h/20 error bound."""
import numpy as np


def accepted_refinement(coarse, fine, finest, reasons, lengths, h):
    error = np.maximum(np.linalg.norm(coarse-fine, axis=-1).max(1),
                       np.linalg.norm(fine-finest, axis=-1).max(1))
    numerical = np.logical_or.reduce([(r >= 3).any(1) for r in reasons])
    finite = np.isfinite(coarse).all((1, 2)) & np.isfinite(fine).all((1, 2)) & np.isfinite(finest).all((1, 2))
    short = (lengths.sum(1) <= h)
    good = finite & ~numerical & ~short & (error <= h/20)
    return good, error
