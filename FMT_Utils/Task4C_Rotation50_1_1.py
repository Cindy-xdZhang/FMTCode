"""Prediction-independent fold sampling with explicit GT-head coverage."""
import numpy as np


def select_test(good, folds, owners, instances, held_out, count, seed, fi, allow_native_head):
    eligible = np.flatnonzero(good & (folds == held_out))
    key = [seed, fi, 11] if held_out == 0 else [seed, fi, 11, held_out]
    rng = np.random.default_rng(key)
    selected = np.sort(rng.choice(eligible, count, replace=False))
    expected = np.unique(instances[folds == held_out])
    exceptions = []
    for inst in expected:
        if np.any(owners[selected] == inst):
            continue
        pool = np.flatnonzero((folds == held_out) & good & (owners == inst))
        if not len(pool):
            if not allow_native_head:
                raise ValueError(f'No old-candidate GT head for flow={fi}, fold={held_out}, instance={inst}')
            pool = np.flatnonzero((folds == held_out) & (owners == inst))
            if not len(pool):
                raise ValueError(f'No native GT head for instance={inst}')
            exceptions.append(dict(flow=int(fi), fold=int(held_out), instance=int(inst)))
        add = int(rng.choice(pool))
        counts = {int(i): int((owners[selected] == i).sum()) for i in expected}
        removable = [int(i) for i in selected if int(owners[i]) not in counts or counts[int(owners[i])] > 1]
        drop = int(rng.choice(removable))
        selected = np.sort(np.r_[selected[selected != drop], add])
    assert len(selected) == count and len(np.unique(selected)) == count
    assert all(np.any(owners[selected] == inst) for inst in expected)
    assert (folds[selected] == held_out).all()
    return selected, exceptions
