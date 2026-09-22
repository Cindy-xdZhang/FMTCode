"""Paired label/split diagnosis; frozen geometry and no validation partition."""
import numpy as np
from scipy.spatial import cKDTree


def distance_allowed(seeds, test, scale):
    """Exclude training centers within each test center's own local 1h."""
    allowed = np.ones(len(seeds), bool)
    tree = cKDTree(seeds)
    for ids in tree.query_ball_point(seeds[test], scale[test]*(1-1e-12)):
        allowed[ids] = False
    allowed[test] = False
    return allowed


def choose_training(pool, owners, test_owners, number, rng, shared):
    pool = np.asarray(pool, np.int64)
    if len(pool) < number:
        raise ValueError(f'Insufficient training centers: {len(pool)} < {number}')
    mandatory = []
    if shared:
        # Coverage is a split requirement, independent of either binary label.
        for owner in np.unique(test_owners):
            candidates = pool[owners[pool] == owner]
            if not len(candidates):
                raise ValueError(f'No distant shared-instance training center for {owner}')
            mandatory.append(int(rng.choice(candidates)))
    mandatory = np.asarray(mandatory, np.int64)
    rest = np.setdiff1d(pool, mandatory)
    result = np.sort(np.r_[mandatory, rng.choice(rest, number-len(mandatory), replace=False)])
    assert len(np.unique(result)) == number
    overlap = set(owners[result]) & set(test_owners)
    assert overlap == (set(test_owners) if shared else set())
    return result


def loss_weights(labels):
    """Fixed target risk with 1/3 positives, preserving identical sample IDs."""
    counts = np.bincount(labels, minlength=2)
    if counts.min() == 0:
        raise ValueError('Both classes are required')
    return np.array([2/3, 1/3])*len(labels)/counts


def score(labels, probability):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, bool); pred = np.asarray(probability) >= .5
    tp = int((labels & pred).sum()); fn = int((labels & ~pred).sum())
    fp = int((~labels & pred).sum()); tn = int((~labels & ~pred).sum())
    recall = tp/max(tp+fn, 1); fpr = fp/max(fp+tn, 1)
    prior = 1/3
    precision_fixed = prior*recall/max(prior*recall+(1-prior)*fpr, 1e-30)
    return dict(samples=len(labels), positive=int(labels.sum()), tp=tp, fn=fn, fp=fp, tn=tn,
        precision=tp/max(tp+fp, 1), recall=recall, f1=2*tp/max(2*tp+fp+fn, 1),
        fpr=fpr, f1_prior_one_third=2*precision_fixed*recall/max(precision_fixed+recall, 1e-30),
        average_precision=float(average_precision_score(labels, probability)),
        roc_auc=float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else None)
