"""Label-free k-means read-out rules, and the metric used to score them.

`docs/2d_kmeans_d8.md` found that on the 2D D8 latent the single largest
improvement was not in training at all but in *which* k-means partition is kept:
inertia discards the good partition by construction (the vortex/non-vortex split
is the well-separated one, not the tightest), and selecting by silhouette instead
was worth +0.20 macro-F1 on both datasets.

This module runs the same family of rules in 3D so the claim can be retested
rather than assumed.  Every rule here is **label-free**; the reference is used
only to score the partition after it has been chosen.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             f1_score, silhouette_score)
from sklearn.mixture import GaussianMixture

RULES = ("inertia", "davies_bouldin", "silhouette", "calinski_harabasz", "oracle")
CLUSTERERS = ("kmeans", "gmm_tied", "gmm_full", "gmm_diag", "spectral",
              "agglomerative_average", "agglomerative_complete")


def _fit_partition(points, clusters, clusterer, seed):
    """One restart of the requested clusterer; returns (labels, inertia).

    `gmm_tied` is k-means under a Mahalanobis metric -- one pooled within-cluster
    covariance, estimated jointly with the partition.  The 2D study found it the
    single largest read-out improvement, because the vortex split is the
    well-separated direction but not the tightest, which is exactly where the
    Euclidean metric prefers the wrong partition.
    """
    if clusterer == "kmeans":
        model = KMeans(n_clusters=clusters, n_init=1, random_state=seed).fit(points)
        return model.labels_, float(model.inertia_)
    if clusterer.startswith("gmm_"):
        model = GaussianMixture(n_components=clusters, covariance_type=clusterer[4:],
                                n_init=1, random_state=seed,
                                reg_covar=1e-6).fit(points)
        labels = model.predict(points)
        return labels, float(-model.score(points))          # lower is better
    if clusterer == "spectral":
        labels = SpectralClustering(n_clusters=clusters, affinity="nearest_neighbors",
                                    n_neighbors=10, random_state=seed,
                                    assign_labels="kmeans").fit_predict(points)
    else:
        linkage = clusterer.split("_")[1]
        labels = AgglomerativeClustering(n_clusters=clusters,
                                         linkage=linkage).fit_predict(points)
    centres = np.stack([points[labels == k].mean(0) for k in range(clusters)])
    inertia = float(((points - centres[labels]) ** 2).sum())
    return labels, inertia


def macro_f1(partition, reference):
    """Macro-F1 with the cluster->class map chosen to maximise it.

    With two clusters there are two maps; taking the better one is the standard
    unsupervised convention and uses no label at selection time -- the map is
    part of the *metric*, not of the method.
    """
    reference = np.asarray(reference).astype(int)
    direct = f1_score(reference, partition, average="macro", zero_division=0)
    flipped = f1_score(reference, 1 - partition, average="macro", zero_division=0)
    return float(max(direct, flipped))


def l2_normalize(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def cluster_readouts(latent, reference, clusters=2, restarts=10, seed=0,
                     silhouette_sample=4096, normalize=True, clusterer="gmm_tied"):
    """Fit ``restarts`` k-means partitions; score each by every criterion.

    Returns ``(scores, diagnostics)`` where ``scores`` maps a rule name to the
    macro-F1 of the partition that rule selects, and ``diagnostics`` carries the
    per-candidate criteria so the rules can be correlated against F1 afterwards.
    """
    points = l2_normalize(latent) if normalize else np.asarray(latent, dtype=np.float64)
    rng = np.random.default_rng(seed)
    candidates = []
    for index in range(int(restarts)):
        try:
            partition, inertia = _fit_partition(points, int(clusters), clusterer,
                                                int(seed) + index)
        except Exception:
            continue
        if len(np.unique(partition)) < 2:
            continue
        sample = None
        if silhouette_sample and len(points) > silhouette_sample:
            sample = int(silhouette_sample)
        candidates.append({
            "partition": partition,
            "inertia": float(inertia),
            "davies_bouldin": float(davies_bouldin_score(points, partition)),
            "silhouette": float(silhouette_score(points, partition, sample_size=sample,
                                                 random_state=int(seed))),
            "calinski_harabasz": float(calinski_harabasz_score(points, partition)),
            "minority_fraction": float(min(np.bincount(partition, minlength=2)) / len(partition)),
            "f1": macro_f1(partition, reference),
        })
    if not candidates:
        return {rule: 0.0 for rule in RULES}, []

    def pick(key, largest):
        values = [c[key] for c in candidates]
        return candidates[int(np.argmax(values) if largest else np.argmin(values))]

    scores = {
        "inertia": pick("inertia", False)["f1"],
        "davies_bouldin": pick("davies_bouldin", False)["f1"],
        "silhouette": pick("silhouette", True)["f1"],
        "calinski_harabasz": pick("calinski_harabasz", True)["f1"],
        "oracle": max(c["f1"] for c in candidates),
    }
    diagnostics = [{k: v for k, v in c.items() if k != "partition"} for c in candidates]
    return scores, diagnostics


def rule_correlations(diagnostics):
    """Within-evaluation Spearman of each criterion against F1.

    The 2D study used this to show Davies-Bouldin is uninformative on one dataset
    and anti-informative on the other, while silhouette correlates ~+0.89.
    """
    from scipy.stats import spearmanr
    if len(diagnostics) < 3:
        return {}
    f1 = [d["f1"] for d in diagnostics]
    result = {}
    for key, sign in (("silhouette", 1.0), ("davies_bouldin", -1.0),
                      ("inertia", -1.0), ("calinski_harabasz", 1.0)):
        values = [sign * d[key] for d in diagnostics]
        if np.std(values) < 1e-12:
            continue
        result[key] = float(spearmanr(values, f1).statistic)
    return result
