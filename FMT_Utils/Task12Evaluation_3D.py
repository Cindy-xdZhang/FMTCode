"""Shared, leakage-resistant evaluation helpers for 3D Task1 and Task2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


def binary_cluster_metrics(reference, labels, vortex_cluster):
    """Score anonymous binary cluster labels using a pre-calibrated cluster ID."""
    reference = np.asarray(reference, dtype=bool)
    labels = np.asarray(labels)
    prediction = labels == int(vortex_cluster)
    return {
        "f1": float(f1_score(reference, prediction, zero_division=0)),
        "iou": float(jaccard_score(reference, prediction, zero_division=0)),
        "precision": float(precision_score(reference, prediction, zero_division=0)),
        "recall": float(recall_score(reference, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(reference, prediction)),
        "ari": float(adjusted_rand_score(reference, labels)),
        "nmi": float(normalized_mutual_info_score(reference, labels)),
        "positive_fraction": float(reference.mean()),
        "predicted_positive_fraction": float(prediction.mean()),
    }


def calibrate_vortex_cluster(reference, labels):
    """Choose the vortex cluster on calibration labels, never on final-test labels."""
    scored = [
        (binary_cluster_metrics(reference, labels, cluster)["f1"], cluster)
        for cluster in (0, 1)
    ]
    return int(max(scored, key=lambda item: (item[0], -item[1]))[1])


@dataclass
class KMeansTransform3D:
    scaler: StandardScaler
    pca: PCA | None
    model: KMeans

    def transform(self, values):
        transformed = self.scaler.transform(np.asarray(values, dtype=np.float32))
        if self.pca is not None:
            transformed = self.pca.transform(transformed)
        return np.asarray(transformed, dtype=np.float32)

    def predict(self, values):
        return self.model.predict(self.transform(values))


def fit_kmeans_transform(
    train_features,
    pca_dim=None,
    random_state=7068,
    n_init=20,
):
    """Fit all feature transforms and KMeans using training features only."""
    train_features = np.asarray(train_features, dtype=np.float32)
    scaler = StandardScaler().fit(train_features)
    transformed = scaler.transform(train_features)
    pca = None
    if pca_dim is not None:
        pca_dim = int(pca_dim)
        maximum = min(transformed.shape[0] - 1, transformed.shape[1])
        if not 1 <= pca_dim <= maximum:
            raise ValueError(f"pca_dim={pca_dim} is invalid for {transformed.shape}")
        pca = PCA(
            n_components=pca_dim,
            svd_solver="randomized",
            random_state=int(random_state),
        ).fit(transformed)
        transformed = pca.transform(transformed)
    model = KMeans(
        n_clusters=2,
        random_state=int(random_state),
        n_init=int(n_init),
    ).fit(transformed)
    return KMeansTransform3D(scaler=scaler, pca=pca, model=model)
