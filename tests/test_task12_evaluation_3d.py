import numpy as np

from FMT_Utils.Task12Evaluation_3D import (
    binary_cluster_metrics, calibrate_vortex_cluster, fit_kmeans_transform,
)


def test_cluster_orientation_is_calibrated_outside_final_metrics():
    reference = np.array([0, 0, 1, 1], dtype=bool)
    labels = np.array([1, 1, 0, 0])
    cluster = calibrate_vortex_cluster(reference, labels)
    assert cluster == 0
    assert binary_cluster_metrics(reference, labels, cluster)["f1"] == 1.0


def test_train_only_transform_predicts_two_clusters():
    train = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    fitted = fit_kmeans_transform(train, pca_dim=1, random_state=3, n_init=5)
    labels = fitted.predict(train)
    assert set(labels.tolist()) == {0, 1}
