"""Independent train-only PCA comparator; never imported by the neural model."""
import numpy as np

from FMT_Utils.FlowMapData_3D import cross_offsets


class GeometryPCABaseline:
    def __init__(self, train_geometry, latent_dim=192):
        x = np.array(train_geometry[:, :, 1:], dtype=np.float64, copy=True).reshape(len(train_geometry), 651)
        self.mean = x.mean(0)
        x -= self.mean
        covariance = x.T @ x / max(1, len(x) - 1)
        values, vectors = np.linalg.eigh(covariance)
        self.basis = vectors[:, np.argsort(values)[::-1][:latent_dim]]
        self.train_samples = len(x)

    def encode(self, geometry):
        x = np.asarray(geometry[:, :, 1:], dtype=np.float64).reshape(len(geometry), 651)
        return (x - self.mean) @ self.basis

    def decode(self, token):
        later = (token @ self.basis.T + self.mean).reshape(-1, 7, 31, 3)
        initial = np.broadcast_to(cross_offsets()[None, :, None], (len(token), 7, 1, 3))
        return np.concatenate([initial, later], axis=2).astype(np.float32)

    def predict(self, geometry):
        return self.decode(self.encode(geometry))
