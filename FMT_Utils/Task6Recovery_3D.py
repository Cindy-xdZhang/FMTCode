"""Task6 recovery: signed Fourier tokens and latent-only geometry reconstruction."""
import numpy as np
import torch
from torch import nn

from FMT_Utils.FlowMapData_3D import cross_offsets


def signed_fmt(geometry, frequencies=6):
    """Keep signed complex coefficients and material-line identity, in radius units."""
    x = np.asarray(geometry, np.float64)
    if x.shape[1:] != (7, 32, 3) or not 1 <= frequencies <= 16:
        raise ValueError("Expected seven 32-point trajectories and 1..16 frequencies")
    relative = x[:, 1:] - x[:, :1]
    signal = np.concatenate([np.diff(x[:, :1], axis=2), np.diff(relative, axis=2)], axis=1)
    spectrum = np.fft.rfft(signal, axis=2, norm="forward")[:, :, :frequencies]
    return np.concatenate([spectrum.real.reshape(len(x), -1), spectrum.imag[:, :, 1:].reshape(len(x), -1)], 1)


def inverse_signed_fmt(token, frequencies=6):
    z = np.asarray(token, np.float64)
    real_count = 7 * frequencies * 3
    if z.shape[1] != 7 * (2 * frequencies - 1) * 3:
        raise ValueError("Wrong token dimension")
    full = np.zeros((len(z), 7, 16, 3), np.complex128)
    full[:, :, :frequencies] = z[:, :real_count].reshape(-1, 7, frequencies, 3)
    full[:, :, 1:frequencies] += 1j*z[:, real_count:].reshape(-1, 7, frequencies-1, 3)
    delta = np.fft.irfft(full, n=31, axis=2, norm="forward")
    series = np.concatenate([np.zeros((len(z), 7, 1, 3)), np.cumsum(delta, axis=2)], 2)
    return np.concatenate([series[:, :1], series[:, :1]+series[:, 1:]+cross_offsets()[None, 1:, None]], 1)


def linear_inverse_matrix(frequencies):
    dim = 7 * (2 * frequencies - 1) * 3
    origin = inverse_signed_fmt(np.zeros((1, dim)), frequencies).reshape(672)
    return inverse_signed_fmt(np.eye(dim), frequencies).reshape(dim, 672)-origin[None], origin


def pca_geometry(train_geometry):
    """Train-only principal component basis; no evaluation data fits this map."""
    x = np.asarray(train_geometry, np.float64).reshape(len(train_geometry), -1)
    mean = x.mean(0)
    x -= mean
    values, vectors = np.linalg.eigh(x.T@x/max(len(x)-1, 1))
    order = np.argsort(values)[::-1]
    return mean, vectors[:, order], np.maximum(values[order], 0.)
