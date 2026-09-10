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
    x = np.array(train_geometry, dtype=np.float64, copy=True).reshape(len(train_geometry), -1)
    mean = x.mean(0)
    x -= mean
    values, vectors = np.linalg.eigh(x.T@x/max(len(x)-1, 1))
    order = np.argsort(values)[::-1]
    return mean, vectors[:, order], np.maximum(values[order], 0.)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers=nn.Sequential(nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))

    def forward(self,x):
        return x+.1*self.layers(x)


class InitializedGeometryVAE(nn.Module):
    """Train-fitted linear latent map plus learned residuals; decoder sees only z.

    PCA provides a stable self-supervised initialization. Every reconstructed
    degree of freedom still passes through the configured latent bottleneck.
    The linear maps are buffers frozen after train-only fitting; both residual
    networks and the posterior variance are trained by the geometry VAE loss.
    """
    def __init__(self,input_mean,input_std,encode_weight,encode_bias,decode_weight,decode_bias,
                 width=512,blocks=3,initial_logvar=-18.):
        super().__init__()
        for name,value in dict(input_mean=input_mean,input_std=input_std,encode_weight=encode_weight,
            encode_bias=encode_bias,decode_weight=decode_weight,decode_bias=decode_bias).items():
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        input_dim,latent_dim=encode_weight.shape
        self.encoder_residual=nn.Sequential(nn.Linear(input_dim,width),*[ResidualBlock(width) for _ in range(blocks)],nn.SiLU(),nn.Linear(width,latent_dim))
        self.decoder_residual=nn.Sequential(nn.Linear(latent_dim,width),*[ResidualBlock(width) for _ in range(blocks)],nn.SiLU(),nn.Linear(width,672))
        for layer in (self.encoder_residual[-1],self.decoder_residual[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.posterior_logvar=nn.Parameter(torch.full((latent_dim,),initial_logvar))
        self.register_buffer("initial_seeds",torch.as_tensor(cross_offsets(),dtype=torch.float32))

    def encode(self,x):
        standardized=(x-self.input_mean)/self.input_std
        mu=standardized@self.encode_weight+self.encode_bias+.01*self.encoder_residual(standardized)
        return mu,self.posterior_logvar.clamp(-24.,0.).expand_as(mu)

    def decode(self,z):
        out=z@self.decode_weight+self.decode_bias+self.decoder_residual(z)
        out=out.reshape(-1,7,32,3)
        # The initial cross is fixed by the primitive contract, not supplied by GT.
        return torch.cat([self.initial_seeds[None,:,None].expand(len(z),-1,-1,-1),out[:,:,1:]],2)

    def forward(self,x,sample=True):
        mu,logvar=self.encode(x)
        z=mu+torch.randn_like(mu)*torch.exp(.5*logvar) if sample else mu
        return self.decode(z),mu,logvar


def initialize_vae(inputs,geometry,arm,latent_dim,frequencies=16,width=512,blocks=3):
    """Fit all initialization statistics on this flow's training primitives only."""
    input_mean=inputs.mean(0,dtype=np.float64)
    input_std=inputs.std(0,dtype=np.float64)
    input_std[input_std<1e-6]=1.
    geo_mean,basis,variance=pca_geometry(geometry)
    basis=basis[:,:latent_dim]
    latent_scale=np.maximum(np.sqrt(variance[:latent_dim]),1e-3)
    if arm=="signed_fmt_vae":
        inverse,initial=linear_inverse_matrix(frequencies)
        encode_weight=(input_std[:,None]*inverse)@basis/latent_scale
        encode_bias=(input_mean@inverse+initial-geo_mean)@basis/latent_scale
    elif arm=="raw_vae":
        encode_weight=(input_std[:,None]*basis)/latent_scale
        encode_bias=(input_mean-geo_mean)@basis/latent_scale
    elif arm=="fmt_all_vae":
        # Original nonlinear invariant token has no linear inverse. Fit a linear
        # initializer from train-only token/geometry pairs, then learn residuals.
        x=(inputs.astype(np.float64)-input_mean)/input_std
        scores=((geometry.reshape(len(geometry),672)-geo_mean)@basis)/latent_scale
        covariance=x.T@x
        ridge=1e-6*np.trace(covariance)/max(1,len(covariance))
        encode_weight=np.linalg.solve(covariance+np.eye(len(covariance))*max(ridge,1e-8),x.T@scores)
        encode_bias=np.zeros(latent_dim)
    else:
        raise ValueError(arm)
    model=InitializedGeometryVAE(input_mean,input_std,encode_weight,encode_bias,
        latent_scale[:,None]*basis.T,geo_mean,width,blocks)
    info=dict(input_dim=inputs.shape[1],latent_dim=latent_dim,frequencies=frequencies if arm=="signed_fmt_vae" else None,
        train_samples=len(inputs),pca_eigenvalues=variance[:latent_dim].tolist(),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        fitted_buffer_values=sum(b.numel() for b in model.buffers()),
        initialization="train-only PCA; original invariant arm additionally uses train-only ridge regression")
    return model,info
