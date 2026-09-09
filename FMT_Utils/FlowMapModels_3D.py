"""Frozen feature adapters and capacity-matched query networks, Task678 v1.1."""
from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from FMT_Utils.DFT_FMT_3D import (pathline_dft_features_3d,
    pathline_velocity_gradient_dft_features_3d, pathline_anchored_kinematic_dft_features_3d)
from FMT_Utils.FMTAllV2_3D import fmt_all_v2
from FMT_Utils.LargeNeighbor_3D import large_neighbor_fmt


def raw_features(paths, radii, arm):
    """Each call is one physical time/visible-source group for the IVD mean."""
    x = np.asarray(paths, np.float32)
    if x.ndim != 4 or x.shape[1:] != (31, 32, 3):
        raise ValueError("Expected the 31-line union of old cross and old shells")
    local = x - x[:, :1, :1]
    unit = local / np.asarray(radii)[:, None, None, None]
    cross = local[:, :7]
    shell = local[:, [0] + list(range(7, 31))]
    if arm in ("raw_pca7", "raw_vae7"):
        result = unit[:, :7].reshape(len(x), -1)
    elif arm == "raw_pca25":
        result = unit[:, [0] + list(range(7, 31))].reshape(len(x), -1)
    elif arm == "center_pca":
        result = unit[:, 0].reshape(len(x), -1)
    elif arm == "coordinates_only":
        result = np.zeros((len(x), 1), np.float32)
    elif arm in ("fmt_all", "fmt_all_kin4"):
        tensor = torch.from_numpy(cross.copy())
        result = pathline_dft_features_3d(tensor)
        if arm == "fmt_all_kin4":
            result = np.concatenate([result, pathline_velocity_gradient_dft_features_3d(tensor, num_freq=4)], 1)
    elif arm == "fmt_all_v2":
        result = fmt_all_v2(cross)
    elif arm == "largeNeighbor":
        result = large_neighbor_fmt(shell)
    elif arm == "aivd1w3_dft":
        result = pathline_anchored_kinematic_dft_features_3d(torch.from_numpy(cross.copy()),
            num_freq=1, window=3, channels=(0,), log_compress=False, endpoint_order=1,
            anchor_names=(), include_dft=True)
    elif arm == "coordinate_fourier":
        coeff = np.fft.rfft(unit[:, :7], axis=2)[:, :, :6]
        result = np.concatenate([coeff.real.reshape(len(x), -1), coeff[:, :, 1:].imag.reshape(len(x), -1)], 1)
    elif arm == "differential_geometry":
        a = unit[:, :7].astype(float)
        v = np.gradient(a, axis=2)
        acceleration = np.gradient(v, axis=2)
        jerk = np.gradient(acceleration, axis=2)
        speed = np.linalg.norm(v, axis=-1)
        cross_va = np.cross(v, acceleration)
        numerator = np.linalg.norm(cross_va, axis=-1)
        curvature = np.divide(numerator, speed ** 3, out=np.zeros_like(speed), where=speed > 1e-8)
        torsion = np.divide((cross_va * jerk).sum(-1), numerator ** 2,
                            out=np.zeros_like(speed), where=numerator > 1e-10)
        # Signed log stabilizes legitimate large torsion near straight paths.
        geometry = np.stack([np.log1p(curvature), np.sign(torsion) * np.log1p(np.abs(torsion))], -1)
        length = np.linalg.norm(np.diff(a, axis=2), axis=-1).sum(-1)
        result = np.concatenate([geometry.reshape(len(x), -1), length, speed.mean(-1), speed.std(-1)], 1)
    else:
        raise ValueError(arm)
    result = np.asarray(result, np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {arm}; do not drop samples per method")
    return result


class VAE(nn.Module):
    def __init__(self, dim, latent, hidden):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(nn.Linear(dim, h1), nn.SiLU(), nn.Linear(h1, h2), nn.SiLU())
        self.mu, self.logvar = nn.Linear(h2, latent), nn.Linear(h2, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, h2), nn.SiLU(), nn.Linear(h2, h1), nn.SiLU(), nn.Linear(h1, dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-12, 12)


class TokenTransform:
    """Fit only training inputs, and explicitly count effective token storage."""
    def __init__(self, width, seed, arm, device):
        self.width, self.seed, self.arm = width, seed, arm
        self.device = torch.device(device)
        self.scaler, self.pca, self.vae = StandardScaler(), None, None

    def fit(self, features, vae_settings):
        started = time.perf_counter()
        values = self.scaler.fit_transform(features).astype(np.float32)
        self.native_dim = values.shape[1]
        self.effective_dim = min(self.width, values.shape[1], len(values) - 1)
        if self.arm == "coordinates_only":
            self.effective_dim = 0
        if self.arm == "raw_vae7":
            torch.manual_seed(self.seed)
            self.vae = VAE(values.shape[1], self.width, vae_settings["hidden_dims"]).to(self.device)
            optim = torch.optim.Adam(self.vae.parameters(), lr=vae_settings["learning_rate"])
            source = torch.as_tensor(values, device=self.device)
            for _ in range(vae_settings["optimizer_steps"]):
                batch = source[torch.randint(len(source), (vae_settings["batch_size"],), device=self.device)]
                mu, logvar = self.vae.encode(batch)
                z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
                reconstruction = self.vae.decoder(z)
                loss = (reconstruction - batch).square().mean() + vae_settings["beta"] * .5 * (mu.square() + logvar.exp() - logvar - 1).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite VAE loss")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            self.vae.eval()
            self.effective_dim = self.width
        elif values.shape[1] > self.width:
            if self.effective_dim != self.width:
                raise ValueError("Too few training samples for the registered token width")
            self.pca = PCA(self.width, svd_solver="randomized", random_state=self.seed)
            self.pca.fit(values)
        self.fit_seconds = time.perf_counter() - started
        return self

    def transform(self, features):
        values = self.scaler.transform(features).astype(np.float32)
        if self.vae is not None:
            with torch.no_grad():
                values = np.concatenate([self.vae.encode(torch.as_tensor(values[i:i+2048], device=self.device))[0].cpu().numpy()
                                         for i in range(0, len(values), 2048)])
        elif self.pca is not None:
            values = self.pca.transform(values).astype(np.float32)
        if self.arm == "coordinates_only":
            return np.zeros((len(values), self.width), np.float32)
        return np.pad(values, ((0, 0), (0, self.width - values.shape[1]))).astype(np.float32)

    def metadata(self):
        state_bytes = sum(a.nbytes for a in (self.scaler.mean_, self.scaler.scale_))
        if self.pca is not None:
            state_bytes += self.pca.components_.nbytes + self.pca.mean_.nbytes
        return dict(native_dimension=self.native_dim, network_token_width=self.width,
                    effective_token_dimension=self.effective_dim, token_bytes=4 * self.effective_dim,
                    input_material_lines=25 if self.arm in ("raw_pca25", "largeNeighbor") else (0 if self.arm == "coordinates_only" else (1 if self.arm == "center_pca" else 7)),
                    preprocessing_state_bytes=int(state_bytes), fit_seconds=self.fit_seconds,
                    vae_parameter_count=sum(p.numel() for p in self.vae.parameters()) if self.vae is not None else 0,
                    pca_fitted_on="training visible supports only", scalar_padding_contains_no_extra_information=True)


class FlowMapDecoder(nn.Module):
    """One capacity-matched network, with permutation-invariant visible context."""
    def __init__(self, width, hidden, output_scale=1.):
        super().__init__()
        self.support = nn.Sequential(nn.Linear(width + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.query = nn.Sequential(nn.Linear(hidden + 6, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.output_scale = float(output_scale)

    def forward(self, tokens, geometry, query, tau, scales):
        context = self.support(torch.cat([tokens, geometry], -1)).mean(1)
        displacement = self.query(torch.cat([context, query, tau[:, None], scales], -1))
        return query + tau[:, None] * self.output_scale * displacement


def training_arrays(data, tokens, task):
    """Package inputs without using any future target as a network input."""
    if task == "Task6":
        origin = np.concatenate([data["origin0"], data["origin1"]])
        radius = np.concatenate([data["radius0"], data["radius1"]])
        duration = np.tile(data["duration"], 2)
        targets = np.concatenate([data["target0"], data["target1"]])
        token = np.concatenate([tokens["support0"], tokens["support1"]])[:, None]
        geometry = np.zeros((len(origin), 1, 4), np.float32)
        geometry[:, :, 3] = 1.
    else:
        origin, radius, duration = data["origin0"], data["radius0"], data["duration"]
        targets, token = data["target0"], tokens["context"]
        pos = (data["context_origins"] - origin[:, None]) / radius[:, None, None]
        geometry = np.concatenate([pos, np.ones((*pos.shape[:2], 1))], -1).astype(np.float32)
    # Query seeds are known initial conditions, not observed future positions.
    query = ((targets[:, :, 0] - origin[:, None]) / radius[:, None, None]).astype(np.float32)
    target = ((targets - origin[:, None, None]) / radius[:, None, None, None]).astype(np.float32)
    scales = np.stack([np.log(radius), np.log(duration)], -1).astype(np.float32)
    return dict(tokens=token, geometry=geometry, query=query, scales=scales,
                target=target, origin=origin, radius=radius, duration=duration)


def train_decoder(arrays, width, settings, seed, device):
    torch.manual_seed(seed)
    started = time.perf_counter()
    displacement = arrays["target"] - arrays["query"][:, :, None]
    scale = max(float(np.sqrt(np.mean(displacement ** 2))), 1e-4)
    model = FlowMapDecoder(width, settings["hidden_width"], scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    data = {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device) for k in ("tokens", "geometry", "query", "scales", "target")}
    n, q, t, _ = data["target"].shape
    for step in range(settings["optimizer_steps"]):
        i = torch.randint(n, (settings["batch_size"],), device=device)
        j = torch.randint(q, (settings["batch_size"],), device=device)
        k = torch.randint(1, t, (settings["batch_size"],), device=device)
        pred = model(data["tokens"][i], data["geometry"][i], data["query"][i, j], k.float() / (t - 1), data["scales"][i])
        loss = ((pred - data["target"][i, j, k]) / scale).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite decoder loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
    model.eval()
    return model, dict(optimizer_steps=settings["optimizer_steps"], last_training_loss=float(loss.detach().cpu()),
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       training_output_scale=scale, fit_seconds=time.perf_counter() - started)


def predict(model, arrays, device, query=None, tau=None, batch_size=8192):
    query = arrays["query"] if query is None else np.asarray(query, np.float32)
    tau = np.linspace(0, 1, arrays["target"].shape[2], dtype=np.float32) if tau is None else np.asarray(tau, np.float32)
    n, q, _ = query.shape
    total = n * q * len(tau)
    source = {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device) for k in ("tokens", "geometry", "scales")}
    queries = torch.as_tensor(query, device=device)
    times = torch.as_tensor(tau, device=device)
    values = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            index = torch.arange(start, min(start + batch_size, total), device=device)
            i = index // (q * len(tau))
            j = (index // len(tau)) % q
            k = index % len(tau)
            values.append(model(source["tokens"][i], source["geometry"][i], queries[i, j], times[k], source["scales"][i]).cpu().numpy())
    return np.concatenate(values).reshape(n, q, len(tau), 3) * arrays["radius"][:, None, None, None] + arrays["origin"][:, None, None]


def split_stages(arrays):
    n = len(arrays["origin"]) // 2
    return ({k: v[:n] for k, v in arrays.items()}, {k: v[n:] for k, v in arrays.items()})


def compose(model, arrays, device):
    first, second = split_stages(arrays)
    a = predict(model, first, device)
    query_second = (a[:, :, -1] - second["origin"][:, None]) / second["radius"][:, None, None]
    b = predict(model, second, device, query=query_second)
    covered = (np.abs(query_second).sum(-1) <= 1.)
    return np.concatenate([a, b[:, :, 1:]], 2), dict(predicted_arrival_outside_fraction=float((~covered).mean()),
                                                   second_query_uses="predicted_first_stage_endpoint")
