"""Diagnose FMT-VAE information loss across half-cylinder Reynolds numbers."""

from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from DeepUtils.utils import EasyConfig
from FMT_Utils.VAE_3D import FeatureVAE3D, vae_loss
from Run_Task2_Universality import _fit_cluster, _load_slices, _prepare


SLOT_WIDTH = 23  # 6 frequencies * (real norm, imaginary norm, cosine) + 5 chirality


def _block_indices(feature_dim):
    if feature_dim % SLOT_WIDTH:
        raise ValueError(f"FMT feature dimension {feature_dim} is not divisible by {SLOT_WIDTH}")
    line_blocks = feature_dim // SLOT_WIDTH
    slots = np.arange(feature_dim).reshape(line_blocks, SLOT_WIDTH)
    gram = np.arange(18).reshape(6, 3)
    return {
        "center": slots[0].ravel(),
        "neighbors": slots[1:].ravel(),
        "real_norm": slots[:, gram[:, 0]].ravel(),
        "imag_norm": slots[:, gram[:, 1]].ravel(),
        "cosine": slots[:, gram[:, 2]].ravel(),
        "chirality": slots[:, 18:23].ravel(),
    }


def _train(train_x, test_x, latent_dim, source, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    task = source.task2
    model = FeatureVAE3D(train_x.shape[1], task.hidden_dims, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(task.learning_rate),
                                  weight_decay=float(task.weight_decay))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x)),
                        batch_size=int(task.batch_size), shuffle=True,
                        generator=generator, drop_last=False)
    epochs = math.ceil(int(task.target_optimizer_steps) / math.ceil(len(train_x) / int(task.batch_size)))
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, mu, logvar = model(batch)
            loss = vae_loss(reconstruction, batch, mu, logvar, task.beta)[0]
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        train_tensor = torch.from_numpy(train_x).to(device)
        test_tensor = torch.from_numpy(test_x).to(device)
        train_mu, _ = model.encode(train_tensor)
        reconstruction, test_mu, _ = model(test_tensor)
    return train_mu.cpu().numpy(), test_mu.cpu().numpy(), reconstruction.cpu().numpy(), epochs


def run(config_path="config/Verify_VAEFailureHighRe_1.1.yaml", reuse_runs=False):
    config = EasyConfig(config_path); source = EasyConfig(str(config.source_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(str(config.output_dir)); output.mkdir(parents=True, exist_ok=True)
    if reuse_runs:
        with (output / "latent_runs.csv").open(encoding="utf-8") as handle:
            run_rows = list(csv.DictReader(handle))
        with (output / "block_reconstruction.csv").open(encoding="utf-8") as handle:
            block_rows = list(csv.DictReader(handle))
        for row in run_rows:
            for key in ("latent_dim", "seed", "epochs"): row[key] = int(row[key])
            for key in ("f1", "total_mse"): row[key] = float(row[key])
        for row in block_rows:
            for key in ("latent_dim", "seed", "dimensions"): row[key] = int(row[key])
            for key in ("mse", "target_energy", "normalized_mse", "loss_fraction"):
                row[key] = float(row[key])
    else:
        run_rows, block_rows = [], []
    direct_rows = []
    for dataset in config.datasets:
        records = _load_slices(Path(source.output.cache_dir) / str(dataset))
        n_train_slices = int(source.task2.train_slice_count)
        train_lengths = [len(record["reference"]) for record in records[:n_train_slices]]
        reference_test = np.concatenate([r["reference"] for r in records[n_train_slices:]])
        values = np.concatenate([r["fmt"] for r in records])
        train_x, test_x = _prepare(values, train_lengths, float(source.task2.fmt_neighbor_weight))
        blocks = _block_indices(train_x.shape[1])
        for block, index in blocks.items():
            _, _, score = _fit_cluster(
                train_x[:, index], test_x[:, index], reference_test, source,
                already_scaled=True,
            )
            direct_rows.append({"dataset": str(dataset), "block": block,
                                "dimensions": len(index), "direct_f1": score["f1"]})
        if reuse_runs:
            continue
        for latent_dim in config.latent_dims:
            for seed in config.training_seeds:
                train_mu, test_mu, reconstruction, epochs = _train(
                    train_x, test_x, int(latent_dim), source, int(seed), device
                )
                _, _, score = _fit_cluster(train_mu, test_mu, reference_test, source)
                error2 = np.square(reconstruction - test_x)
                run_rows.append({"dataset": str(dataset), "latent_dim": int(latent_dim),
                                 "seed": int(seed), "f1": score["f1"], "epochs": epochs,
                                 "total_mse": float(error2.mean())})
                for block, index in blocks.items():
                    mse = float(error2[:, index].mean())
                    energy = float(np.square(test_x[:, index]).mean())
                    block_rows.append({
                        "dataset": str(dataset), "latent_dim": int(latent_dim), "seed": int(seed),
                        "block": block, "dimensions": len(index), "mse": mse,
                        "target_energy": energy, "normalized_mse": mse / max(energy, 1e-12),
                        "loss_fraction": float(error2[:, index].sum() / error2.sum()),
                    })
                print(f"[{dataset}] latent={latent_dim} seed={seed}: F1={score['f1']:.4f}, "
                      f"MSE={error2.mean():.5f}")

    for name, rows in (("latent_runs.csv", run_rows), ("block_reconstruction.csv", block_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "block_direct_f1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direct_rows[0]))
        writer.writeheader(); writer.writerows(direct_rows)
    summary = {}
    for dataset in config.datasets:
        summary[str(dataset)] = {}
        for latent_dim in config.latent_dims:
            selected = [r for r in run_rows if r["dataset"] == str(dataset)
                        and r["latent_dim"] == int(latent_dim)]
            summary[str(dataset)][str(latent_dim)] = {
                "mean_f1": float(np.mean([r["f1"] for r in selected])),
                "std_f1": float(np.std([r["f1"] for r in selected])),
                "mean_total_mse": float(np.mean([r["total_mse"] for r in selected])),
            }
    (output / "summary.json").write_text(json.dumps({"config": config.dict(), "results": summary}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dataset in config.datasets:
        means = [summary[str(dataset)][str(v)]["mean_f1"] for v in config.latent_dims]
        stds = [summary[str(dataset)][str(v)]["std_f1"] for v in config.latent_dims]
        axes[0].errorbar(config.latent_dims, means, yerr=stds, marker="o", capsize=3, label=str(dataset))
    axes[0].set(xlabel="Latent dimension", ylabel="Held-out-timeslice F1",
                title="Discriminative information retained by VAE")
    axes[0].set_xscale("log", base=2); axes[0].legend()
    semantic = ["real_norm", "imag_norm", "cosine", "chirality"]
    x = np.arange(len(semantic)); width = .25
    for offset, dataset in enumerate(config.datasets):
        selected = [r for r in block_rows if r["dataset"] == str(dataset)
                    and r["latent_dim"] == 16]
        values = [np.mean([r["normalized_mse"] for r in selected if r["block"] == block])
                  for block in semantic]
        axes[1].bar(x + (offset - 1) * width, values, width, label=str(dataset))
    axes[1].set_xticks(x, semantic); axes[1].set(ylabel="Normalized reconstruction MSE",
        title="Feature-block bias at baseline latent=16")
    axes[1].legend(); fig.tight_layout(); fig.savefig(output / "diagnosis.png", dpi=220); plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_VAEFailureHighRe_1.1.yaml")
    parser.add_argument("--reuse-runs", action="store_true")
    args = parser.parse_args(); run(args.config, args.reuse_runs)
