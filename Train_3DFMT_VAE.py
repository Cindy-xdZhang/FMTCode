"""Task2: compare raw-primitive VAE against FMT3D-feature VAE without labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from DeepUtils.utils import EasyConfig
from FMT_Utils.VAE_3D import FeatureVAE3D, vae_loss
from Verify_3DFMTHyperparam import _frozen_reference


def _raw_local_features(pathlines):
    xyz = np.asarray(pathlines[..., :3], dtype=np.float32)
    return (xyz - xyz[:, :1, :1, :]).reshape(len(xyz), -1)


def _cluster_score(train_features, test_features, reference_test, config):
    latent_scaler = StandardScaler().fit(train_features)
    train_scaled = latent_scaler.transform(train_features)
    test_scaled = latent_scaler.transform(test_features)
    kmeans = KMeans(n_clusters=2, random_state=int(config.evaluation.kmeans_seed),
                    n_init=int(config.evaluation.kmeans_n_init))
    kmeans.fit(train_scaled)
    labels = kmeans.predict(test_scaled)
    candidates = []
    for cluster in (0, 1):
        prediction = labels == cluster
        candidates.append({
            "cluster_as_vortex": cluster,
            "f1": float(f1_score(reference_test, prediction, zero_division=0)),
            "iou": float(jaccard_score(reference_test, prediction, zero_division=0)),
            "precision": float(precision_score(reference_test, prediction, zero_division=0)),
            "recall": float(recall_score(reference_test, prediction, zero_division=0)),
            "predicted_fraction": float(prediction.mean()),
        })
    return max(candidates, key=lambda row: row["f1"]), labels


def _train_vae(train_x, test_x, config, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = FeatureVAE3D(train_x.shape[1], config.vae.hidden_dims,
                         int(config.vae.latent_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.vae.learning_rate),
                                  weight_decay=float(config.vae.weight_decay))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x.astype(np.float32))),
                        batch_size=int(config.vae.batch_size), shuffle=True,
                        generator=generator, drop_last=False)
    history = []
    model.train()
    for epoch in range(int(config.vae.epochs)):
        sums = np.zeros(3, dtype=np.float64); count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, mu, logvar = model(batch)
            loss, rec, kl = vae_loss(reconstruction, batch, mu, logvar, config.vae.beta)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            n = len(batch); sums += n * np.array([loss.item(), rec.item(), kl.item()]); count += n
        history.append((sums / count).tolist())
    model.eval()
    with torch.no_grad():
        train_mu, _ = model.encode(torch.from_numpy(train_x.astype(np.float32)).to(device))
        test_tensor = torch.from_numpy(test_x.astype(np.float32)).to(device)
        test_reconstruction, test_mu, test_logvar = model(test_tensor)
        test_loss = vae_loss(test_reconstruction, test_tensor, test_mu, test_logvar,
                             config.vae.beta)
    return (train_mu.cpu().numpy(), test_mu.cpu().numpy(), history,
            {"loss": float(test_loss[0]), "reconstruction": float(test_loss[1]),
             "kl": float(test_loss[2]),
             "parameter_count": sum(p.numel() for p in model.parameters())})


def run(config):
    run_dir = Path(config.input.run_dir)
    with np.load(run_dir / "clustering_result.npz") as data:
        pathlines = np.asarray(data["pathlines"], dtype=np.float32)
        fmt_features = np.asarray(data["features"], dtype=np.float32)
        seeds = np.asarray(data["seeds"], dtype=np.float64)
    raw_features = _raw_local_features(pathlines)
    reference = _frozen_reference(config, seeds)
    rng = np.random.default_rng(int(config.seed))
    order = rng.permutation(len(seeds))
    train_count = int(round(float(config.split.train_fraction) * len(order)))
    train_index, test_index = order[:train_count], order[train_count:]
    reference_test = reference[test_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    representations = {"raw": raw_features, "fmt": fmt_features}
    rows = []; histories = {}; best_labels = {}
    for name, values in representations.items():
        scaler = StandardScaler().fit(values[train_index])
        train_x = scaler.transform(values[train_index]).astype(np.float32)
        test_x = scaler.transform(values[test_index]).astype(np.float32)
        direct, direct_labels = _cluster_score(train_x, test_x, reference_test, config)
        rows.append({"representation": name, "training_seed": "none",
                     "variant": f"{name}_direct", **direct})
        best_labels[f"{name}_direct"] = (direct["f1"], direct_labels.copy(), "none")
        for seed in config.vae.training_seeds:
            train_mu, test_mu, history, losses = _train_vae(
                train_x, test_x, config, int(seed), device
            )
            score, latent_labels = _cluster_score(train_mu, test_mu, reference_test, config)
            rows.append({"representation": name, "training_seed": int(seed),
                         "variant": f"{name}_vae", **score, **losses})
            histories[f"{name}_{seed}"] = history
            key = f"{name}_vae"
            if key not in best_labels or score["f1"] > best_labels[key][0]:
                best_labels[key] = (score["f1"], latent_labels.copy(), int(seed))
            print(f"{name} VAE seed={seed}: F1={score['f1']:.4f}, "
                  f"test_rec={losses['reconstruction']:.5f}")

    summary = {}
    for variant in ("raw_direct", "raw_vae", "fmt_direct", "fmt_vae"):
        selected = [row for row in rows if row["variant"] == variant]
        f1s = [row["f1"] for row in selected]
        summary[variant] = {
            "runs": len(selected), "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)), "min_f1": float(np.min(f1s)),
            "max_f1": float(np.max(f1s)),
        }
    summary["fmt_vae_minus_raw_vae_mean_f1"] = (
        summary["fmt_vae"]["mean_f1"] - summary["raw_vae"]["mean_f1"]
    )
    summary["fmt_vae_minus_fmt_direct_mean_f1"] = (
        summary["fmt_vae"]["mean_f1"] - summary["fmt_direct"]["mean_f1"]
    )

    output_dir = Path(config.output.dir); output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    result = {
        "experiment": str(config.experiment), "device": str(device),
        "train_count": len(train_index), "test_count": len(test_index),
        "test_positive_count": int(reference_test.sum()), "summary": summary,
        "config": config.dict(),
        "warning": "Single field/time result; held-out samples are random spatial seeds.",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "split.npz", train_index=train_index,
                        test_index=test_index, reference=reference)
    np.savez_compressed(output_dir / "training_histories.npz",
                        **{key: np.asarray(value) for key, value in histories.items()})
    np.savez_compressed(
        output_dir / "best_test_cluster_labels.npz",
        test_index=test_index, reference_test=reference_test,
        **{key: value[1] for key, value in best_labels.items()},
    )

    labels = ["Raw", "Raw+VAE", "FMT", "FMT+VAE"]
    keys = ["raw_direct", "raw_vae", "fmt_direct", "fmt_vae"]
    means = [summary[key]["mean_f1"] for key in keys]
    errors = [summary[key]["std_f1"] for key in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=errors, capsize=4,
                  color=["#999999", "#577590", "#f8961e", "#43aa8b"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylabel="Held-out clustering F1", ylim=(0, 1),
           title="Task2: raw VAE versus FMT3D VAE")
    fig.tight_layout(); fig.savefig(output_dir / "task2_f1_comparison.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, color in (("raw", "#577590"), ("fmt", "#43aa8b")):
        curves = np.asarray([histories[f"{name}_{seed}"] for seed in config.vae.training_seeds])
        axes[0].plot(curves[:, :, 1].mean(axis=0), label=name, color=color)
        axes[1].plot(curves[:, :, 2].mean(axis=0), label=name, color=color)
    axes[0].set(title="Reconstruction loss", xlabel="epoch", ylabel="MSE")
    axes[1].set(title="KL divergence", xlabel="epoch", ylabel="KL")
    for ax in axes: ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output_dir / "task2_training_curves.png", dpi=220); plt.close(fig)

    test_seeds = seeds[test_index]
    fig = plt.figure(figsize=(18, 5))
    panels = (("raw_direct", "Raw direct"), ("raw_vae", "Raw+VAE best seed"),
              ("fmt_direct", "FMT direct"), ("fmt_vae", "FMT+VAE best seed"))
    span = np.maximum(test_seeds.max(axis=0) - test_seeds.min(axis=0), 1e-12)
    for panel, (key, title) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 4, panel, projection="3d")
        _, cluster_labels, selected_seed = best_labels[key]
        vortex_cluster = max(
            (0, 1), key=lambda cluster: f1_score(reference_test, cluster_labels == cluster,
                                                  zero_division=0)
        )
        mask = cluster_labels == vortex_cluster
        ax.scatter(test_seeds[~mask, 0], test_seeds[~mask, 1], test_seeds[~mask, 2],
                   s=3, color="#d9d9d9", alpha=0.25)
        ax.scatter(test_seeds[mask, 0], test_seeds[mask, 1], test_seeds[mask, 2],
                   s=7, color="#00b4d8", alpha=0.9)
        suffix = "" if selected_seed == "none" else f"\nseed={selected_seed}"
        ax.set(title=title + suffix, xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect(span)
    fig.tight_layout(); fig.savefig(output_dir / "task2_latent_clusters_3d.png", dpi=220); plt.close(fig)
    print(json.dumps(summary, indent=2)); print(f"output: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mainExp_3DFMTVAE_1.1.yaml")
    args = parser.parse_args()
    run(EasyConfig(args.config))
