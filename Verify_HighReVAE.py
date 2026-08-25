"""Screen VAE capacity and label-free relational geometry preservation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from DeepUtils.utils import EasyConfig
from FMT_Utils.DFT_FMT_3D import fmt_feature_indices_3d
from FMT_Utils.RawPathline_3D import normalize_raw_train_eval, raw_pathline_representation
from FMT_Utils.VAE_3D import FeatureVAE3D, relational_distance_loss, vae_loss
from Run_Task2_Universality import _fit_cluster, _load_slices, _prepare
from Verify_HighReSampling3D import _load_common_records


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def _prepare_representation(records, representation, fit_slices, eval_slices, source,
                            raw_normalization="standard", fmt_feature_subset="all"):
    selected = list(range(fit_slices)) + list(eval_slices)
    train_lengths = [len(records[index]["reference"]) for index in range(fit_slices)]
    if representation == "fmt":
        values = np.concatenate([records[index]["fmt"] for index in selected])
        values = values[:, fmt_feature_indices_3d(fmt_feature_subset)]
        train_x, eval_x = _prepare(
            values, train_lengths,
            float(source.task2.fmt_neighbor_weight) if fmt_feature_subset == "all" else 1.0,
        )
    else:
        values = np.concatenate([
            raw_pathline_representation(records[index]["raw"], representation)
            for index in selected
        ])
        split = sum(train_lengths)
        sampled_steps = records[0]["raw"].shape[1] // (7 * 3)
        train_x, eval_x = normalize_raw_train_eval(
            values[:split], values[split:], representation, sampled_steps,
            raw_normalization,
        )
    reference = np.concatenate([records[index]["reference"] for index in eval_slices])
    return train_x, eval_x, reference


def _train(train_x, eval_x, settings, source, seed, device):
    _seed(seed)
    model = FeatureVAE3D(
        train_x.shape[1], settings["hidden_dims"], int(settings["latent_dim"])
    ).to(device)
    pca_components = pca_mean = None
    if settings.get("pca_init", False):
        if list(settings["hidden_dims"]):
            raise ValueError("pca_init currently requires a linear VAE (hidden_dims: [])")
        with threadpool_limits(limits=4):
            pca = PCA(n_components=int(settings["latent_dim"]), svd_solver="randomized",
                      random_state=int(seed)).fit(train_x)
        components = torch.from_numpy(pca.components_).to(
            device=device, dtype=model.mu.weight.dtype
        )
        pca_components = components.detach().clone()
        pca_mean = torch.from_numpy(pca.mean_).to(
            device=device, dtype=model.mu.weight.dtype
        )
        with torch.no_grad():
            model.mu.weight.copy_(components); model.mu.bias.zero_()
            model.logvar.weight.zero_()
            model.logvar.bias.fill_(float(settings.get("logvar_init", -6.0)))
            model.decoder[0].weight.copy_(components.T); model.decoder[0].bias.zero_()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings.get("learning_rate", source.task2.learning_rate)),
        weight_decay=float(settings.get("weight_decay", source.task2.weight_decay)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x)),
                        batch_size=int(settings.get("batch_size", source.task2.batch_size)),
                        shuffle=True,
                        generator=generator, drop_last=False)
    optimizer_steps = int(settings.get(
        "optimizer_steps", source.task2.target_optimizer_steps
    ))
    epochs = math.ceil(optimizer_steps / len(loader))
    started = time.perf_counter(); model.train()
    sums = np.zeros(5, dtype=np.float64); count = 0
    completed_steps = 0
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, mu, logvar = model(batch)
            base, rec, kl = vae_loss(
                reconstruction, batch, mu, logvar, float(settings["beta"])
            )
            relational = relational_distance_loss(
                batch, mu, int(settings["pair_count"])
            ) if float(settings["relational_weight"]) else mu.sum() * 0.0
            pca_anchor = F.mse_loss(
                mu, (batch - pca_mean) @ pca_components.T
            ) if float(settings.get("pca_anchor_weight", 0.0)) else mu.sum() * 0.0
            loss = (base + float(settings["relational_weight"]) * relational
                    + float(settings.get("pca_anchor_weight", 0.0)) * pca_anchor)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            n = len(batch); count += n
            sums += n * np.array([float(loss), float(rec), float(kl),
                                  float(relational), float(pca_anchor)])
            completed_steps += 1
            if completed_steps >= optimizer_steps:
                break
        if completed_steps >= optimizer_steps:
            break
    model.eval()
    with torch.no_grad():
        train_mu, _ = model.encode(torch.from_numpy(train_x).to(device))
        eval_tensor = torch.from_numpy(eval_x).to(device)
        reconstruction, eval_mu, eval_logvar = model(eval_tensor)
        _, rec, kl = vae_loss(
            reconstruction, eval_tensor, eval_mu, eval_logvar, float(settings["beta"])
        )
        relation = relational_distance_loss(
            eval_tensor, eval_mu, min(int(settings["pair_count"]), max(len(eval_x), 2) * 4)
        )
        eval_anchor = F.mse_loss(
            eval_mu, (eval_tensor - pca_mean) @ pca_components.T
        ) if pca_components is not None else eval_mu.sum() * 0.0
    return train_mu.cpu().numpy(), eval_mu.cpu().numpy(), {
        "epochs": epochs, "target_optimizer_steps": optimizer_steps,
        "completed_optimizer_steps": completed_steps,
        "pca_init": bool(settings.get("pca_init", False)),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.perf_counter() - started,
        "train_total": float(sums[0] / count), "train_reconstruction": float(sums[1] / count),
        "train_kl": float(sums[2] / count), "train_relational": float(sums[3] / count),
        "train_pca_anchor": float(sums[4] / count),
        "eval_reconstruction": float(rec), "eval_kl": float(kl),
        "eval_relational": float(relation), "eval_pca_anchor": float(eval_anchor),
    }


def run(config_path="config/Verify_HighReVAE_1.1.yaml", resume=False):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    sampling = yaml.safe_load(Path(spec["sampling_spec"]).read_text(encoding="utf-8"))
    source = EasyConfig(str(spec["source_config"]))
    variant_ids = [value["id"] for value in sampling["variants"]]
    cache_root = Path(sampling["output_dir"]) / "cache"
    if sampling.get("comparison_population", "common") == "native":
        records = {dataset: _load_slices(
            cache_root / spec["sampling_variant"] / dataset
        ) for dataset in spec["datasets"]}
    else:
        records = {dataset: _load_common_records(
            cache_root, variant_ids, spec["sampling_variant"], dataset
        ) for dataset in spec["datasets"]}
    output = Path(spec["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    result_path = output / "screening_runs.csv"
    rows = []
    if resume and result_path.exists():
        with result_path.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    completed = {(row["dataset"], row["representation"], row["vae_variant"],
                  int(row["training_seed"])) for row in rows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_slices = int(spec.get("fit_slices", 6))
    eval_slices = list(spec.get("eval_slices", [6, 7]))
    for dataset in spec["datasets"]:
        for representation in spec["representations"]:
            train_x, eval_x, reference = _prepare_representation(
                records[dataset], representation, fit_slices, eval_slices, source,
                spec.get("raw_normalization", "standard"),
                spec.get("fmt_feature_subset", "all"),
            )
            for settings in spec["variants"]:
                if representation not in settings.get("applies_to", spec["representations"]):
                    continue
                for seed_value in spec["screening_seeds"]:
                    key = (dataset, representation, settings["id"], int(seed_value))
                    if key in completed: continue
                    train_mu, eval_mu, losses = _train(
                        train_x, eval_x, settings, source, int(seed_value), device
                    )
                    _, _, score = _fit_cluster(train_mu, eval_mu, reference, source)
                    row = {"dataset": dataset, "representation": representation,
                           "vae_variant": settings["id"],
                           "raw_normalization": spec.get("raw_normalization", "standard"),
                           "fmt_feature_subset": spec.get("fmt_feature_subset", "all"),
                           "training_seed": int(seed_value), "f1": score["f1"],
                           "precision": score["precision"], "recall": score["recall"],
                           "hidden_dims": "x".join(str(v) for v in settings["hidden_dims"]),
                           "latent_dim": settings["latent_dim"], "beta": settings["beta"],
                           "relational_weight": settings["relational_weight"], **losses}
                    rows.append(row); _write(result_path, rows)
                    print(f"{dataset}/{representation}/{settings['id']}: F1={score['f1']:.4f}, "
                          f"relation={losses['eval_relational']:.4f}", flush=True)
    (output / "metadata.json").write_text(json.dumps({
        "experiment": spec["experiment"], "device": str(device), "config": spec,
        "note": (f"Uses first {fit_slices} slices for training and slices "
                 f"{eval_slices} for evaluation; IVD labels are evaluation-only.")
    }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_HighReVAE_1.1.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); run(args.config, args.resume)
