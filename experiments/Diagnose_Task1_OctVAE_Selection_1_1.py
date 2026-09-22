"""Why did Davies-Bouldin pick the checkpoint it picked?

Seed 1 of `exp_Task1_OctVAE_3D_1.1` selected essentially the first eligible
checkpoint on three of four datasets, and the `plain`/`gmean` readouts collapsed
to F1 ~ .12 on two of them.  Two explanations fit that pattern and they imply
opposite next steps:

  (a) clustering quality really does peak at initialisation and decay -- the
      method does not work here, and more seeds are wasted compute;
  (b) clustering quality peaks later but Davies-Bouldin prefers the early,
      barely-trained encoder because its latent is two compact spherical blobs
      -- the selection criterion is the bug, not the representation.

This script separates them.  It trains one seed and, at every evaluation,
records the label-free selection score alongside an **oracle** F1 curve.

The oracle is scored on the *calibration* ordinals only (8-9).  Those labels are
already used legitimately to fix the anonymous cluster identity, so no
confirmation slice is touched and the frozen test set stays frozen.  The oracle
curve is a diagnostic upper bound and must never be used to select anything --
exactly the `--checkpoint_select f1` / `best_f1_anywhere` distinction the 2D
method keeps.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.OctahedralGroup_3D import (  # noqa: E402
    apply_norm_stats_oh, build_signal_from_raw, group_tensors,
)
from FMT_Utils.SirenVAE_3D import encode_readouts, fit_octvae, l2_normalize  # noqa: E402
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference  # noqa: E402
from sklearn.cluster import KMeans as _KMeans  # noqa: E402

from FMT_Utils.Task12Evaluation_3D import (  # noqa: E402
    binary_cluster_metrics, calibrate_vortex_cluster,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_READOUTS = ("plain", "gmean", "gmax")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=7068)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=500)
    parser.add_argument("--probe-samples", type=int, default=4096)
    parser.add_argument("--output", default="outputs/exp_Task1_OctVAE_3D_1.1_diagnostic")
    arguments = parser.parse_args()

    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    dataset = [d for d in config["datasets"] if d["id"] == arguments.dataset][0]
    splits = config["splits"]
    settings = dict(config["training"])
    settings.update(config["model"])
    if arguments.steps:
        settings["steps"] = int(arguments.steps)
    steps_per_probe = int(arguments.probe_every)
    if steps_per_probe % int(settings["eval_every"]):
        raise SystemExit("--probe-every must be a multiple of training eval_every")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development = load_cache_records(ROOT / dataset["development_cache"])
    sampled = int(config["sampled_steps"])

    def signals(ordinals):
        return np.concatenate([build_signal_from_raw(development[i]["raw"], sampled)
                               for i in ordinals])

    train_signal = signals(splits["vae_train"])
    validation_signal = signals(splits["vae_validation"])
    fit_signal = signals(splits["kmeans_train"])
    probe_signal = signals(splits["cluster_calibration"])
    probe_reference = stack_reference([development[i] for i in splits["cluster_calibration"]])

    limit = int(arguments.probe_samples)
    rng = np.random.default_rng(int(arguments.seed))
    if len(probe_signal) > limit:
        pick = rng.choice(len(probe_signal), limit, replace=False)
        probe_signal, probe_reference = probe_signal[pick], probe_reference[pick]
    fit_pick = rng.choice(len(fit_signal), min(limit, len(fit_signal)), replace=False)
    fit_signal = fit_signal[fit_pick]

    matrices, permutation = group_tensors(config["group"], device=device)
    fit_raw = torch.as_tensor(fit_signal, dtype=torch.float32, device=device)
    probe_raw = torch.as_tensor(probe_signal, dtype=torch.float32, device=device)

    records = []

    def probe(step, total, history, model, stats):
        if step % steps_per_probe and step != total:
            return
        fit_n = apply_norm_stats_oh(fit_raw, stats, settings["clip_sigma"])
        probe_n = apply_norm_stats_oh(probe_raw, stats, settings["clip_sigma"])
        with torch.no_grad():
            fit_z = encode_readouts(model, fit_n, matrices, permutation,
                                    batch_size=4096, readouts=PROBE_READOUTS)
            probe_z = encode_readouts(model, probe_n, matrices, permutation,
                                      batch_size=4096, readouts=PROBE_READOUTS)
        row = {"step": step,
               "selection_score": history["selection"][-1],
               "zinv_std": history["zinv_std"][-1],
               "recon": history["recon"][-1],
               "contrast": history["contrast"][-1]}
        for readout in PROBE_READOUTS:
            train_features = l2_normalize(fit_z[readout])
            probe_features = l2_normalize(probe_z[readout])
            # spread of the code itself, before any clustering
            row[f"{readout}_std"] = float(np.std(probe_z[readout], axis=0).mean())
            row[f"{readout}_rank"] = int(np.linalg.matrix_rank(
                probe_z[readout] - probe_z[readout].mean(0), tol=1e-4))
            # L2-normalised, NOT standardised -- see exp_Task1_OctVAE_3D.md §9.
            fitted = _KMeans(n_clusters=2, random_state=int(arguments.seed),
                             n_init=config["kmeans_n_init"]).fit(train_features)
            labels = fitted.predict(probe_features)
            vortex = calibrate_vortex_cluster(probe_reference, labels)
            metrics = binary_cluster_metrics(probe_reference, labels, vortex)
            row[f"{readout}_oracle_f1"] = metrics["f1"]
            row[f"{readout}_oracle_ari"] = metrics["ari"]
            row[f"{readout}_predicted_positive"] = metrics["predicted_positive_fraction"]
            sizes = np.bincount(labels, minlength=2)
            row[f"{readout}_min_cluster_frac"] = float(sizes.min() / sizes.sum())
        records.append(row)
        print(f"  step {step:6d}  db={row['selection_score']}  "
              f"zstd={row['zinv_std']:.4f}  "
              + "  ".join(f"{r}: F1={row[f'{r}_oracle_f1']:.4f} "
                          f"minclu={row[f'{r}_min_cluster_frac']:.4f}"
                          for r in PROBE_READOUTS), flush=True)

    print(f"[{arguments.dataset}] seed {arguments.seed}, {settings['steps']} steps, "
          f"probe every {steps_per_probe}", flush=True)
    _, _, history = fit_octvae(train_signal, validation_signal, settings,
                               arguments.seed, device, group=config["group"],
                               progress=probe)

    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{arguments.dataset}_seed{arguments.seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(records[0]))
        writer.writeheader()
        writer.writerows(records)

    scored = [r for r in records if r["selection_score"] is not None]
    chosen = max(scored, key=lambda r: r["selection_score"]) if scored else None
    print(f"\nselection chose step {history['selected_step']} "
          f"(probe grid best {chosen['step'] if chosen else '—'})")
    for readout in PROBE_READOUTS:
        best = max(records, key=lambda r: r[f"{readout}_oracle_f1"])
        at_selected = min(records, key=lambda r: abs(r["step"] - history["selected_step"]))
        print(f"  {readout:6s}: oracle peak F1 {best[f'{readout}_oracle_f1']:.4f} "
              f"at step {best['step']:6d}   |   F1 at selected step "
              f"{at_selected[f'{readout}_oracle_f1']:.4f}   "
              f"regret {best[f'{readout}_oracle_f1'] - at_selected[f'{readout}_oracle_f1']:+.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
