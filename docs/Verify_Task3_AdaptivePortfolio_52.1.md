# Verify_Task3_AdaptivePortfolio_52.1

## Purpose

This training-free stage prevents the adaptive low-gamma and high-dropout
follow-ups from being omitted by the earlier 49.1 portfolio. It selects the
strongest guarded development recipe per physical family across 44.1, 45.1,
48.1, 50.1, and 51.1.

The selection rule remains paired dataset-macro F1 gain first, then absolute
FMT F1, Average Precision gain, absolute FMT Average Precision, positive
dataset count, worst-dataset gain, and worst-seed gain. Every source winner
must already pass its source search's zero-tolerance FMT F1 and Average
Precision guard. Confirmation remains closed.

## Artifact retention

The selector freezes seed 40 and 41 for both FMT and train-only Raw-PCA:
10 datasets x 2 seeds x 2 arms = 40 models. It copies each selected checkpoint
and per-run CSV into its own `frozen_artifacts` directory and verifies all
SHA-256 values before publishing the portfolio selection. The source searches
may therefore clean their temporary checkpoints only after 52.1 succeeds.

No model is trained, no confirmation primitive is generated, and no
confirmation metric is read by this stage. The earlier 49.1 and 7.1 records
remain unchanged; the 7.1 entry is held until an adaptive final-confirmation
version is frozen.

## Pre-deployment verification

- The five-source static preflight reports 10 datasets, source seeds 40--42,
  frozen seeds 40--41, zero training runs, and closed confirmation data.
- Five dedicated `unittest` contracts pass, including an in-memory copy and
  SHA-256 verification of all 40 model records and 80 frozen files.
- Both the selector and its test module pass Python byte-code compilation.
- The config raw SHA-256 at implementation time is
  `fc33380b1af18d2869d3eb18ff0bf8646797dc66f7d3951448a0628f69d091b0`.
