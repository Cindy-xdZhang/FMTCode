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

## Independent evidence gate

`Audit_Task3_AdaptivePortfolio.py` does not import either portfolio selector.
It independently reloads all five registered source selections, reconstructs
the physical-family winners using the frozen metric order, recomputes all six
dataset-macro quantities, verifies source/config/preflight hashes, and checks
the content hash and paired parameter count of every frozen result and model.
The synthetic five-source contract also proves that checkpoint tampering is
rejected. The independent audit is a required gate before 7.2 may open its
fresh spatial confirmation population.

The auditor and evidence script are frozen in commit `287acba`. Their local
and remote SHA-256 values are respectively
`1fd484d5ff9bd6117760a76a349761a5544e1b5335115e28164154d93d750c22`
and `6b569f489f13afa74bf7c0780b1599c1d4bfc58e35a6dab8039e45a2bec82721`;
remote Python compilation and `bash -n` pass. Ibex evidence job `51073078`
waits strictly for selector `51059320`. The 7.2 entry job now waits for both
the selector and this independent audit.
