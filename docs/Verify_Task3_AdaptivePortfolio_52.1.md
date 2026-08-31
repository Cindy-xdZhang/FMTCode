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

## Deployed source-identity gate

The selector now has a separate precondition that reads only the five deployed
source config files. It checks each experiment identity and the SHA-256 of text
after normalizing CRLF/LF line endings; it does not open a selection, result,
checkpoint, or performance metric. The first two jobs (`51074003`, `51074118`)
failed closed because the initial whitelist contained remote raw-byte hashes
while the field and verifier required normalized-text hashes. No selector ran
and both reports stopped before reading performance artifacts.

Commit `48e7ddb` replaces all five values with normalized-text hashes and binds
7.2 to the resulting 52.1 config hash
`fc33380b1af18d2869d3eb18ff0bf8646797dc66f7d3951448a0628f69d091b0`.
Recovery job `51074226` completed with exit 0 and empty stderr. Its report
records `performance_artifacts_read=false`, all five source identities, and
SHA-256 `e84de56450899b32ad82afd9a478e179ed8990120c8dc1baafc2c36e2a8c9706`.
The completed gate was added to selector `51059320` without removing its
remaining `afterok:51058835` dependency.

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
waited strictly for selector `51059320`; the 7.2 entry job waited for both the
selector and this independent audit.

## Completed result

Selector `51059320` completed at `2026-08-31T09:18:29+03:00` without training
or opening confirmation data. Development Raw-PCA/FMT F1 is
`0.6847011551/0.8902179682`, paired gain `+0.2055168131`; Raw-PCA/FMT Average
Precision is `0.7285773718/0.9487193186`, gain `+0.2201419468`. All ten dataset
gains are positive. The `+0.20` gain target passed, but absolute FMT F1 remained
below `0.893`, so the joint target did not pass.

The selected sources are full-stack for Boeing and Smoke, safe-factor for
Channel, DeltaWing, F22, and halfcylinder, and head/alpha/clipping for
Tangaroa. The selector copied 40 models and 40 result files. Independent audit
job `51073078` verified all 80 file hashes, equal paired parameter counts, and
independently reconstructed every family choice and macro metric. Its maximum
absolute discrepancy was `2.220446049250313e-16`; portfolio-selection and
audit SHA-256 values are respectively
`257893ee754737c11d8b66a1d11b3e04edc084e8e340674540f55b6db3de4393`
and `289de6cd7a2ceda807b33638750a57e1ff06b98856c0b4edda2f54452ec27600`.
Only the subsequent 7.2 fresh spatial population is paper-level evidence.
