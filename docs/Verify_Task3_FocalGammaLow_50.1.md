# Verify_Task3_FocalGammaLow_50.1

## Question

Does resolving focal gamma below `0.10` increase the paired F1 advantage of
FMT without lowering absolute FMT F1 or Average Precision?

## Motivation and evidence boundary

Completed development search 39.1 selected gamma `0.10` for Channel and the
half-cylinder family, exactly at that grid's lower focal boundary. All other
families retained weighted binary cross entropy, and larger gamma values
generally failed the zero-regression absolute-FMT guard. The frozen parent
selection SHA-256 is `1e99c432...9e994`.

This is an adaptive development follow-up. It uses no fresh confirmation
population and cannot establish a paper-level claim by itself.

## Frozen comparison

- The development populations, IVD-p95 labels, split, Raw checkpoints,
  family-specific 22.1 FMT features, network, optimizer, batch size, epoch
  budget, early stopping, and seeds remain identical to 39.1.
- Both FMT and train-only Raw-PCA receive the same loss and gamma in every
  candidate.
- The control is the exact no-override weighted-binary-cross-entropy path.
- Paired seeds are `40`, `41`, and `42`; confirmation remains closed.

## Registered grid and selection

The focal grid is `{0.01,0.025,0.05,0.075,0.10,0.15,0.20}` plus the exact
control. Eight candidates across ten datasets yield 80 array mappings and
480 paired trainings.

Selection remains physical-family specific. A candidate must first keep both
FMT F1 and FMT Average Precision at least equal to the same-family exact
control. Eligible candidates are ranked by paired dataset-macro F1 gain,
absolute FMT F1, and the same registered robustness tie-breakers as 39.1.

The joint target remains F1 gain at least `+0.195` and absolute FMT F1 at
least `0.893`. Failure of either target is a negative result. Any development
winner must enter a separately frozen fresh spatial confirmation before use
as paper-level evidence.

## Main files

- `config/Verify_Task3_FocalGammaLow_50.1.yaml`
- `tests/test_task3_focal_gamma_low_50_1.py`
- `Search_Task3_LossOptimization_7_1.py`
- `ibex_bash/verify_task3_focal_gamma_low_50.1_*.sh`

## Status

Completed. Full preflight `51056257`, all 80 GPU children in `51056260`,
selector `51056263`, and no-delete evidence job `51072242` exited successfully
with empty stderr. The array contains all 480 paired trainings. The evidence
archive contains all 480 per-run CSV files, no model files, and its SHA-256 is
`ff85edf2...f973e`; all 480 temporary checkpoints remain available for the
training-free 52.1 portfolio.

The selected development result is:

- Raw-PCA/FMT F1: `0.69743 / 0.88757`; gain `+0.19014`.
- Raw-PCA/FMT Average Precision: `0.74048 / 0.94652`; gain `+0.20604`.
- Positive F1 gain on 10/10 datasets; worst dataset gain `+0.05272`.
- Channel and half-cylinder select gamma `0.075`; F22 selects `0.01`;
  Tangaroa, Delta Wing, Boeing 747, and Smoke Buoyancy retain the exact
  weighted-binary-cross-entropy control.

Relative to the exact control, selection raises absolute FMT F1 by only
`+0.00045` and paired F1 gain by `+0.00036`. It misses the registered gain
target by `0.00486` and the absolute FMT F1 target by `0.00543`. Therefore the
search resolves the lower boundary for three families but does not support a
new overall winner and does not open confirmation.

An implementation-independent audit reconstructed all candidate/seed/arm
records, absolute-FMT guards, family choices, and macro metrics. Its maximum
difference from the selector is `4.44e-16`; all paired parameter counts and
source hashes pass. Selection, leaderboard, evidence archive, and audit
SHA-256 values are respectively `22b4b46f...f4ab8`, `0f8500b0...99930`,
`ff85edf2...f973e`, and `dbe33bc7...1bcfe`.

Cleanup job `51056264` remains dependency-gated on the selector, 52.1 model
copy, and evidence completion. It cannot delete source checkpoints before
52.1 has frozen any selected family models.
