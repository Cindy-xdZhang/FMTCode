# Verify_Task3_TrainingHorizon_47.1

## Question

Does the common training horizon change the absolute supervised IVD-p95
classification quality of FMT and its paired advantage over the same-width,
same-structure train-only Raw-PCA residual arm?

This is a development-only hyperparameter search. It does not open a new
spatial population and cannot supply final confirmation evidence.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored FMT feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Epoch candidates: exact historical 100-epoch control and
  `{20,40,60,80,125,150,200,300}`.
- Within every candidate, FMT and train-only Raw-PCA use the same epoch count,
  initialization, batches, optimizer, learning rate, loss, split, head, and
  decision protocol.
- The control has no override and therefore executes the frozen historical
  100-epoch path exactly.
- No confirmation data are read.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact 100-epoch control. Eligible
candidates are ordered by paired F1 gain, then absolute FMT F1 and the frozen
robustness tie-breakers. Consequently, an undertrained candidate cannot win by
degrading Raw more strongly if it also lowers FMT on any family.

The joint development target remains dataset-macro F1 gain at least `.195`
and absolute FMT F1 at least `.893`. Failure of either requirement is retained
as a negative result. Any development winner still requires evaluation on an
unseen spatial population.

## Ibex deployment

Local Python compilation and 20 relevant unit tests passed. Static preflight
confirmed 10 datasets, 9 candidates, 90 array mappings, 540 paired trainings,
and closed confirmation. Full local preflight then checked every frozen split,
feature recipe, checkpoint, model-capacity guard, and loss contract; all 90
dataset/candidate mappings are eligible. Its manifest SHA-256 is
`ce557cd3...a2681`.

Ibex preflight job `51028066` completed successfully on `cn604-14` in 85 s.
The remote canonical preflight manifest SHA-256 is
`aefe405a42d731ed1283f1c71f1ddcf5a59301eb951efe5014c1d81b2ee53d56`.
It was downloaded to the local Ibex mirror and independently rehashed to the
same value; remote stderr is empty.
GPU array `51028067` and selector `51028071` are submitted with strict
dependencies. The array is limited to 24 concurrent jobs; each element
evaluates one dataset/epoch candidate and performs three paired seeds for both
arms. Checkpoints are not retained. Evidence archive job `51058293` was added
with strict `afterok:51028071`; its committed script SHA-256 is
`6f2e7392...80eed`, identical locally and remotely, and remote `bash -n`
passes. It requires all 540 per-run CSV files, refuses any unexpected model
checkpoint, and publishes hashes only after a 30 s archive-stability check.

## Development result

All 90 array mappings and all 540 paired trainings completed successfully.
Selector job `51028071` completed with exit code 0 on 2026-08-31. The selected
family horizons are 40 epochs for halfcylinder and SmokeBuoyancy, 80 epochs
for deltaWing, and the exact 100-epoch control for Channel, Tangaroa, F-22,
and Boeing 747.

The selected Raw-PCA/FMT dataset-macro F1 scores are `.69527/.88709`, giving
an FMT gain of `+.19182`. Dataset-macro Average Precision is
`.73770/.94557`, giving `+.20787`. All ten dataset entries retain positive F1
gain. The selection and leaderboard SHA-256 values are respectively
`79706980361cd89dd1fc9e718b24f221d5e0814bdcc783347e5ebd51d462cda9` and
`1d0061a762d73484cf05fbfc9b92b3e33d9f797449d0f1e6bec4dbc10e1d77e3`.
Confirmation remained closed.

The experiment does not reach the joint development target: F1 gain is below
`.195`, and absolute FMT F1 is below `.893`. It is also weaker than the
completed Dropout search (`+.19648/.88836`). Training horizon is therefore
retained only as an input to the preregistered full-stack combination search,
not as a standalone confirmation candidate.

## Archive recovery

The first archive job `51058293` failed after detecting 540 temporary model
checkpoints. This exposed an incorrect archive-script assumption: the generic
residual trainer necessarily writes one checkpoint for each paired training,
although this search does not need those models after selection. The recovery
script now requires exactly 540 checkpoints, verifies and stabilizes the
540-CSV archive, deletes only `.pt/.pth/.ckpt` files below this experiment's
candidate directory, and requires zero remaining checkpoints. Its local and
remote SHA-256 is
`05e8b88cacad6e27da8f4d4e6a8cf3424e2872c733d86f03318ca87a94c0b706`;
remote `bash -n` passes. Recovery job `51065692` is registered separately so
the failed first attempt remains visible.

Recovery job `51065692` completed with exit code 0. It archived all 540 CSVs,
removed exactly 540 temporary checkpoints, and verified that zero model files
remain. The stable CSV archive SHA-256 is
`aca0ab63921e447d396ac937a0d425c0e660a55b5607805cad2cb96ad4a5c250`.
The result artifacts were downloaded without checkpoints and independently
audited from all 540 CSVs. The audit passed with all source hashes consistent,
equal paired parameter counts, and maximum metric difference
`2.220446049250313e-16` versus the selector. Its SHA-256 is
`01e49d6a2c88abb0880f7383af4b4187d23011b7caef1bb880717faf6530fede`.
The exact 100-epoch control had Raw-PCA/FMT F1 `.69785/.88708`, gain
`+.18923`; family-specific horizon selection therefore increased gain by
only `+.00259` while changing absolute FMT F1 by about `+.00001`.
