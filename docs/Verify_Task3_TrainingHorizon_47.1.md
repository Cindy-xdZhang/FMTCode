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
GPU array `51028067` and selector `51028071` are submitted with strict
dependencies. The array is limited to 24 concurrent jobs; each element
evaluates one dataset/epoch candidate and performs three paired seeds for both
arms. Checkpoints are not retained.
