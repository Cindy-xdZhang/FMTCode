# Verify_Task3_ParameterEMA_40.1

## Question

Can an exponential moving average (EMA) of trainable residual-head parameters
improve absolute FMT classification while increasing its paired advantage over
the same-width train-only Raw-PCA arm?

EMA keeps a non-trainable running average of each trainable residual parameter
after every optimizer step. Validation, early stopping, checkpoint selection,
and inference use the averaged parameters; optimizer updates continue from the
unsmoothed live parameters. The frozen Raw backbone is never averaged.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: exact no-EMA control and decay
  `{0.5,0.8,0.9,0.95,0.98,0.99,0.995,0.999}`.
- Within every candidate, FMT and train-only Raw-PCA use the same decay,
  initialization, batches, optimizer, epoch budget, split, and head.
- The no-EMA control does not construct, update, swap, or checkpoint averaged
  parameters and is therefore a strict code-level no-op.
- Confirmation remains closed.

Nine candidates across ten datasets produce 90 array mappings and 540 paired
trainings. The decay is recorded in histories, checkpoints, and per-run output.

## Why this search follows 33.1

The completed 33.1 logs contain 480 trainings, of which only 21 (4.4%) reached
epoch 100. The earlier 7.1 160-epoch candidate was not selected for any
physical family. A larger epoch budget therefore has limited support. EMA
instead tests whether noisy late residual-head updates obscure a better FMT
solution without changing features, labels, capacity, or the paired baseline.

This search was declared after the complete 33.1 selector result and without
reading partial metrics from 34.1--39.1.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the exact no-EMA control. Eligible
candidates are ordered by paired F1 gain, then absolute FMT F1 and the
registered robustness tie-breakers.

The joint development target remains F1 gain at least `0.195` and absolute FMT
F1 at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Local Python compilation and 48 relevant unit tests passed. Full local
preflight confirmed 10 datasets, 9 candidates, 90 array mappings, 540 paired
trainings, all parameter budgets, and closed confirmation state. The local
preflight manifest SHA-256 is `edbbe882...a8f9`.

Remote validation and deployment are pending.
