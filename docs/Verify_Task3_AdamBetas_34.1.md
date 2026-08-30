# Verify_Task3_AdamBetas_34.1

## Question

Can AdamW momentum timescales improve absolute FMT classification while
increasing its paired advantage over the same-width train-only Raw-PCA arm?

AdamW `beta1` controls the exponential moving average of gradients; `beta2`
controls the moving average of squared gradients. These values can change how
quickly the residual branch follows feature-specific signals without changing
the network, loss, data, labels, or parameter count.

## Why this is distinct

Task3 search 27.1 covers learning rate and weight decay, and 28.1 compares
optimizer families. No prior registered Task3 experiment systematically varies
Adam-family momentum coefficients. This experiment was declared without
reading partial or final performance from searches 27.1--33.1 and does not
depend on their selectors.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Optimizer: AdamW with the frozen base learning rate and weight decay.
- Seeds: 40, 41, 42.
- FMT and train-only Raw-PCA use the same beta pair, initialization, batches,
  optimizer budget, split, and network in every candidate.
- Confirmation remains closed.

The complete grid is `beta1 ∈ {0.5,0.9,0.95}` by
`beta2 ∈ {0.9,0.99,0.999}`. The historical PyTorch default
`(0.9,0.999)` is represented by a strict no-override control. Nine candidates
across ten datasets produce 90 array mappings and 540 paired trainings.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to the default-beta control. Eligible
candidates are ordered by paired F1 gain, then absolute FMT F1 and the
registered robustness tie-breakers.

The joint development target is F1 gain at least `0.195` and absolute FMT F1
at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Implementation commit `4dc40ad` was pushed before deployment. The immutable
archive SHA-256 is `c5d2524a...a4b2`; local raw and remote canonical config
SHA-256 values are `71e164db...be44` and `3db86f16...66ee`, respectively.

Local and remote runs passed the same 22 relevant unit tests. Remote Python
compilation, three `bash -n` checks, and static preflight also passed. Full
preflight job `51004187`, GPU array `51004190[0-89%24]`, and selector
`51004191` were submitted with strict `afterok` dependencies. No partial metric
may be read before selection.

Full preflight ran on `cn604-14` from 03:27:50 to 03:28:50, exited zero
with empty stderr, and produced manifest SHA-256 `e2db5ad3...6184d`. It
confirmed ten datasets, nine candidates, 540 paired trainings, zero ineligible
recipes, and closed confirmation state. The GPU dependency is released and the
array waits for scheduler priority.
