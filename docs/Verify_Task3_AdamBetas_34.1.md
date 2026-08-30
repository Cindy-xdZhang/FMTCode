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
