# Verify_Task3_GradientClipping_33.1

## Question

Can global gradient-norm clipping improve absolute FMT classification while
increasing its paired advantage over the same-width train-only Raw-PCA arm?

Gradient clipping rescales all trainable residual-head gradients only when
their joint L2 norm exceeds a fixed threshold. It is applied after the
finite-gradient check and immediately before the optimizer step. It does not
change labels, features, model capacity, loss terms, validation thresholds, or
fusion-alpha selection.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Seeds: 40, 41, 42.
- Candidates: no clipping and global norm limits
  `{0.1,0.25,0.5,1,2,5,10}`.
- Within every candidate, FMT and train-only Raw-PCA use the same threshold,
  initialization, batches, optimizer, budget, split, and head.
- The no-clipping control is a strict code-level no-op.
- Confirmation remains closed.

Eight candidates across ten datasets produce 80 array mappings and 480 paired
trainings. The maximum pre-clipping gradient norm is recorded per epoch for
auditing whether each threshold actually activates.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to no clipping. Eligible candidates are
ordered by paired F1 gain, then absolute FMT F1 and the registered robustness
tie-breakers.

The joint development target is F1 gain at least `0.195` and absolute FMT F1
at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

This experiment was specified without reading partial or final performance
from Task3 searches 27.1--32.1.

## Ibex deployment

Implementation commit `ba1816f` was pushed before deployment. The immutable
archive SHA-256 is `0a3d9b7e...b804a`; the local raw and remote canonical
config SHA-256 values are `66a62c1a...5cde` and `8c5b9b9d...60e6`
respectively, reflecting Windows CRLF versus Git-archive LF line endings.

Local and remote runs passed the same 15 relevant unit tests. Remote Python
compilation, three `bash -n` checks, and static preflight also passed. Full
remote preflight job `51004078` ran on `cn604-14` from 03:15:43 to 03:16:40,
exited zero with empty stderr, and produced manifest SHA-256
`402e7855...276d`. It confirmed ten datasets, eight candidates, 480 paired
trainings, zero ineligible recipes, and closed confirmation state.

GPU array `51004081[0-79%24]` and selector `51004084` were submitted through
strict `afterok` dependencies. The preflight dependency is released; the GPU
array waits for scheduler priority. No partial metric may be read before the
selector runs.
