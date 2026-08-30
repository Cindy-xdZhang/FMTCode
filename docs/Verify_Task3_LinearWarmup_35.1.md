# Verify_Task3_LinearWarmup_35.1

## Question

Can a linear learning-rate warmup improve absolute FMT classification while
increasing its paired advantage over the same-width train-only Raw-PCA arm?

The frozen Raw classifier already supplies useful logits, whereas the residual
head starts from random weights. A short warmup may prevent large early updates
from disrupting that starting point while the FMT branch begins to learn a
correction.

## Why this is distinct

Task3 search 7.1 included a cosine schedule and one longer training-budget
candidate, but did not test warmup. Search 27.1 covers the constant base
learning rate. This experiment holds that base learning rate fixed and changes
only how many initial epochs linearly ramp from 10% to 100% of it. It was
declared without reading partial or final performance from searches 27.1--34.1.

## Frozen comparison

- Development population and split: completed 5.2 protocol.
- Feature: completed 22.1 family-specific anchored feature.
- Head: two hidden layers, width 64, LayerNorm, GELU, zero dropout.
- Optimizer, base learning rate, loss, and epoch budget are unchanged.
- Seeds: 40, 41, 42.
- FMT and train-only Raw-PCA use the same schedule, initialization, batches,
  split, and network in every candidate.
- Confirmation remains closed.

Candidates are no warmup and warmup lengths `{1,2,5,10,20,40}` epochs. The
no-warmup control does not construct a scheduler and exactly reproduces the
historical constant-learning-rate path. Seven candidates across ten datasets
produce 70 array mappings and 420 paired trainings.

## Selection and stopping rule

Candidates must first satisfy zero-tolerance per-family absolute FMT F1 and
Average Precision guards relative to no warmup. Eligible candidates are
ordered by paired F1 gain, then absolute FMT F1 and the registered robustness
tie-breakers.

The joint development target is F1 gain at least `0.195` and absolute FMT F1
at least `0.893`. Failure of either requirement is retained as a negative
result. Any development winner still requires evaluation on a fresh spatial
population.

## Ibex deployment

Implementation commit `10436a4` was pushed before deployment. The immutable
archive SHA-256 is `c55531af...b75c`; local raw and remote canonical config
SHA-256 values are `69eb9cdb...c413` and `c7d6c4c3...78ec`, respectively.

Local and remote runs passed the same 25 relevant unit tests. Remote Python
compilation, three `bash -n` checks, and static preflight also passed. Full
preflight job `51004391`, GPU array `51004395[0-69%24]`, and selector
`51004397` were submitted with strict `afterok` dependencies. No partial metric
may be read before selection.

Full preflight ran on `cn604-12` from 03:38:56 to 03:40:05, exited zero with
empty stderr, and produced manifest SHA-256 `01672ff5...756db`. It confirmed
ten datasets, seven candidates, 420 paired trainings, zero ineligible recipes,
and closed confirmation state. This released the GPU dependency; the array was
then scheduled without changing the preregistered candidate set.

## Completed result

GPU array `51004395[0-69%24]` completed all 70 children by 11:07:15 on
9 Tesla P100 and 61 GTX 1080 Ti allocations. All 70 stdout files exist, all 70
stderr files are empty, and every child exited zero. Selector `51004397` then
ran on `cn604-14` from 11:07:20 to 11:07:31 and exited zero with empty stderr.

The zero-warmup control was selected for every physical family. Selected
dataset-macro results are:

| Arm | F1 | Average Precision |
|---|---:|---:|
| train-only Raw-PCA residual | 0.697351 | 0.739773 |
| FMT residual | 0.887125 | 0.945745 |
| FMT minus Raw-PCA | +0.189774 | +0.205972 |

The joint target was not reached: F1 gain is below `+0.195` and absolute FMT
F1 is below `0.893`. A few guarded warmup candidates raised absolute FMT F1
slightly in individual families (at most `+0.001377` for Tangaroa), but their
paired gain was lower because the Raw-PCA arm improved at least as much. Linear
warmup therefore does not improve the frozen Task3 objective and will not be
carried into a combination or confirmation experiment.

Selection and leaderboard SHA-256 values are
`18543e09...4e0` and `57bd7b20...858`. Confirmation remains closed.
