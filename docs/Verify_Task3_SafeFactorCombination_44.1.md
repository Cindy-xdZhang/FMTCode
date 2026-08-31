# Verify_Task3_SafeFactorCombination_44.1

## Question

Can family-specific Task3 factors that individually preserve absolute FMT F1
and Average Precision combine with the completed 32.1 training residual scale
to exceed both development targets: F1 gain `>= +0.195` and absolute FMT F1
`>= 0.893`?

## Timing and evidence boundary

This adaptive development experiment was declared after 35.1 completed and
before any 36.1--43.1 metric was read. Searches 32.1--34.1 were already
complete. Linear warmup 35.1 is excluded because every family selected its
exact no-warmup control and its overall gain and absolute FMT F1 both fell
below target. This exclusion and the complete 44.1 candidate list are frozen
before the remaining selectors complete.

Every source must identify the same seven physical families, remain
confirmation-closed, and provide a completed family-specific recipe. Preflight
records each source selector SHA-256 and rejects feature or hyperparameter
conflicts before any GPU child starts.

## Frozen comparison

- Population and split: completed 5.2 development protocol.
- Feature: completed 22.1 family-specific anchored FMT representation.
- Head: two hidden layers, width 64, LayerNorm, GELU.
- Paired seeds: 40, 41, 42.
- Within every candidate, FMT and train-only Raw-PCA use the same trainable
  head, optimizer/loss recipe, initialization, batches, split, and budget.
- Confirmation remains closed.

Sources are 32.1 training alpha, 33.1 gradient clipping, 34.1 AdamW betas,
36.1 batch size, 37.1 positive-class weight, 38.1 dropout, 39.1 focal gamma,
40.1 parameter exponential moving average, 41.1 label smoothing, 42.1 AdamW
epsilon, and 43.1 cosine terminal learning rate.

## Preregistered grid

A complete 11-factor binary factorial would require 2,048 candidates and is
not proportionate. The fixed 28-candidate grid contains:

1. the exact anchored-feature control;
2. each of the eleven selected factors alone;
3. the 32.1 alpha winner paired separately with each other factor;
4. optimizer, loss/regularization, and stability stacks;
5. all non-scheduler factors, all factors, and all factors without alpha.

If a later selector chooses its exact control, the corresponding declared
44.1 candidate is retained rather than removed after seeing data. Across ten
datasets, 28 candidates produce 280 array mappings and 1,680 paired trainings.

## Selection rule

Each family/candidate must preserve FMT F1 and Average Precision relative to
the exact anchored-feature control with zero tolerance. Eligible candidates
are ordered by paired F1 gain, absolute FMT F1, paired Average Precision gain,
absolute FMT Average Precision, and registered robustness tie-breakers.

Reaching only one target is not success. Any development winner still requires
a fresh spatial-population confirmation before it can support a paper claim.

## Deployment

- Implementation commit: `db1c353`.
- Immutable archive SHA-256: `88867bb2a282765f03ca19efa517e6bce008e4485fed7353c93a6fc486e42b26`.
- Preflight job: `51018914`, with strict `afterok` dependencies on all eight
  36.1--43.1 selector jobs.
- GPU array: `51018920[0-279%24]`, strict `afterok:51018914`.
- Selector: `51018923`, strict `afterok:51018920_*`.
- Evidence archive: `51071240`, strict `afterok:51018923`. It will archive
  and hash all 1,680 per-run CSV files while requiring all 1,680 temporary
  checkpoints to remain present for 52.1; it never deletes a model. The local
  and remote script SHA-256 is
  `e463529b2ecaa2ec60560232acc261ecfd945b7241684204350d4eba2cb2e28e`.
- Local and remote combination contracts: 7/7 passed. Remote Python
  compilation and all three Slurm-script syntax checks passed.
