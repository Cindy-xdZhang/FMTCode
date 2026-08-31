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

## Final development result

All 280 GPU children completed with exit code zero, producing all 1,680
declared paired trainings. Selector `51018923` completed in 17 seconds with
empty stderr. The family-specific selected Raw-PCA/FMT dataset-macro F1 is
`0.6856919/0.8904569`, hence the paired F1 gain is `+0.2047649`. Raw-PCA/FMT
Average Precision is `0.7298719/0.9490535`, hence the Average Precision gain is
`+0.2191816`. All 10 dataset entries have positive F1 gain; the worst entry is
deltaWing-resampled at `+0.0538466`.

The exact anchored-feature control has Raw-PCA/FMT F1
`0.6972269/0.8871248`, giving `+0.1898980`. Selection therefore raises
absolute FMT F1 by `+0.0033320` and paired F1 gain by `+0.0148670`, while
Raw-PCA F1 falls by `0.0115350`. Relative to the earlier 38.1 Dropout winner,
absolute FMT F1 rises by `+0.0021002` and paired F1 gain rises by `+0.0082859`.
The improvement is therefore not solely produced by weakening Raw-PCA.

The preregistered `+0.195` gain target is reached. The absolute FMT target
`0.893` is missed by `0.0025431`, so the joint target is not reached and this
remains a development result. The selected recipes are:

| Physical family | Selected combination |
|---|---|
| Boeing 747 | `k13_alpha_betas` |
| channel | `k26_all` |
| delta wing | `k05_positive_weight` |
| F-22 Raptor | `k25_all_non_scheduler` |
| half-cylinder | `k06_dropout` |
| Smoke Buoyancy | `k25_all_non_scheduler` |
| Tangaroa | `k18_alpha_ema` |

## Independent audit and retention

Evidence job `51071240` completed with empty stderr. It archived 1,680/1,680
per-run CSV files, retained all 1,680 temporary checkpoints for 52.1, and
published archive SHA-256
`ac55850ef26fd9b79f7301314025190abc61fcea3aba7dc7f6f7bd4aadae82e4`.
The independent audit does not import the selector implementation. It rebuilt
all seven family decisions, confirmed equal paired parameter counts and all
source hashes, and matched the selector to a maximum absolute difference of
`1.11e-16`. Its SHA-256 is
`f1fcff66125993b8b1b846ef9e5fd85b626099fcbea4008b48bd63532f7bf0ac`.

Final selector artifact SHA-256 values are:

- selection: `48dfa466ddb21a4faee2eaa071e1a204615ba2e7f2354944f57bfdc54ad55f12`;
- leaderboard: `b9e96f9e8cbf68baeb48ba4e88dd79792f49066c36514b328f0e8e8f872dbaad`;
- preflight manifest: `b9b0cbcfae89cd987cfcfb9dedb1085efc75658d19efefbf76d81e360377629b`.

The winner is an eligible source for 52.1, but only the later fresh
spatial-population confirmation can replace the current paper result.
