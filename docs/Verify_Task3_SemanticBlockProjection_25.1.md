# Verify_Task3_SemanticBlockProjection_25.1

## Question

Can a trainable projection that preserves the fixed semantic blocks of FMT
increase Task3 FMT-minus-Raw-PCA F1 while improving, rather than sacrificing,
absolute FMT classification quality?

## Mechanism

The historical auxiliary projection mixes every feature dimension in one
dense layer.  In 25.1, each contiguous label-free block is projected first and
the resulting branch outputs are concatenated before the unchanged two-layer
residual head:

- cached `fmt_all`: one block for each of the seven pathlines;
- anchored IVD features: one temporal discrete Fourier transform block and one
  scalar block for each retained time-domain anchor;
- concatenated features: the ordered concatenation of those declarations.

Three branch variants are tested: linear with GELU, linear with root-mean-
square normalization and GELU, and a two-layer branch with hidden width 16.
Raw-PCA receives the identical block widths and trainable network.  Thus the
paired arms differ only in their fixed FMT versus train-only Raw-PCA values.

## Frozen development protocol

- The complete family-specific `Verify_Task3_CombinedOptimization_11.1`
  winner supplies the high-absolute-quality training recipe.
- The complete `Verify_Task3_AnchoredFeatureDecomposition_22.1` selector
  supplies the alternative family-specific feature.  No partial 22.1 result
  is read.
- The grid is a preregistered 2-by-4 factorial: frozen 5.2 feature versus 22.1
  feature, each with dense control or one of three blockwise projections.
- Ten datasets, seven physical families, exposed development populations,
  IVD-p95 labels, splits, and paired seeds 40--42 remain unchanged.
- Eight candidates x 10 datasets x three seeds x two arms give 480 trainings
  in 80 GPU array children.  Confirmation remains closed.

## Decision rule

Within each physical family, candidates must keep both FMT F1 and Average
Precision within `0.002` of the exact `s00_control`.  Eligible candidates are
then ranked by FMT-minus-Raw-PCA F1 gain, with Average Precision gain and
absolute FMT metrics as the first tie-breakers.

The development targets are dataset-macro F1 gain `>= +0.160` and absolute
FMT F1 `>= 0.890`.  A larger gap obtained by lowering FMT does not satisfy the
joint target.  Any development winner still needs a fresh unseen spatial
population before it can support a paper-level generalization claim.

## Result

Ibex jobs `50996832` and `50996835` completed all 480 paired trainings and the
registered selector. Dataset-macro Raw-PCA/FMT F1 was `0.69666/0.88717`, a
gain of `+0.19052`; Average Precision was `0.74250/0.94809`, a gain of
`+0.20559`. All ten data entries had positive F1 gain.

The gain target passed, but the joint target failed because absolute FMT F1
was below `0.890`. Relative to 22.1, FMT F1 and Average Precision changed by
`-0.00205/-0.00236`; Raw-PCA deteriorated more. Therefore this experiment
supports a larger paired difference, not a stronger absolute FMT classifier.
Confirmation remains closed. Selection and leaderboard SHA-256 values are
`528dc082...eed9` and `64a0c9ed...ddf`.
