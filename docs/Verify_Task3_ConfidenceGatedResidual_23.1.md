# Verify_Task3_ConfidenceGatedResidual_23.1

## Question

Can a deterministic gate derived from the frozen Raw classifier's confidence
preserve the high absolute FMT classifier quality of 11.1 while increasing the
paired FMT-minus-Raw-PCA Task3 gain?

## Motivation

The residual model currently applies every learned auxiliary correction with a
single validation-selected scalar `alpha`.  A correction is most useful where
the frozen Raw classifier is uncertain, whereas unrestricted corrections can
perturb already confident Raw decisions.  Experiment 20.1 produced the largest
development F1 gap (`+0.17375`) but lowered absolute FMT F1, so 23.1 explicitly
optimizes the gap subject to an absolute-quality guard.

## Frozen paired protocol

- The complete family-specific 11.1 winner is the starting point.  Its
  selection file is frozen by SHA-256 during preflight.
- The gate reads only the frozen Raw logit and uses no label at inference:
  `p=sigmoid(raw_logit/T)`, `u=4p(1-p)`, and
  `gate=floor+(1-floor)u`.
- FMT and train-only Raw-PCA receive the identical gate, architecture,
  auxiliary width, optimizer, split, training budget, alpha search, and seeds
  40--42.  Raw-PCA remains fitted on training data only.
- The grid is the complete factorial of temperatures `{0.5,1,2}` and floors
  `{0,0.25,0.5,0.75}`, plus an exact no-gate 11.1 control.
- Only the exposed development populations are read.  Confirmation remains
  closed.

## Scale and decision

Thirteen candidates x 10 datasets x 3 seeds x 2 paired arms give 780
trainings in 130 GPU array children.

Selection is family-specific.  Before ranking, a candidate must keep both
absolute FMT F1 and absolute FMT Average Precision within `0.002` of the exact
same-family control.  Eligible candidates are ranked by FMT-minus-Raw-PCA F1
gain, then Average Precision gain, absolute FMT F1, absolute FMT Average
Precision, positive-dataset count, worst-dataset gain, and worst-seed gain.

The preregistered development targets are F1 gain `>= +0.160` and absolute FMT
F1 `>= 0.887`.  A larger difference caused by degrading FMT cannot win.  A
development winner still requires evaluation on a new unseen spatial
population before supporting a paper-level generalization claim.
