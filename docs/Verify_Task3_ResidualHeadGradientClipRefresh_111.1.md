# Verify_Task3_ResidualHeadGradientClipRefresh_111.1

## Purpose

Test whether limiting only the gradient norm of downstream residual-head
parameters improves the paired FMT residual over the same-width train-only
Raw-PCA residual. This factor has not been isolated before: 30.1 changed a
whole-model cap and 77.1 changes only the trainable auxiliary projection.

## Frozen protocol

- The configuration was frozen while 77.1 remained incomplete, before any
  109.1 metric existed, and before 110.1 could select a portfolio.
- Independently audited 110.1 contributes one complete source recipe per
  physical family; `h00_control_source` reproduces it exactly.
- Candidate head-only caps are `0.01, 0.03, 0.10, 0.30, 1, 3, 10, 30`.
- If a source recipe has global clipping, it runs first. Head-only clipping
  then covers all trainable parameters outside `fmt_encoder`; any source
  auxiliary-only cap subsequently covers only `fmt_encoder`. These two branch
  parameter sets are disjoint and the frozen Raw backbone is excluded.
- FMT and Raw-PCA use the same cap, model capacity, split, labels, mini-batch,
  initialization, seed and budget. Confirmation remains closed.
- Nine candidates x 10 datasets x three seeds x two arms give 540 trainings.

Selection first enforces zero loss in absolute FMT F1 and Average Precision
against the family control, then maximizes dataset-macro paired F1 gain. The
joint development target is F1 gain `>= +.250` and absolute FMT F1 `>= .908`.
Missing either target is a valid negative result.

## Status

Implementation and preregistration only. No performance artifact has been
read and no result conclusion is claimed.
