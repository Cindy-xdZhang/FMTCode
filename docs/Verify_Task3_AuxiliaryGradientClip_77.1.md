# Verify_Task3_AuxiliaryGradientClip_77.1

## Purpose

This development-only search tests whether limiting only the gradient norm of
the trainable auxiliary projection improves the paired FMT residual over the
same-width train-only Raw-PCA residual. It is distinct from the earlier global
gradient clipping search: the frozen Raw backbone and downstream residual head
retain the complete 76.1 family recipe.

## Frozen protocol

- The configuration was frozen while 75.1 was still running and before any
  75.1 performance artifact was read.
- 76.1 must first complete its independent audit. Its one recipe per physical
  family becomes the exact `g00_control_none` source.
- Candidate caps are `0.01, 0.03, 0.10, 0.30, 1, 3, 10, 30`; the control is a
  strict no-op. If global clipping is present in a source recipe, it runs first
  and the projection-only cap runs second.
- FMT and Raw-PCA use identical cap, architecture, capacity, training split,
  labels, mini-batches, seed and budget. Confirmation remains closed.
- Nine candidates × 10 datasets × three seeds × two arms give 540 trainings.

Selection first requires zero loss in absolute FMT F1 and Average Precision
against the family control, then maximizes dataset-macro paired F1 gain. The
joint development target is F1 gain `>= +0.218` and absolute FMT F1 `>= 0.893`.
Missing either target remains a valid negative result.

## Status

Implementation and preregistration only; no performance result has been read.
