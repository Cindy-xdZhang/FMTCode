# Verify_Task3_AnchoredFeatureSpatialReplay_46.1

## Question

Does the frozen family-specific anchored FMT representation selected by 22.1
retain its paired advantage on the third spatial population previously opened
by the failed 12.2 confirmation?

## Evidence boundary

This is a retrospective development replay, not a new confirmation. The 12.2
population and its results are already public to the development process. No
new spatial population is opened, and 46.1 must not be used as final test
evidence.

## Frozen comparison

- Ten datasets and whole-field IVD-p95 labels.
- Two 22.1 training seeds: 40 and 41.
- Per-family 22.1 feature winner, checkpoint, threshold, residual scale, Raw
  normalization, and train-only Raw-PCA transform are loaded unchanged.
- `FMT residual` and `Raw-PCA residual` keep the same auxiliary width,
  trainable architecture, Raw backbone, training budget, and target samples.
- Forty evaluations are performed; no training or threshold selection occurs.
- No checkpoint is created, copied, downloaded, or archived by 46.1.

## Interpretation

The preregistered descriptive target is dataset-macro F1 gain `>= +0.15`.
Passing would show that the 22.1 representation repairs the specific spatial
generalization weakness exposed by 12.2. Failing would mean the next method
must use all three exposed spatial populations during development before a
new, untouched confirmation population is generated.
