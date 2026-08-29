# Verify_Task3_AuxiliaryBottleneck_17.1

## Question

Does FMT preserve supervised IVD-vortex information more efficiently than
train-only Raw-PCA when both residual arms pass through the same narrow
auxiliary bottleneck?

## Frozen protocol

- Data, development split, IVD labels, frozen Raw classifier,
  family-specific FMT recipes, deep-MLP head, optimizer, epochs, alpha search,
  and seeds `[40, 41, 42]` are unchanged.
- The only candidate variable is `auxiliary_dim`: 4, 8, 16, 24, 32, 48,
  64, 96, or 128. Width 64 is the current control.
- For each dataset, width, and seed, FMT and train-only Raw-PCA use the same
  model class, bottleneck width, head, parameter budget, and training recipe.
- Preflight verifies paired parameter counts and rejects any residual model
  reaching the Raw-wide parameter cap.
- No confirmation population is opened and no other running selector is read.

## Scale and decision

Nine widths × 10 datasets × 3 seeds × 2 arms give 540 trainings in 90 GPU
array children. The primary metric remains dataset-macro F1 gain over
Raw-PCA, followed by Average Precision gain and the frozen robustness
tie-breakers. The development target is `+0.16`.

Absolute FMT and Raw-PCA F1/AP are reported beside the gain. A large gain
caused only by both methods degrading is not evidence that the FMT classifier
improved. Any selected width requires an unseen-population confirmation.
