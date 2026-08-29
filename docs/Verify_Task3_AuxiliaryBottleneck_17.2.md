# Verify_Task3_AuxiliaryBottleneck_17.2

## Revision from 17.1

The 17.1 preflight rejected only `auxiliary_dim=128` because the
`deltaWing_resampled` residual model exceeded the frozen Raw-wide parameter
cap. No GPU child or performance evaluation ran. Version 17.2 therefore
removes only width 128; data, labels, seeds, model, training, selection, and
the `+0.16` development target are unchanged.

## Question and protocol

The experiment asks whether FMT preserves supervised IVD-vortex information
more efficiently than train-only Raw-PCA under the same narrow auxiliary
bottleneck. Widths are 4, 8, 16, 24, 32, 48, 64, and 96; width 64 is the
current control. Each paired arm uses the same model class, width, head,
parameter budget, optimizer, and training recipe. Preflight checks every
dataset and seed against the Raw-wide parameter cap.

Eight widths × 10 datasets × 3 seeds × 2 arms give 480 trainings in 80 GPU
array children. Selection uses dataset-macro F1 gain over Raw-PCA followed by
the frozen robustness tie-breakers. Absolute FMT and Raw-PCA F1/AP must be
reported beside gain; shared degradation is not evidence of improvement.
Confirmation remains closed.
