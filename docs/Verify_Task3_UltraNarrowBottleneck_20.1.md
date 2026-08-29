# Verify_Task3_UltraNarrowBottleneck_20.1

## Question

Does the FMT advantage continue to increase when the identical trainable
auxiliary projection given to FMT and train-only Raw-PCA is narrower than four
dimensions?

## Motivation and frozen protocol

17.2 selected its minimum tested width, `auxiliary_dim=4`, for five of seven
physical families; Tangaroa selected 8 and F22 selected 96. This development
follow-up therefore tests widths 1, 2, and 3, while retaining 4, 8, and 96 as
exact controls and adding 6 and 12 to cover the transition.

- FMT and Raw-PCA use the same auxiliary width and classifier architecture.
- The 5.2 family feature recipes, 2×64 deep MLP, training protocol, labels,
  splits, and paired seeds 40–42 remain frozen.
- Every candidate must remain below the frozen Raw-wide parameter ceiling.
- Only the development population is read; confirmation remains closed.

## Scale and decision

Eight widths × 10 datasets × 3 seeds × 2 arms give 480 trainings in 80 GPU
array children. The preregistered target is dataset-macro F1 gain `+0.165`.
Absolute Raw-PCA and FMT F1/Average Precision are reported beside the gain.
If both degrade, the result can support robustness under a narrow bottleneck,
but cannot be called an absolute classifier improvement.
