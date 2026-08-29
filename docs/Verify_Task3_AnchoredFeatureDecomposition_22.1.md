# Verify_Task3_AnchoredFeatureDecomposition_22.1

## Question

Can the anchored IVD feature be simplified into a more useful combination of
early-time anchors and temporal discrete Fourier transform coefficients while
increasing the FMT gain without lowering absolute FMT classifier quality?

## Frozen development protocol

This is a development-only decomposition of the family-specific feature search
from `Verify_Task3_SpatialRobust_5.2`.

- The same 10 datasets, exposed training and robust-validation spatial
  populations, IVD-p95 labels, split, training budget, and paired seeds 40–41
  are reused.
- Five exact 5.2 Stage1 feature controls are included. Therefore every physical
  family has a registered fallback that reproduces its prior feature recipe.
- FMT and train-only Raw-PCA use the same auxiliary width and trainable residual
  architecture for every candidate.
- Feature construction is deterministic and label-free. Only the validation
  labels rank completed candidates.
- No current confirmation or outer ordinal is opened.

## Candidates

The 11 new candidates separate the original anchored representation into:

- `first`: seed-time IVD-deviation magnitude only;
- `early`: first value plus the early-quarter mean;
- `dft`: temporal Fourier coefficients only;
- `core`: Fourier coefficients plus first and early-quarter mean;
- `stats`: `core` plus full-window mean and standard deviation.

The same decomposition is tested near the one-frequency/three-step and
two-frequency/eight-step controls, including second-order endpoint and
`fmt_all` concatenations. Extrema and the last-time anchor are omitted from the
new recipes because they may encode late-window noise instead of seed-time IVD.

## Scale and preregistered decision

Sixteen candidates × 10 datasets × two seeds × two paired arms give 640
trainings in 160 GPU array children.

Selection is family-specific and ordered as follows:

1. require both FMT F1 and FMT Average Precision to remain within `0.002` of
   that family's frozen 5.2 Stage1 control;
2. maximize FMT minus same-width Raw-PCA validation F1;
3. tie-break by Average Precision gain, worst-seed F1 gain, and absolute FMT F1.

The development targets are dataset-macro F1 gain `>= +0.150` and absolute FMT
F1 `>= 0.890`. A candidate that increases the gap by degrading FMT cannot win.
Any new development winner still requires a fresh, unseen spatial population
before it can replace a paper method.
