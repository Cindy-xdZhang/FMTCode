# Verify_Task3_ProjectionFeatureCombination_24.1

## Question

Can the family-specific auxiliary projection selected by 21.1 and the
family-specific FMT feature selected by 22.1 jointly recover absolute FMT
quality while retaining or increasing the large FMT-minus-Raw-PCA gap produced
by the ultra-narrow 20.1 bottleneck?

## Frozen 2x2 comparison

Only completed development selectors are admissible. The four candidates are:

| Candidate | Width | Projection | FMT feature |
|---|---|---|---|
| `c00_control` | 20.1 winner | historical `linear_layernorm_gelu` | frozen 5.2 feature |
| `c01_projection` | 21.1 winner | 21.1 winner | frozen 5.2 feature |
| `c02_feature` | 20.1 winner | historical `linear_layernorm_gelu` | 22.1 winner |
| `c03_projection_feature` | 21.1 winner | 21.1 winner | 22.1 winner |

Every selector path is bound by SHA-256 in the preflight manifest. Candidate-
specific FMT features are loaded separately; preflight requires the Raw arrays,
labels, sample counts, and split populations to remain byte-identical across
all four candidates.

## Paired protocol

- Ten development datasets and seven physical families are unchanged from 5.2.
- FMT and train-only Raw-PCA use the same candidate-specific auxiliary width,
  projection, head, optimizer, split, alpha search, and seeds 40--42.
- Raw-PCA dimensionality follows the candidate's FMT dimensionality and is fit
  using training data only.
- Confirmation remains closed. No 21.1 or 22.1 partial result is inspected.

Four candidates x 10 datasets x 3 seeds x 2 paired arms give 240 trainings in
40 GPU array children.

## Selection and preregistered targets

Within each physical family, a candidate must keep absolute FMT F1 and Average
Precision within `0.002` of `c00_control`. Eligible candidates are ranked by
FMT-minus-Raw-PCA F1 gain, then Average Precision gain, absolute FMT F1,
absolute FMT Average Precision, positive-dataset count, worst-dataset gain, and
worst-seed gain.

The joint development target is dataset-macro F1 gain `>= +0.175` and absolute
FMT F1 `>= 0.887`. A larger gap caused only by degrading FMT does not satisfy
the joint target. Any selected development recipe still needs a new unseen
spatial population before supporting a paper-level generalization claim.

