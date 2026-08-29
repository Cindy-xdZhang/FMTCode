# Verify_Task3_SupervisedContrastive_18.1

## Question

Can supervised contrastive regularization make the trainable FMT auxiliary
embedding more separable for IVD vortex/non-vortex classification than the
same-width train-only Raw-PCA embedding?

## Frozen protocol

- The frozen Raw classifier, development data, IVD labels, family-specific
  FMT recipes, deep-MLP architecture, optimizer, training budget, fusion
  selection, and paired seeds `[40, 41, 42]` are unchanged.
- The loss acts only on the existing trainable auxiliary projection; it adds
  no model parameters. Same-class embeddings are attracted and different
  classes appear in the softmax denominator.
- FMT and Raw-PCA use identical loss weight, temperature, mini-batch,
  Raw-derived sample weights, model, and random seed.
- Self-pairs are excluded. One-class or no-positive-pair mini-batches produce
  an exact zero auxiliary loss. The default weight is an exact no-op.
- No running selector or confirmation population is read.

## Search and decision

The 11 candidates contain a control, five loss weights from 0.001 to 0.1,
and temperature tests from 0.05 to 0.5. Ten datasets × 11 candidates × 3
seeds × 2 arms give 660 trainings in 110 GPU array children.

Selection uses dataset-macro F1 gain over Raw-PCA, then the frozen Average
Precision and robustness tie-breakers. The development target is `+0.16`.
Absolute FMT and Raw-PCA F1/AP remain mandatory. Any winner needs an unseen
spatial-population confirmation.
