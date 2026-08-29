# Verify_Task3_ResidualHeadNormActivation_29.1

## Question

Can the normalization and activation inside the two-layer residual multilayer
perceptron improve both the absolute FMT classifier and its paired advantage
over train-only Raw-PCA?

## Why this is distinct

Completed Task3 searches have varied FMT features, auxiliary projections,
residual width/depth, losses, confidence gating, and several training
hyperparameters. Experiments 25.1--28.1 test semantic block projection,
training-only auxiliary supervision, learning rate/weight decay, and optimizer
family. None changes the normalization and nonlinear activation inside the
otherwise frozen two-layer residual head.

29.1 was fixed without reading partial or final 25.1--28.1 results. It is
therefore independent of those running searches rather than an adaptive search
chosen from their partial outputs.

## Frozen protocol

- Development populations, labels, splits, frozen Raw checkpoints, optimizer,
  training budget, early stopping, and fusion search come from 5.2.
- Each physical family uses the completed 22.1 anchored FMT feature.
- Both arms use the same two-layer, width-64 residual multilayer perceptron,
  zero dropout, and paired seeds `40, 41, 42`.
- Confirmation data remain closed.

## Complete factorial

Normalization is one of Layer Normalization, root-mean-square normalization,
or no normalization. Activation is one of Gaussian Error Linear Unit (GELU),
Sigmoid Linear Unit (SiLU), or Rectified Linear Unit (ReLU). Their Cartesian
product gives nine candidates. `LayerNorm + GELU` is the byte-compatible exact
historical control.

Every candidate is applied unchanged to both the FMT and train-only Raw-PCA
arms. The experiment contains 9 candidates, 10 data entries, 3 paired seeds,
and 2 arms: 540 training runs in 90 array jobs.

Normalization variants do not have identical parameter counts: each RMSNorm
layer removes the LayerNorm bias, while `none` removes both normalization
parameters. This is an explicit part of the tested head design, not hidden
capacity matching. Within every candidate, however, FMT and Raw-PCA use the
same head definition and dimensional contract. The preflight records every
total/trainable parameter count; all candidates remain below the frozen
Raw-wide cap of 148,225 parameters.

## Selection and targets

Selection is family-specific on exposed development data only. A candidate is
eligible only if its absolute FMT F1 and FMT Average Precision are each no
lower than the exact control. Eligible candidates are ranked first by paired
F1 gain, then absolute FMT F1 and the registered robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure to reach either value is a negative result. A development winner must
still be tested on a fresh spatial population before supporting a paper-level
generalization claim.

## Main files

- `config/Verify_Task3_ResidualHeadNormActivation_29.1.yaml`
- `FMT_Utils/PathlineClassifier_3D.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_residual_head_norm_activation_29_1.py`
- `ibex_bash/verify_task3_residual_head_norm_activation_29.1_*.sh`
