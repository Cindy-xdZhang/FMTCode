# Verify_Task3_AuxiliaryLinearWeightInitialization_83.1

## Question

Can a better initial basis for the paired auxiliary projection increase
supervised 3D Task3 FMT gain without reducing absolute FMT F1 or Average
Precision relative to the exact 82.1 source recipe?

This experiment isolates the initial weights of every `nn.Linear` layer inside
`fmt_encoder`. It is distinct from 30.1, which changed only the terminal output
layer of the residual head, and from 65.1, which changed normalization affine
parameters.

## Frozen protocol

This development experiment was preregistered while 77.1 was incomplete and
before 82.1 could produce any portfolio metric. It may use only the
independently audited 82.1 portfolio. Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control, Xavier-uniform and orthogonal weight
  initialization, each at gains 0.25, 0.5, 1.0 and sqrt(2);
- nine candidates × ten datasets × three seeds × two arms = 540 trainings;
- candidate initialization changes projection weights only: biases, parameter
  count, source recipe, loss, optimizer and residual head are frozen;
- the initialization helper restores the random-number-generator state, so
  candidates do not change later sampling or dropout sequences by consuming a
  different number of random draws.

The exact source control does not invoke an initializer and is byte-compatible
with the historical constructor. Both paired arms receive the same candidate
scheme and gain.

## Selection and safeguards

Selection occurs only after all 90 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source control values. Eligible candidates are
ranked by paired dataset-macro F1 gain and the frozen tie-breakers in the
configuration.

Targets are F1 gain at least `+0.222` and absolute FMT F1 at least `0.894`.
Meeting them does not open a confirmation population. Missing them is a valid
negative result and remains recorded.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 540 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 540
temporary checkpoints remain only until 84.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`e5e5d883659601de8f74a805069a92fc417cf10c0861574dcbdf1c83651ea948`.
