# Verify_Task3_AuxiliaryProjectionActivation_87.1

## Question

Can the auxiliary projection activation increase supervised 3D Task3 FMT gain
without reducing absolute FMT F1 or Average Precision relative to the exact
86.1 source recipe?

Experiment 21.1 changed activation together with projection normalization and
depth. Experiment 29.1 changed the residual-head activation rather than the
auxiliary projection. The present experiment isolates only activation modules
already present inside `fmt_encoder`.

## Frozen protocol

This development experiment was preregistered while 77.1 was incomplete and
before 86.1 could produce any portfolio metric. It may use only the
independently audited 86.1 portfolio. Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control, Identity, SiLU, ReLU, LeakyReLU with slope
  0.01, ELU, Mish and Tanh;
- eight candidates × ten datasets × three seeds × two arms = 480 trainings;
- candidates replace every existing projection activation recursively, so
  multilayer and blockwise projections use one common activation;
- no layer is added to a projection that had no activation;
- projection width, normalization, Linear weights and biases, parameter count,
  source recipe, loss, optimizer and residual head remain unchanged;
- activation replacement consumes no random numbers.

`source` performs no module replacement. Both paired arms receive the same
candidate activation.

## Selection and safeguards

Selection occurs only after all 80 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source control values. Eligible candidates are
ranked by paired dataset-macro F1 gain and the frozen tie-breakers.

Targets are F1 gain at least `+0.226` and absolute FMT F1 at least `0.896`.
Meeting them does not open a confirmation population. Missing them is retained
as a valid negative result.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 480 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 480
temporary checkpoints remain only until 88.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`c15561293c7d7023d3f5a0ebe8baecd26ffadd154947045eca004c9b96b31f9e`.
