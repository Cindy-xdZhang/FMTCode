# Verify_Task3_AuxiliaryLinearBiasScale_85.1

## Question

Can the initial magnitude of auxiliary Linear biases increase supervised 3D
Task3 FMT gain without reducing absolute FMT F1 or Average Precision relative
to the exact 84.1 source recipe?

PyTorch initializes each Linear bias independently from a fan-in-dependent
uniform distribution. This experiment retains those sampled directions and
scales only their magnitude. It is distinct from 67.1, which changed
normalization bias, and from 83.1, which changes only Linear weights.

## Frozen protocol

This development experiment was preregistered while 77.1 was incomplete and
before 84.1 could produce any portfolio metric. It may use only the
independently audited 84.1 portfolio. Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control and bias scales 0, 0.1, 0.25, 0.5, 2, 4
  and 8;
- eight candidates × ten datasets × three seeds × two arms = 480 trainings;
- candidates multiply every existing `fmt_encoder` Linear bias by one common
  scale after all modules are built;
- weights, parameter count, source recipe, loss, optimizer and residual head
  remain unchanged; the operation consumes no random numbers.

A scale of one is represented only by the exact source control and does not
invoke an in-place operation. Both paired arms receive the same candidate
scale.

## Selection and safeguards

Selection occurs only after all 80 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source control values. Eligible candidates are
ranked by paired dataset-macro F1 gain and the frozen tie-breakers.

Targets are F1 gain at least `+0.224` and absolute FMT F1 at least `0.895`.
Meeting them does not open a confirmation population. Missing them is retained
as a valid negative result.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 480 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 480
temporary checkpoints remain only until 86.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`c8e7cc647e10862d8dba077ed2d1d439c5754f783742974dd61302f756e6e69f`.
