# Verify_Task3_AuxiliaryActivationInputShift_91.1

## Question

Can a fixed offset on values entering each auxiliary projection activation
increase supervised 3D Task3 FMT gain without reducing absolute FMT F1 or
Average Precision relative to the exact 90.1 source recipe?

Experiment 85.1 scales the initialized Linear bias, whereas 91.1 adds a
non-trainable offset after the optional 89.1 activation-input scale and
immediately before the activation. This directly changes the nonlinear
operating threshold while leaving Linear parameters untouched.

## Frozen protocol

This development experiment was preregistered while 77.1 was incomplete and
before 90.1 could produce any portfolio metric. It may use only the
independently audited 90.1 portfolio. Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control and activation-input shifts -2, -1, -0.5,
  -0.25, 0.25, 0.5, 1 and 2;
- nine candidates × ten datasets × three seeds × two arms = 540 trainings;
- a candidate adds the same fixed shift before every existing projection
  activation; it adds no parameters or buffers;
- projection activation type, input scale, width, normalization, Linear
  weights and biases, post-projection feature scale, source recipe, loss,
  optimizer and residual head remain unchanged;
- wrapping consumes no random numbers.

Shift zero is represented only by `source`, which performs no module wrapping.
Both paired arms receive the same candidate shift.

## Selection and safeguards

Selection occurs only after all 90 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source control values. Eligible candidates are
ranked by paired dataset-macro F1 gain and the frozen tie-breakers.

Targets are F1 gain at least `+0.230` and absolute FMT F1 at least `0.898`.
Meeting them does not open a confirmation population. Missing them is retained
as a valid negative result.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 540 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 540
temporary checkpoints remain only until 92.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`9135a5bac0f28a5e5de416ffb95224f957aa5ae9991ed7e6c25b63ee515ffeaf`.
