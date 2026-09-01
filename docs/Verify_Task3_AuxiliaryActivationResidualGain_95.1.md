# Verify_Task3_AuxiliaryActivationResidualGain_95.1

## Question

Can an additive linear bypass around every complete auxiliary-projection
activation response increase supervised 3D Task3 FMT gain without reducing
absolute FMT F1 or Average Precision relative to the exact 94.1 source recipe?

For the current activation response `f(x)`, the candidate computes
`f(x) + g*x`. This differs from 93.1: residual mix attenuates `f(x)` as the
bypass grows, whereas residual gain preserves the complete learned nonlinearity
and independently controls the signed linear path.

## Frozen protocol

This development experiment was preregistered while 77.1 was incomplete and
before 94.1 could produce any portfolio metric. It may use only the
independently audited 94.1 portfolio. Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control and residual gains 0.01, 0.03, 0.1, 0.25,
  0.5, 1, 2 and 4;
- nine candidates × ten datasets × three seeds × two arms = 540 trainings;
- the bypass is added after any activation override, input scale, input shift
  and 93.1 residual mix; it adds no parameters or buffers;
- projection width, normalization, Linear weights and biases, source recipe,
  optimizer, loss and residual head remain unchanged;
- wrapping consumes no random numbers.

Gain zero is represented only by `source`, which performs no module wrapping.
Both paired arms receive the same candidate gain.

## Selection and safeguards

Selection occurs only after all 90 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source-control values. Eligible candidates are
ranked by paired dataset-macro F1 gain and the frozen tie-breakers.

Targets are F1 gain at least `+0.234` and absolute FMT F1 at least `0.900`.
Meeting them does not open a confirmation population. Missing them is retained
as a valid negative result.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 540 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 540
temporary checkpoints remain only until 96.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`4852e22d9eac699280c94c60ca3881a6bec8d2e4538d50818a5f45177b51290e`.
