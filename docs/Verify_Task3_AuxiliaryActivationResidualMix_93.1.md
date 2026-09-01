# Verify_Task3_AuxiliaryActivationResidualMix_93.1

## Question

Can a fixed residual bypass around every auxiliary projection activation
increase supervised 3D Task3 FMT gain without reducing absolute FMT F1 or
Average Precision relative to the exact 92.1 source recipe?

For activation input `x`, the candidate computes
`(1-r) * activation(x) + r * x`. Unlike replacing the activation, this gives a
continuous path from the source nonlinearity (`r=0`) to identity (`r=1`) and
can preserve signed linear information and gradients.

## Frozen protocol

This development experiment was preregistered while 77.1 was incomplete and
before 92.1 could produce any portfolio metric. It may use only the
independently audited 92.1 portfolio. Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control and residual mixes 0.05, 0.1, 0.25, 0.5,
  0.75, 0.9 and 1;
- eight candidates × ten datasets × three seeds × two arms = 480 trainings;
- a candidate wraps every existing projection activation with the same fixed
  bypass; it adds no parameters or buffers;
- activation type, input scale and shift, projection width, normalization,
  Linear weights and biases, source recipe, optimizer, loss and residual head
  remain unchanged;
- wrapping consumes no random numbers.

Mix zero is represented only by `source`, which performs no module wrapping.
Both paired arms receive the same candidate mix.

## Selection and safeguards

Selection occurs only after all 80 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source control values. Eligible candidates are
ranked by paired dataset-macro F1 gain and the frozen tie-breakers.

Targets are F1 gain at least `+0.232` and absolute FMT F1 at least `0.899`.
Meeting them does not open a confirmation population. Missing them is retained
as a valid negative result.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 480 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 480
temporary checkpoints remain only until 94.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`7307df51308fe47cb35e7c9de4942b9947ee146aca5073b5eeae60ae03a5e20e`.
