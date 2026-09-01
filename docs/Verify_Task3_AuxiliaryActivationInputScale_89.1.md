# Verify_Task3_AuxiliaryActivationInputScale_89.1

## Question

Can a fixed scale on values entering the auxiliary projection activation
increase supervised 3D Task3 FMT gain without reducing absolute FMT F1 or
Average Precision relative to the exact 88.1 source recipe?

Experiment 73.1 scaled the completed auxiliary feature after projection.
Experiment 89.1 instead changes the operating range of each existing
nonlinearity while leaving its output unscaled afterward. Multilayer and
blockwise projections apply the same scale before every activation.

## Frozen protocol

This development experiment was preregistered while 77.1 was incomplete and
before 88.1 could produce any portfolio metric. It may use only the
independently audited 88.1 portfolio. Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control and activation-input scales 0.125, 0.25,
  0.5, 0.75, 1.5, 2 and 4;
- eight candidates × ten datasets × three seeds × two arms = 480 trainings;
- a candidate wraps every existing projection activation with the same fixed
  input multiplier; it adds no parameters or buffers;
- projection activation type, width, normalization, Linear weights and biases,
  post-projection feature scale, source recipe, loss, optimizer and residual
  head remain unchanged;
- wrapping consumes no random numbers.

Scale one is represented only by `source`, which performs no module wrapping.
Both paired arms receive the same candidate scale.

## Selection and safeguards

Selection occurs only after all 80 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source control values. Eligible candidates are
ranked by paired dataset-macro F1 gain and the frozen tie-breakers.

Targets are F1 gain at least `+0.228` and absolute FMT F1 at least `0.897`.
Meeting them does not open a confirmation population. Missing them is retained
as a valid negative result.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 480 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 480
temporary checkpoints remain only until 90.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`92ec2cac4029b71d59263d10a9b9dbaa861467734a75a44c8cd7adc3ab2414cc`.
