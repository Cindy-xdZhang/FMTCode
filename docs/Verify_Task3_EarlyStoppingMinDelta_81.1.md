# Verify_Task3_EarlyStoppingMinDelta_81.1

## Question

Can the early-stopping minimum improvement threshold increase supervised 3D
Task3 FMT gain without reducing absolute FMT F1 or Average Precision relative
to the exact 80.1 source recipe?

`min_delta` is the smallest increase in the validation selection score that
counts as a new best checkpoint. It changes neither the model nor the loss; it
only controls checkpoint replacement and the stale-epoch counter.

## Frozen protocol

This development experiment was preregistered before 79.1 produced any
performance metric. It may use only the independently audited 80.1 portfolio.
Confirmation data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control and `min_delta` 0, 1e-6, 3e-6, 1e-5,
  3e-5, 3e-4, 1e-3, 3e-3 and 1e-2;
- only `training.min_delta` changes in a non-control cell;
- ten candidates × ten datasets × three seeds × two arms = 600 trainings;
- source patience and maximum epoch budget remain unchanged.

The historical source value is expected to be 1e-4 and is retained by the
exact source control. No completed Task3 experiment isolated `min_delta`.

## Selection and safeguards

Selection occurs only after all 100 array children succeed. Within each
physical family, a candidate is eligible only when its FMT F1 and FMT Average
Precision are both at least the exact source control values. Eligible
candidates are ranked by paired dataset-macro F1 gain and the frozen
tie-breakers in the configuration.

Targets are F1 gain at least `+0.220` and absolute FMT F1 at least `0.893`.
Meeting them does not open a confirmation population. Missing them is a valid
negative result and remains recorded.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 600 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 600
temporary checkpoints remain only until 82.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`e18fb48102551a67fa505604db4f9a2cf351d9ff78a602eb8ab3a824fd977a73`.
