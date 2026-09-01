# Verify_Task3_EarlyStoppingPatience_79.1

## Question

Can early-stopping patience improve supervised 3D Task3 FMT gain without
reducing absolute FMT F1 or Average Precision relative to the exact 78.1
source recipe?

Early-stopping patience is the number of consecutive validation epochs without
an improved selected F1 score that training tolerates before stopping. It does
not change the model, optimizer, maximum epoch budget, label, split or inference
procedure.

## Frozen protocol

This development experiment was preregistered while 75.1 was incomplete and
before reading any 75.1 performance metric. It may use only the independently
audited 78.1 portfolio after that portfolio becomes available. Confirmation
data remain closed.

- datasets: the ten frozen Task3 development entries;
- paired seeds: 40, 41 and 42;
- arms: `FMT -> projection -> residual head` and the same-width train-only
  `Raw-PCA -> projection -> residual head`;
- candidates: exact source control and patience 3, 5, 10, 15, 20, 30, 50, 80;
- only `training.patience` changes in a non-control cell;
- nine candidates × ten datasets × three seeds × two arms = 540 trainings;
- each source recipe's maximum epoch budget remains unchanged.

Previous experiments changed the maximum epoch budget or combined a longer
budget with patience 30, but no completed Task3 experiment isolated patience
around the current family-specific method. Therefore 79.1 tests a genuinely
unresolved stopping criterion rather than repeating TrainingHorizon 47.1.

## Selection and safeguards

Selection occurs only after all 90 array children succeed. Within each physical
family, a candidate is eligible only when its FMT F1 and FMT Average Precision
are both at least the exact source control values. Eligible candidates are
ranked by paired dataset-macro F1 gain, then the frozen tie-breakers in the
configuration.

Targets are F1 gain at least `+0.219` and absolute FMT F1 at least `0.893`.
Meeting them does not open a confirmation population. Missing them is a valid
negative result and must remain recorded.

## Evidence contract

`Audit_Task3_ParameterSearch.py` independently reconstructs all 540 run rows,
the guarded winner and macro metrics. The evidence job archives all per-run CSV
files without model checkpoints and verifies archive stability. The 540
temporary checkpoints remain only until 80.1 freezes 40 models and 40 result
files and passes its independent audit.

Canonical config SHA-256:
`9f7f6210c4a625f437b6d7bdbbb5686a5c54a7cfe409e4543ef6c2d398983977`.
