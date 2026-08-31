# Verify_Task3_AuxiliaryBeta1_61.1

## Question

After the independently audited 60.1 portfolio freezes the current best
development recipe per physical family, does assigning Adam's first-moment
coefficient (`beta1`) separately to the trainable auxiliary projection improve
absolute FMT classification and the paired FMT-minus-Raw-PCA advantage?

## Distinction from previous searches

Experiment 34.1 changed Adam moment coefficients globally. Experiment 59.1
changes only auxiliary `beta2`. This experiment inherits the 60.1 recipe,
including any selected auxiliary `beta2`, and changes only auxiliary `beta1`.
The downstream residual head retains both source optimizer betas.

## Frozen sequential protocol

- The source is the complete, independently audited 60.1 development portfolio.
- Only trainable auxiliary-projection parameters receive the candidate `beta1`;
  they retain source `beta2`. Frozen Raw parameters stay outside the optimizer.
- FMT and train-only Raw-PCA use the same candidate, parameter groups,
  initialization, split, seed, and training budget.
- The no-override control reproduces the exact source recipe. Confirmation is
  closed, and unfinished-array metrics cannot be inspected.

## Grid and decision

The absolute auxiliary `beta1` grid is
`{0,.25,.5,.7,.8,.85,.9,.95,.98,.99}`, plus one exact control. Eleven
candidates across ten datasets and three paired seeds give 110 array mappings
and 660 paired trainings.

A non-control winner must preserve both control FMT F1 and FMT Average
Precision under zero tolerance. Selection then uses paired dataset-macro F1
gain and the frozen robustness tie-breakers. The joint target is F1 gain at
least `+0.209` and absolute FMT F1 at least `0.893`. Missing either target is
retained and cannot open a confirmation population.

## Status

Preregistered before 59.1 or 60.1 produced any performance result. There is no
61.1 performance conclusion yet.
