# Verify_Task3_AuxiliaryLearningRate_55.1

## Question

Does assigning the trainable auxiliary projection a learning rate distinct
from the downstream residual head increase the paired Task3 F1 advantage of
FMT without reducing absolute FMT F1 or Average Precision?

## Motivation and scope

Completed searches 27.1 and 28.1 changed the optimizer globally, so projection
and residual-head parameters always moved at the same rate. Experiments 21.1,
25.1, and 53.1 changed the projection architecture or regularization but did
not test its optimization timescale. Fixed Fourier/kinematic FMT features and
train-only Raw-PCA components can have different conditioning even when their
width, trainable projection, and downstream head are matched.

This is a development-only single-factor experiment. It cannot replace the
audited spatial-population confirmations 6.1, 7.2, or 8.1.

## Frozen comparison

- IVD-p95 labels, exposed development populations, split, frozen Raw models,
  family-specific 22.1 FMT features, residual network, optimizer family, base
  learning rate, weight decay, epoch budget, early stopping, and paired seeds
  remain unchanged.
- Only parameters inside the trainable auxiliary projection receive the
  candidate multiplier. The downstream residual head keeps the base learning
  rate. Frozen Raw parameters remain excluded from the optimizer.
- FMT and train-only Raw-PCA use the same multiplier, parameter groups,
  trainable parameter count, initialization, split, and budget in each cell.
- Multiplier `1` uses the historical single optimizer parameter group, with no
  local candidate override. Confirmation remains closed.

## Registered grid and selection

The multiplier grid is `{0.05,0.10,0.25,0.50,1,2,4,8,16}`. Nine candidates
across ten datasets and three paired seeds yield 90 array mappings and 540
paired trainings.

Selection is physical-family specific. A non-control candidate is eligible
only if its FMT F1 and FMT Average Precision are both no lower than the exact
multiplier-one control. Eligible candidates are ranked by paired dataset-macro
F1 gain, absolute FMT F1, and the registered robustness tie-breakers. The joint
target is F1 gain at least `+0.200` and absolute FMT F1 at least `0.893`.
Failure of either target is retained as a negative result.

## Main files

- `Verify_Task3_FMTResidual.py`
- `Search_Task3_LossOptimization_7_1.py`
- `config/Verify_Task3_AuxiliaryLearningRate_55.1.yaml`
- `tests/test_task3_auxiliary_learning_rate_55_1.py`
- `ibex_bash/verify_task3_auxiliary_learning_rate_55.1_*.sh`

## Status

Completed and independently audited on Ibex.

- Jobs `51091862[0-89]`, `51091865`, and `51091883` completed with exit code
  zero and empty stderr. All 90 array cells and 540 paired trainings finished.
- The selected development result is Raw-PCA/FMT F1
  `0.696707/0.887717`, a gain of `+0.191010`. The Average Precision gain is
  `+0.206293`; all ten datasets are positive and the worst dataset F1 gain is
  `+0.049498`.
- Relative to the exact multiplier-one control, the family-specific search
  improves F1 gain by only `+0.001622` and absolute FMT F1 by `+0.000655`.
  It does not reach the registered `+0.200/0.893` joint target and remains
  weaker than the completed 54.1 portfolio (`+0.205517/0.890218`).
- Selected multipliers are `4` for channel, deltaWing, and f22raptor; `0.25`
  for halfcylinder; `16` for tangaroa; and the control `1` for boeing747 and
  smokeBuoyancy.
- The remote independent audit differs from the selector by at most
  `2.22e-16`. The downloaded archive SHA-256 is
  `3c043032088f8088f11d8f84f3ce59691bde90a15cb69f3b6d3fd72b6ed02921`;
  extracting all 540 per-run CSV files and rerunning the auditor locally
  reproduces the selected recipes and macro metrics exactly.
- After the 56.1 portfolio audit, cleanup job `51092670` deleted exactly 540
  temporary checkpoints and left zero; the CSV archive and summaries remain.

Conclusion: assigning the auxiliary projection a distinct learning rate is a
valid but small local effect and does not improve the current best development
portfolio. This negative result is retained; confirmation was not opened.
