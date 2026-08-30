# Verify_Task3_BatchSize_36.1

## Question

Can mini-batch size improve absolute FMT classification while increasing its
paired advantage over the same-width train-only Raw-PCA residual arm?

## Evidence motivating a complete grid

The completed 7.1 search tested only batch sizes 256 and 1024 around the 512
control. Their effects were small and physical-family dependent: batch 256 was
rank 2 for the half-cylinder family, while batch 1024 modestly improved the
absolute FMT result in several single-dataset families. Neither sparse point
establishes whether the useful regime lies at a smaller or larger batch.

36.1 therefore treats batch size as an independent optimization variable and
tests `{32, 64, 128, 256, 512, 1024, 2048}`. The 512 cell is the exact
historical control. The epoch budget is fixed, so every candidate sees the same
number of data passes; optimizer-step count changes as an intentional property
of batch size and will be reported rather than hidden.

## Frozen comparison

- Development populations, labels, split, frozen Raw checkpoints, optimizer,
  learning rate, loss, epoch budget, early stopping, and fusion search follow
  the completed 5.2 development protocol.
- The completed 22.1 selector supplies one anchored FMT feature per physical
  family.
- Both arms use the same two-hidden-layer width-64 residual multilayer
  perceptron with LayerNorm, GELU, and zero dropout.
- Paired seeds are `40, 41, 42`.
- Within every candidate, FMT and train-only Raw-PCA use the same batch size,
  sample-order seed, initialization, labels, network, and training budget.
- Confirmation data remain closed.

Seven candidates across ten datasets produce 70 array mappings and 420 paired
trainings.

## Selection and targets

Selection is physical-family specific on exposed development data. A candidate
is eligible only when absolute FMT F1 and FMT Average Precision are each no
lower than the batch-512 control. Eligible candidates are ranked by paired F1
gain, then absolute FMT F1 and the registered robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure of either target is retained as a negative result. A development
winner still requires a fresh spatial-population evaluation before it can
support a paper-level generalization claim.

## Main files

- `config/Verify_Task3_BatchSize_36.1.yaml`
- `Verify_Task3_FMTResidual.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_batch_size_36_1.py`
- `ibex_bash/verify_task3_batch_size_36.1_*.sh`

## Deployment status

Implementation commit `26f0218` was archived and deployed byte-for-byte to
Ibex. The immutable archive SHA-256 is
`9f7c3dcc4caf491ea317ca037147bfd937d2560baf93ffebe3056a34041d6b60`.
Local full preflight passed; local and remote test suites each passed 25/25
tests; remote Python compilation and all three Slurm scripts passed their
static checks.

Submitted at `2026-08-30T05:24:33+03:00`:

- preflight `51006631`, completed `05:25:39` on `cn604-11`, exit code 0 with
  empty stderr; and
- GPU array `51006632[0-69%24]`, waiting on `QOSMaxGRESPerUser`; and
- selector `51006633`, waiting on the complete array.

The downloaded preflight manifest SHA-256 is `f167c936…8355`; it confirms ten
datasets, seven candidates, 70 mappings, 420 paired trainings, zero ineligible
cells, and `confirmation_opened=false`.

No partial performance output will be read. The selector alone may update the
performance conclusion after all 70 children finish successfully.
