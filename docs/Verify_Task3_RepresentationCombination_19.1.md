# Verify_Task3_RepresentationCombination_19.1

## Question

Do the frozen Task3 training core, same-width auxiliary bottleneck, and
supervised contrastive objective have complementary effects on the FMT gain
over train-only Raw-PCA?

## Frozen protocol

- This experiment was declared before the 18.1 selector was read.
- Preflight waits for the complete 11.1, 17.2, and 18.1 selectors, then freezes
  their SHA-256 hashes and family-specific recipes. Partial results are never
  consumed.
- The eight candidates form the complete 2^3 factorial: base; each of core,
  bottleneck, and contrastive alone; all three pairs; and all three together.
- FMT and same-width train-only Raw-PCA use identical recipes, architecture,
  random seeds, batches, optimizer, and training budget for each candidate.
- Conflicting source hyperparameters are rejected instead of overwritten.
- Only the development population is available. Confirmation remains closed.

## Scale and decision

Eight candidates × 10 datasets × 3 seeds × 2 arms give 480 trainings in 80
GPU array children. Selection uses dataset-macro F1 gain over Raw-PCA, then
Average Precision gain, positive-dataset count, worst-dataset F1 gain, and
worst-seed F1 gain. The preregistered development target is `+0.165`.

Absolute Raw-PCA and FMT F1/Average Precision must be reported beside their
difference. A larger difference caused by both arms degrading is not described
as an absolute classifier improvement. Any new development winner needs a new
unseen spatial population before supporting the paper conclusion.
