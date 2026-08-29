# Verify_Task3_OverlapBalancedCombination_16.1

## Question

Do the independently selected class-balanced mini-batch and Soft
Dice/Tversky overlap-loss recipes add to the frozen four-factor Task3 core?

## Frozen protocol

- The config is declared before the 11.1, 13.1, and 15.1 selectors are read.
- Preflight waits for all three complete selectors and freezes their SHA-256
  hashes and family-specific recipes. Partial rankings are never consumed.
- The four candidates form a complete 2×2 factorial around the 11.1 core:
  core, core+balanced, core+overlap, and core+balanced+overlap.
- FMT and same-width train-only Raw-PCA use identical combined training
  recipes, batches, architectures, random seeds, and training budgets.
- A duplicated hyperparameter with different source values is an explicit
  preflight error; no source silently overrides another.
- Only the development population is available. Confirmation remains closed.

## Scale and decision

Four candidates × 10 datasets × 3 seeds × 2 arms give 240 trainings in 40
GPU array children. Selection uses dataset-macro F1 gain over Raw-PCA, then
paired Average Precision gain, positive-dataset count, worst-dataset F1 gain,
and worst-seed F1 gain. The preregistered development target is `+0.17`.

A result below the target, including a negative interaction between balanced
batches and overlap loss, is retained. Any new development winner requires a
new unseen spatial population before it can support the paper conclusion.
