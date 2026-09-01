# Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1

## Purpose

After 81.1 is complete and independently audited, compare its guarded winner
with the independently audited 80.1 portfolio. This is a training-free
per-physical-family selection step; it performs zero new trainings.

## Frozen contract

- sources: 80.1 current portfolio and 81.1 early-stopping `min_delta` search;
- ten frozen development dataset entries and seven physical families;
- source paired seeds: 40, 41 and 42;
- frozen artifacts: seeds 40 and 41, both paired arms, giving 40 models and
  40 per-run result files;
- zero-tolerance absolute FMT F1/Average Precision source guard;
- primary rank: paired dataset-macro F1 gain, followed by the preregistered
  tie-breakers;
- confirmation data remain closed.

The selector must verify the 600-row 81.1 archive and both source audits before
copying artifacts. An independent auditor that does not import the 82.1
selector reconstructs all seven family choices, macro metrics and 80 frozen
file hashes. Only after that audit passes may the 600 temporary 81.1 candidate
checkpoints be removed.

Targets are dataset-macro F1 gain at least `+0.220` and absolute FMT F1 at
least `0.893`. Audit success means evidence consistency, not target success.

Canonical config SHA-256:
`806c0a71011bd55418b1f064c96d82412c1dd7c74f31b61f962f1dacbf3acbb4`.
