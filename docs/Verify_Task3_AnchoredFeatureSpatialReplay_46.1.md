# Verify_Task3_AnchoredFeatureSpatialReplay_46.1

## Question

Does the frozen family-specific anchored FMT representation selected by 22.1
retain its paired advantage on the third spatial population previously opened
by the failed 12.2 confirmation?

## Evidence boundary

This is a retrospective development replay, not a new confirmation. The 12.2
population and its results are already public to the development process. No
new spatial population is opened, and 46.1 must not be used as final test
evidence.

## Frozen comparison

- Ten datasets and whole-field IVD-p95 labels.
- Two 22.1 training seeds: 40 and 41.
- Per-family 22.1 feature winner, checkpoint, threshold, residual scale, Raw
  normalization, and train-only Raw-PCA transform are loaded unchanged.
- `FMT residual` and `Raw-PCA residual` keep the same auxiliary width,
  trainable architecture, Raw backbone, training budget, and target samples.
- Forty evaluations are performed; no training or threshold selection occurs.
- No checkpoint is created, copied, downloaded, or archived by 46.1.
- Evaluation may use CPU or GPU because all learned values are frozen; the
  actual device is recorded per Slurm process and no device result is mixed
  within a dataset shard.

## Interpretation

The preregistered descriptive target is dataset-macro F1 gain `>= +0.15`.
Passing would show that the 22.1 representation repairs the specific spatial
generalization weakness exposed by 12.2. Failing would mean the next method
must use all three exposed spatial populations during development before a
new, untouched confirmation population is generated.

## Completed result

Ibex jobs `51025812`, `51026592`, and `51026610` completed successfully on
2026-08-30. The frozen comparison produced:

- Raw-PCA residual / FMT residual F1: `0.695856 / 0.865860`.
- Dataset-macro F1 gain: `+0.170004`.
- Raw-PCA residual / FMT residual average precision: `0.753390 / 0.940384`.
- Dataset-macro average-precision gain: `+0.186994`.
- Positive F1 gain on all `10/10` datasets and all `7/7` physical families.
- Minimum dataset F1 gain: `+0.051862`.

The descriptive `+0.15` target therefore passed. This result supports taking
the 22.1 representation unchanged to a new fourth spatial population, but it
does not constitute final confirmation because the replayed population was
already exposed by 12.2.

Artifact identities:

- Preflight manifest SHA-256: `c2be783b19642deecb51e64b5daf90dedc6389a08b10074a30e824cdca8dd3be`.
- Per-run CSV SHA-256: `26d475a87c572460992b47357cb0b9a1f62de224c0dfd40b503098375c66ef26`.
- Summary JSON SHA-256: `23e777a13e2b684016ae4869c3a1c68efadf91dda87829adbb77f0cd8220d2b1`.
