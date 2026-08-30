# mainExp_Task3_3D_6.1

## Scientific question

Does the frozen Task3 22.1 FMT residual retain at least `+0.15`
dataset-macro F1 advantage over its same-width, same-structure train-only
Raw-PCA residual on a genuinely unobserved fourth spatial primitive
population?

## Frozen method

- Ten 3D datasets and whole-field IVD-p95 binary labels.
- The 22.1 family-specific feature, model architecture, checkpoint, threshold,
  residual scale, Raw normalization, and train-only Raw-PCA transform.
- Paired seeds 40 and 41; 10 datasets x 2 seeds x 2 arms = 40 evaluations.
- No training, threshold tuning, feature selection, or model selection in 6.1.
- FMT and Raw-PCA arms use identical target samples and equal trainable model
  capacity.

## New population

The physical times and integration settings remain fixed so this experiment
isolates spatial primitive generalization.  Only the seed-grid phase changes.
Before any new primitive is generated, the phase is derived by:

1. SHA-256 of `mainExp_Task3_3D_6.1|fourth-spatial-population-v1`.
2. First eight hexadecimal digits modulo 1024, plus one: Halton index 417.
3. Centered radical inverses in bases 2, 3, and 5.

The resulting phase is
`[0.021484375, -0.34224965706447186, 0.0328]`.  It differs from all three
previously exposed phases.  The code, phase, source-model hashes, all 40
checkpoint hashes, and target are committed and frozen before cache or label
generation.

The existing temporal source packs are reused only as exact velocity windows;
they are independent of spatial seed phase.  Their parent manifest SHA-256 is
`020b48ffa9ee6d59b5c333a1b6513e015523c1e0db984884f1b9ade20a3e2a61`.

## Decision rule

- Primary: dataset-macro F1 gain at least `+0.15`.
- Aspirational: dataset-macro F1 gain at least `+0.17`.
- Always report Raw-PCA and FMT absolute F1, Average Precision, dataset-macro
  and family-macro gains, positive-dataset counts, and the worst dataset.
- Results are read only after all ten paired dataset shards finish.

## Execution boundary

The required order is source-manifest derivation, static preflight, recipe
freeze, source preflight, cache generation, IVD label generation, evaluation
preflight, ten paired evaluations, and one final summary.  A failed or partial
stage is recorded but never used to alter the frozen method.
