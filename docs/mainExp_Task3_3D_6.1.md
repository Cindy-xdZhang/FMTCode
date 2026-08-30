# mainExp_Task3_3D_6.1

> Status: **COMPLETED and independently validated.**  The frozen FMT arm
> improves dataset-macro F1 by `+0.17066` and Average Precision by
> `+0.19202` on the previously unseen fourth spatial population.  Both the
> primary `+0.15` target and the aspirational `+0.17` target are met.

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

## Final confirmation results

Values are the mean and population standard deviation over paired seeds 40
and 41.  F1 and Average Precision gains are computed seed by seed before
aggregation.

| Flow | Raw-PCA F1 | Raw+FMT F1 | F1 gain | Raw-PCA AP | Raw+FMT AP | AP gain |
|---|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .789538±.002161 | .867603±.002101 | **+.078065±.000060** | .865797±.001505 | .947667±.000035 | **+.081870±.001540** |
| Channel observer | .138257±.003402 | .840652±.004198 | **+.702395±.007600** | .086549±.003690 | .904102±.014977 | **+.817553±.011287** |
| Half-cylinder Re160 | .718851±.009662 | .826600±.006484 | **+.107750±.016146** | .781454±.029986 | .931750±.003942 | **+.150296±.033928** |
| Delta-wing original LBM | .853935±.001476 | .910195±.000311 | **+.056260±.001787** | .942036±.001014 | .976750±.000161 | **+.034713±.000853** |
| Delta-wing resampled | .856523±.001238 | .914197±.002080 | **+.057674±.000842** | .951593±.002119 | .976659±.000259 | **+.025065±.001860** |
| F-22 | .804927±.013832 | .917333±.002196 | **+.112406±.011636** | .860859±.006906 | .976365±.000009 | **+.115505±.006915** |
| Half-cylinder Re640 | .679887±.014504 | .900564±.004372 | **+.220677±.010132** | .751821±.009500 | .966088±.001074 | **+.214267±.008425** |
| Half-cylinder Re6400 | .556994±.009116 | .797466±.002077 | **+.240472±.007039** | .580743±.011710 | .888833±.003268 | **+.308090±.014978** |
| Smoke buoyancy | .784442±.007154 | .832955±.006537 | **+.048513±.000617** | .878167±.005726 | .938092±.001753 | **+.059926±.007479** |
| Tangaroa | .764686±.005639 | .847051±.004838 | **+.082366±.000800** | .827288±.004815 | .940211±.001319 | **+.112923±.003496** |
| **Dataset macro** | **.694804** | **.865462** | **+.170658** | **.752631** | **.944652** | **+.192021** |
| **Family-macro gain** | — | — | **+.181478** | — | — | **+.205983** |

All 10 datasets and all 7 physical families have positive paired F1 and
Average Precision gains.  The smallest dataset-level F1 gain is Smoke
buoyancy at `+0.04851`; the result is therefore not driven only by the channel
outlier.  This experiment supports the paper claim that adding FMT improves
the frozen supervised IVD-p95 vortex detector across the tested 3D flows.

## Execution and audit evidence

- Static preflight, freeze, and source preflight jobs: `51028047`, `51028079`,
  and `51028080`; all completed before the fourth spatial population was
  generated.  Frozen recipe SHA-256:
  `0469dcc85a83822fae53d7205a75a99339e915ea026ec256f07cd63409931bd5`.
- Cache array `51028082[0-9]` completed on GTX 1080 Ti and Tesla P100 nodes;
  label array `51028083[0-1]`, evaluation preflight `51028084`, CPU evaluation
  array `51028087[0-9]`, and summary `51028091` all completed with exit code 0.
- Evaluation-preflight SHA-256:
  `1c1232f77ef91fc5bd17d1169e50696a0a8a1fa7a6ee620449ee7af68513e7bf`.
  `per_run.csv` SHA-256:
  `bc5a61b992e776fe24497f28047c1fe2ba5dd1400e780653eefe57a6f91ef591`.
  `summary.json` SHA-256:
  `f46a307a76fa7674574d6751bec27ac1e8d3443d706349757d8e86ebac5fb756`.
- The downloaded CSV/JSON mirror was independently rehashed and recomputed:
  40 unique dataset-seed-arm rows, 10 datasets, 40 frozen models, exact
  manifest hashes, and all reported macro values matched the remote summary
  within `1e-12`.  No checkpoint was downloaded or created by 6.1.
