# mainExp_Task4B_1.1 supervised implementation

> **Superseded before the full experiment ran.** Audit found cross-split
> primitive overlap in the 1.1 cache. The spatially buffered, contract-checked
> replacement is `mainExp_Task4B_1.2`; 1.1 smoke numbers are not research
> results.

## Status

The full three-seed experiment is implemented and frozen but not run locally.
A timing smoke test shows that the complete four-variant, 50-epoch budget would
exceed the project's five-minute local-run boundary, so it should be deployed
to Ibex rather than silently reduced.

The independent `Verify_Task4B_ClassifierSmoke_1.1` run used one seed, two
epochs and only FMT-only/Raw+FMT. It passed forward/backward training,
validation-only epoch selection, multiclass metrics and prediction export. Its
numbers are engineering diagnostics and are not Task4-b research results.

## Frozen comparison

- Raw: temporal-convolution pathline geometry only;
- Raw-wide: a no-FMT branch with more parameters than Raw+FMT;
- FMT-only: fixed 161-dimensional FMT followed by a neural classifier;
- Raw+FMT: Task3-style fusion of orientation-bearing Raw geometry and fixed FMT.

Every exact-balanced training batch contains the same number of examples from
all four classes. Raw coordinate normalization and per-dimension FMT
normalization are fitted on train only. Validation macro-F1 selects the best
epoch. Test is scored once after selection. The best state is retained in
memory only; no `.pt`, `.pth` or `.ckpt` file is written.

The full experiment uses seeds 7068, 7069 and 7070. Primary output is test
macro-F1, supplemented by per-class F1, balanced accuracy, one-vs-rest macro
Average Precision and confusion matrices. The main claim is paired
`Raw+FMT - Raw`; Raw-wide controls parameter count, while FMT-only diagnoses
whether frozen FMT is sufficient without Raw geometry.

## Important limitation

The seed-centered local primitive contains neither absolute z nor whole-instance
context. It cannot explicitly evaluate `Hx>Nx>Lx` or `Hz>Nz>Lz`; it can only
infer the proxy type from local flow geometry. All targets also remain proxy
labels from one steady channel volume.

## Code

- `FMT_Utils/Task4B_Classifier_3D.py`
- `experiments/Train_Task4B_FourClassClassifier_1_1.py`
- `config/mainExp_Task4B_1.1.yaml`
- `tests/test_task4b_classifier_3d.py`
