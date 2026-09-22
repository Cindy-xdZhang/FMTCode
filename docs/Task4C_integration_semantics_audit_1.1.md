# Task4C integration semantics audit 1.1

2026-09-22: the user reported that the C++ viewer produces much longer Couette vortex lines with the same numerical step setting. Existing datasets, labels, training inputs, and services were not changed by this audit.

## Correction to the previous conclusion

Previous statement: 50 × 0.0002 gives physical half-length 0.01 and reproduces the user's C++ integration parameters.

Corrected statement: the existing Python dataset uses normalized-vector, arc-length integration. The matching local C++ source integrates the unnormalized sampled vector. The numerical step values therefore have different meanings. The prior construction/reintegration audit establishes internal reproducibility, not equivalence to the C++ software. Calling it verified reproduction of the user's integration was unjustified.

## Evidence

- `FMT_Utils/Task4C_FixedDataset_2_1.py:198–225`: `vtkStreamTracer`, Runge–Kutta45, `LENGTH_UNIT`, min/initial/max step all equal ds, maximum propagation tied to ds × steps. `cut_and_clean` checks physical arc against that product.
- Installed VTK 9.5.0 constant-field probe: for vector magnitudes 1, 10, and 100, 50 steps of 0.0002 all produce forward arc 0.00999999977648. Direct integration of those constant vectors would produce 0.01, 0.1, and 1.0 respectively.
- Official explanation: <https://vtk.org/doc/nightly/html/classvtkStreamTracer.html> explicitly states that streamline integration adopts normalized vectors and arc-length step units.
- The supplied curl formula matches local `C:/Users/xingdi/sources/optimal-connection/src/flow3d/Discrete3DFlowField.cpp:2600` and `C:/Users/xingdi/sources/GenericVariationalVortexCore/src/flow3d/Discrete3DFlowField.cpp:2613`.
- In optimal-connection, `StreamlineIntegrateOneDirImpl` at line 528 uses raw `sampleV` directly in Euler or RK4 updates; the sampler at line 613 returns the interpolated vector without normalization. GenericVariationalVortexCore has the same raw-vector structure at line 554.
- This identifies local source semantics. The currently running C++ executable, its selected method, input field preprocessing, and source revision have NOT been verified.

Equations: existing dataset integrates dx/ds = omega/|omega|. Raw-vector integration uses dx/dtau = omega, whose spatial length is the integral of |omega| over tau. Therefore steps × step size is not generally physical length in the latter method. Reparameterization preserves the ideal geometric orbit for the same nonzero field, but a fixed number of steps covers a different segment; numerical discretization and field reconstruction can add differences.

## Couette numerical control using the same existing Python curl field

All 199,490 seed curl magnitudes were queried. Minimum 13.5008, median 43.9733, maximum 393.4295. The field cache stores raw curl, not unit vectors; normalization happens inside the VTK tracer.

For seed #184465, position (1.07517189505, 1.19915822241, 0.31729558524), |omega| = 81.4961:

| Integration | Backward arc | Forward arc | Merged arc |
|---|---:|---:|---:|
| Saved middle-scale dataset curve | approximately .01 | approximately .01 | .02000005 |
| Raw-curl RK4, 50 updates, step .0002 | .57465450 | .70295475 | 1.27760925 |
| Raw-curl RK4, 49 updates, step .0002 | .56487554 | .68367683 | 1.24855236 |

The 50-update raw-vector control is about 63.88 times longer. This isolates a large effect of integration semantics; it is NOT a numerical reproduction of the user's C++ output. Raw RK4 controls use SciPy interpolation on the existing Python curl field, without a step-size convergence claim.

Two further differences need to be resolved before claiming exact C++ reproduction:

1. Local C++ code inserts the seed first and loops while `results.size() < maxIterationCount`; if 50 reaches this function unchanged, it gives 49 updates, not 50. Its RK4 and RK5 enum values currently enter the same RK4 branch. The selected UI path was not traced here.
2. Python curl uses coordinate-aware `np.gradient` on native, possibly nonuniform axes. Local C++ derivative functions use offsets span/(N−1), sample velocity at displaced positions, clamp at boundaries, and divide by the actual displacement. The curl component formula agrees, but the discrete derivative/interpolation and grid loading are not yet shown equivalent.

## Channel and TBL scope

`Task4C_FixedDataset_2_1.py` defines Channel ds=.001 with half-lengths .07/.10/.13, TBL ds=.025 with half-lengths 3.5/5/6.5. Both call the same normalized VTK tracer; `Task4C_V2NewLabel_1_1.py` reuses it. Earlier `Task4C_Bundles_1_1.py:140` and `Task4C_PaperBundles_3_1.py:165` also use VTK LENGTH_UNIT (with different adaptive step bounds). Normalized integration is not unique to newly added Couette.

This proves a parameter-semantics difference for those Python paths, not the size of their discrepancy against C++ and not a causal explanation for historical F1 scores. No old data was relabeled or retrained.

## Reproduction

Run `python -m experiments.Verify_Task4C_IntegrationSemantics_1_1`. Numeric output, installed VTK version, example seeds, and source hashes are saved in `outputs/Verify_Task4C_IntegrationSemantics_1.1/report.json`. No integrated curve arrays are published as a new dataset by this diagnostic.

Until C++ equivalence is established, retain mainExp_Task4C_CouetteDataset_1.1 as an arc-length-integrated historical artifact; do not represent it as reproduction of the C++ 50-step setting or use its low positive fraction as evidence about that intended setting.
