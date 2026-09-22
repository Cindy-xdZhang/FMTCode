# Couette raw-curl deployment 2.1 — submitted

2026-09-22, 11:16:41 UTC observed status. Construction and independent audit are complete; GPU training results are not available at this observation.

- Source commit pushed to `origin/codex/new_objective_primitive_rft`: `c5b3eb379000e0a6bb4b47dc99ea6dc8708ef853`.
- Experiment directory: `/ibex/user/zhanx0o/FMT_Task4C_CouetteRawCurl_2p1_20260922`.
- 110 source files and 29 incremental data files uploaded and remotely hash-verified. Source manifest SHA256 `ab70e231b93a9f021f4dc9cdaa2c8755261365f27e02f8a7bb33babdea7ae4e2`; data transfer manifest SHA256 `8c2ac79652faaeed61592785e0772d3973ddb35fa1f059ccf6922f8d5f8ab4aa`.
- Dataset audit SHA256 `6ade347de505c8cf3371e7a330743100e1cf2d7e9da5e9ad19dfd5576d4cb4a4`: 176059 seeds, 528177 curves, positive16029/negative160030. All 2068527 evaluated longest raw points independently checked. Old Channel/TBL geometry/labels remain read-only references.
- Local and remote tests: three raw-curl integration/boundary tests, four tangent augmentation tests pass. Additional local three-flow majority-vote metric check passes.

| Phase | Slurm job | Observed status |
|---|---|---|
| Append existing rows and verify sources |52285895|RUNNING; prepared.json already complete|
| Encode new Couette features/voxels |52285896_0–3, concurrency2|Dependency pending|
| Two GPU model pilots |52285897_0–1|Dependency pending|
| Four folds × FMT/Conv8 |52285898_0–7, concurrency2|Dependency pending|
| Audit all per-epoch test predictions |52285899|Dependency pending|

Remote preparation completed with exact old Channel/TBL row preservation and instance isolation. Couette appended 12000train/3000test to folds0–2,7044/1761 to fold3. Combined counts36000/9000 and31044/7761 respectively; no validation partition. Source verification and preparation report no errors. Single seed96611,50epochs maximum,FMT tangent jitter/dropout.35 and Conv8.15; original two-flow fold4 reused because Couette has no fifth test group.

Original8768 now serves `Other_Task4C_BundleGallery_1.6` and actual `mainExp_Task4C_CouetteDataset_2.1` labels/curves. Browser verified counts, unnormalized RK4 parameter text, test seed162595 instance0 with short17/32GT and longest131raw/head15/leg11, and no console warning/error. This replaces the old-label in-memory preview as the default actual-data view; original Channel/TBL gallery files remain unchanged.

Read status with `python -m experiments.Collect_Task4C_CouetteTraining_2_1`. Latest raw evidence: `outputs/mainExp_Task4C_CouetteTraining_2.1/remote_evidence/latest_status.json`; source/git association: `deployment_provenance.json` locally and at the remote experiment root. No duplicate submission or extra seed should be launched.

See [construction and training protocol](Task4C_couette_raw_training_execution_2.1.md) for full rules. Subsequent progress must be read from the collector, not inferred from this submission snapshot.
