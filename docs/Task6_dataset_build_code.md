# Task6 dataset construction code

The canonical input is the shared `Task6_CorelineDataset_2.7_share` package: full instantaneous observed velocity (v-u, or v for steady tornado), physical coordinates, and published human coreline labels. The optional preintegrated bundles never replace those inputs. Dataset arrays, checkpoints, annotation drafts, generated binaries and machine logs are not committed to Git.

## Build the optional 21-streamline samples

Use Python 3.11 with NumPy, SciPy, Numba, VTK, netCDF4 and PyTorch. PyTorch is currently an import dependency of historical shared utilities; bundle integration runs on the CPU. The detailed frozen protocol is in [Task6_icosa_bundles_1.1.md](Task6_icosa_bundles_1.1.md).

Copy `config/mainExp_Task6_IcosaBundles_1.1.json` to a local configuration, then set `dataset_root`, `output_root`, `workers` and `threads_per_worker` for your machine. Paths in the committed configuration describe the original run. Keep the scientific parameters unchanged to reproduce it. Run from the repository root:

```powershell
python -m experiments.Build_Task6_IcosaBundles_1_1 pilot --config path/to/local_config.json
python -m experiments.Build_Task6_IcosaBundles_1_1 build --config path/to/local_config.json
python -m experiments.Publish_Task6_IcosaBundles_1_1 --config path/to/local_config.json
```

`--keys-file` optionally selects a JSON list of frame keys. The two committed key lists reproduce the original parallel scheduling groups; they do not change the dataset split. Do not launch overlapping builds on the same frame directory. Repeating the build resumes checkpoints. The publisher retries incomplete frames, verifies hashes and numerical outputs, recomputes sample labels against the current human corelines, and writes `bundles.json`. Only a catalog with `state: COMPLETE` indicates the whole build has finished. A preallocated array or a partial catalog is not evidence of completion. The optional `--wait-pid` publisher option is Windows-only; without it the publisher runs directly after the builder exits.

Each frame has 10,000 centers, each with 20 icosahedron face-center neighbors at radius h and one center streamline. All curves use the frozen adaptive integration and 65-point equal-arclength representation. SquareCylinder uses the explicitly configured valid-region exclusion and 95% maximum threshold; other fields use 80%. No threshold is silently relaxed after rejection.

`experiments/task6_field_tools/task6_bundles.py` is the standalone collaborator reader and exact bundle reintegration helper. The publisher copies it into the shared package. Preserve the package's accompanying `task6_fields.py` and `tools/` directory.

## Generate additional observed fields and automatic corelines

Use the shared package's `tools/ggt17_observe.py` for a new observed field; see [Task6_ggt17_portable_1.1.md](Task6_ggt17_portable_1.1.md). `experiments/Task6_BuildFieldFrames.py` consumes a JSON frame plan and that package's tools, stages new fields, extracts VTK corelines, merges gaps below 4h and filters merged arc lengths below 16h. Human-reviewed labels must be preserved and published through the existing field store.

The exact native adapter and frozen original computation used by the packaged observer are also preserved in `experiments/task6_field_tools/native/`. To compile this source snapshot, supply Eigen headers under `native/third_party/eigen/Eigen` (including the Eigen license files), then use its CMake project with a C++17 compiler and OpenMP. Alternatively compile the complete `tools/native` tree already distributed in the dataset package, which includes Eigen. DLL files are not part of this code commit. `source_audit.json` records the original source identities. Windows numerical parity was verified previously; Linux parity has not been verified.

`Export_Task6_FieldPackage_1_1.py` and `Publish_Task6_FieldPackage_1_1.py` preserve the historical migration implementation. They require the original local work directories and are not fresh-clone setup commands. Collaborators should start from the current shared field package, not rerun that migration over reviewed labels.

## Checks

```powershell
python -m pytest -q tests/test_task6_refinement_1_3.py tests/test_task6_sampling_2_1.py tests/test_task6_cpp_extraction_2_1.py tests/test_task6_core_length_1_4.py tests/test_task6_square_cylinder_2_5.py
```

These cover numerical refinement, strict continuous coreline-distance labels, VTK extraction/merging, physical arc-length filtering and Amira reading. Full-frame checks and h/64 reintegration checks also execute in the builder and publisher. Committing the code does not imply that the running full dataset build has completed.
