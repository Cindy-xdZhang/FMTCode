# Task6 preintegrated spherical bundles 1.1

The complete observed v−u fields (steady tornado: v), physical coordinates and current human coreline labels remain the canonical data. The user explicitly authorized this optional materialization on 2026-09-22, then reduced the count from 50,000 to **10,000 centers per frame** before bulk generation. No 50,000-center arrays were generated.

Each center lies strictly inside `IVD(v) > 0.8 * frame_maximum`. Candidate centers are sampled uniformly in physical volume using trilinear IVD, followed by rejection when any curve fails the unchanged numerical criteria. Nonuniform candidate cells are weighted by physical cell volume. The threshold applies only to the center. All 21 seeds must lie inside the available physical domain; no extrapolated velocity or duplicate/padded curves are manufactured.

The user confirmed 20 directions through the triangular face centers of an icosahedron, normalized to a sphere of radius h. These are not the 12 vertex directions. h is the minimum of the three mean axis spacings, `(maximum-minimum)/(grid_count-1)`. The global direction order is deterministic; streamline index 0 is the center, indices 1–20 are its neighbors. The sphere is not randomly rotated.

Integration is unchanged: normalized instantaneous velocity, adaptive Dormand–Prince 5(4), two directions with target arc length 0.25 each, maximum 20,000 attempts per direction, boundary/stagnation termination. Three maximum-step levels h/8, h/16, h/32 use absolute position tolerances h×1e−7, h×1e−8, h×1e−9. Save the h/16 result after combining both halves and resampling the combined polyline to 65 equal-arclength points. All lines must have length >h, finite coordinates, no numerical/max-attempt failure, and coarse/fine and fine/finest corresponding-point differences ≤h/20. A frame also gets 16 independently selected complete-bundle checks at h/64, tolerance h×1e−10. Raw adaptive vertices are not saved. The true seed need not coincide with stored point 32 after asymmetric truncation.

Storage is under `preintegrated/icosa_bundles_1.1`, leaving each canonical `frames/...` directory's eight-file contract unchanged. Per derived frame:

- `curves.npy`: float32 `[10000,21,65,3]`, physical xyz.
- `centers.npy`: float64 `[10000,3]`; reconstruct neighbor seeds using metadata unit directions and h.
- `labels.npy`: uint8 `[10000]`, center-to-continuous-coreline distance strictly <h.
- `half_lengths.npy`: float32 `[10000,21,2]`.
- `termination.npy`: int8 `[10000,21,2]`, 0 target reached, 1 boundary/stagnation.
- `refinement_errors.npy`: float32 `[10000,21]`.
- `metadata.json`: actual threshold, directions, field/grid/coreline hashes and numerical checks.
- `complete.json`: final checksums and completed marker. Without this marker a frame must not be consumed. `progress.json` is a resumable build checkpoint, not a completion marker.

For 101 frames, coordinates use exactly 16,543,800,000 bytes (16.5438 GB, about 15.408 GiB), excluding NPY headers. Other arrays add approximately 0.322 GB; no compression assumption is used. Full fields/coreline files are additional existing storage.

The reusable builder is `experiments/Build_Task6_IcosaBundles_1_1.py`; configuration is `config/mainExp_Task6_IcosaBundles_1.1.json`. Frame keys may be supplied via `--keys-file`; process count and threads per process are generic settings. Outputs are resumable. Early rejection only avoids computing remaining lines after a bundle already fails; an independent 640-center comparison across five flows found identical accepted bundles and byte-identical retained curves/lengths/errors. No model training, data upload or frontend modification is part of this build.

Audit evidence is under `outputs/Verify_Task6_IcosaBundles_1.1`. The source audit checked all101 frames and1344 manifest files at SHA256 `2fe772d566e8535edb9cf8c42a952851df004670f17d2914d7d9213621aeda8d`, with359 published manual operations and1853 corelines. Labels must always refer to the latest published corelines, not an old automatic baseline. Later edits invalidate only derived center labels, not streamline geometry; the standalone loader rejects stale labels.

## SquareCylinder boundary handling confirmed by the user

The user rejected the40% pilot and specified95%. Rechecking all16 frames showed that95% of the untrimmed maximum selects only the z=6 boundary grid plane. Raw Amira frame7 has an identically zero velocity plane at z=6, adjacent to nonzero velocity at z=5.87234; recomputed IVD is byte-identical to the saved array. The user explicitly authorized excluding this plane and the neighboring derivative-affected plane, and automatically rejecting any seed that cannot produce a complete valid bundle.

Formal SquareCylinder sampling uses the existing IVD on planes z indices0–45 (of48), z range[0,5.7446808511], and threshold95% of that valid-region maximum. The whole radius-h neighbor sphere must initially fit in that candidate domain. Complete field arrays, IVD arrays and human corelines are not modified. Integration still uses the original full observed field and unchanged numerical settings. All16 frames passed the revised pilot. Other flows keep80% of the full-frame maximum.

Formal builds use32 frame processes ×2 integration threads for the85 other frames, plus4×2 for SquareCylinder. This follows a real1/2/4/8-thread timing comparison with unchanged retained geometry. Checkpoints from the initial8×8 allocation were verified and resumed. Bulk results are still building; do not describe them as a completed101-frame bundle dataset until the final catalog says COMPLETE.

The optional reader and PARTIAL catalog are already in the shared package. Four full10,000-center Delta frames and four Square frames have completed in an intermediate check. The original source field/coreline hashes and both annotation draft hashes remained unchanged after documentation publication; canonical manifest SHA is now `6c0d28f95dc35724478041ff51eacd972a14959111c14733a05f8d4999439f43` because only the root README changed. OneDrive caused a transient access-denied error when replacing one Re160119 checkpoint; the atomic writer now retries transient reader locks, and the completion verifier will resume incomplete frames after the two main builders exit. Runtime PIDs are stored in the evidence directory rather than treated as permanent identifiers.
