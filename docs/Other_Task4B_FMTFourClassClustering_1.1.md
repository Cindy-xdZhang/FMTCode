# Other_Task4B_FMTFourClassClustering_1.1

> **Invalidated for formal evaluation after independent audit.** The held-out
> `VortexId` identities were disjoint, but many cross-split primitives sampled
> overlapping physical space. The output remains an exploratory failure record;
> the corrected baseline is `Other_Task4B_FMTFourClassClustering_1.2`.

## Question and frozen method

This deliberately untuned baseline asks whether the Task1 161-dimensional FMT
representation alone separates the four proxy vortex types. It uses the
frozen six-frequency Gram/chirality encoder, sorted-neighbour pooling and
post-train-standardization neighbour weight 0.5. KMeans uses k=4,
`n_init=20`, and random state 7068.

All 16,711 manual hairpin cubes are retained. Ordinary streamwise/spanwise
cubes are sampled only from the buffered train/validation/test x intervals.
The resulting cache contains 26,711 primitives, each with 7 lines x 33 samples
x 3 coordinates and a 161-dimensional FMT vector. KMeans and StandardScaler
fit 15,872 training cubes. The cluster-to-class permutation is selected once
on 6,141 validation cubes by enumerating all 24 permutations; 4,698 test cubes
are evaluated without remapping.

## Test result

| Metric | Value |
|---|---:|
| macro-F1 | 0.3083 |
| balanced accuracy | 0.3852 |
| Adjusted Rand Index | 0.1063 |
| Normalized Mutual Information | 0.1464 |
| macro Intersection over Union | 0.2103 |

Per-class F1 is 0.5020 ordinary streamwise, 0.0000 ordinary spanwise,
0.1155 hairpin head and 0.6157 hairpin limb. On the 14 held-out hairpin
instances, the VortexId-equal head/limb macro-F1 is 0.3244 (median 0.3337).
The non-deployable best test permutation is identical to the frozen validation
mapping and therefore gives the same macro-F1; mapping choice is not the cause
of failure.

The strict two-limb/one-head voxel topology proxy falls from 24/73 for the proxy
labels to 9/73 for mapped KMeans. Mean soft topology score falls from 0.5271 to
0.1951.

## Interpretation

This is a clear failure of the frozen pure-FMT four-cluster baseline. It does
not mean that supervised Task4-b must fail. Standard FMT intentionally keeps
SO(3) rotation invariants and sorted neighbours, so it removes fixed channel
x/y/z identity while the four proxy labels are defined by those axes. KMeans
also assigns one approximately compact cluster per class even though each
semantic class may be multimodal.

The appropriate supervised comparison is therefore Raw, a larger Raw control,
FMT-only and Raw+FMT. FMT-only remains useful as a lower-bound diagnostic; the
Task3-like Raw+FMT model restores orientation-bearing pathline geometry.

## Artifacts

- `outputs/Other_Task4B_FMTFourClassClustering_1.1/channel_GTs/summary.json`
- `outputs/Other_Task4B_FMTFourClassClustering_1.1/channel_GTs/split_metrics.csv`
- `outputs/Other_Task4B_FMTFourClassClustering_1.1/channel_GTs/task4b_fmt_four_class_kmeans_result.npz`
- `outputs/Other_Task4B_FMTFourClassClustering_1.1/channel_GTs/task4b_fmt_four_class_kmeans_summary.{png,pdf,svg,tiff}`
