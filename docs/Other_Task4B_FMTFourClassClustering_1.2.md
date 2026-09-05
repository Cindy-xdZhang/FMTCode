# Other_Task4B_FMTFourClassClustering_1.2

## Frozen diagnostic

This is the deliberately untuned `streamline -> frozen Task1 FMT -> KMeans(k=4)`
baseline under the corrected buffered spatial split. The cache contains 20,641
valid 7-line by 33-sample primitives and 161-dimensional FMT vectors. KMeans
and StandardScaler fit 13,386 train cubes; the cluster permutation is frozen on
3,925 validation cubes, and 3,330 stratified test cubes are evaluated once.

## Result

Test macro-F1 is 0.23787, balanced accuracy 0.31307, Adjusted Rand Index 0.03363
and Normalized Mutual Information 0.03919. Per-class F1 is 0.49814 ordinary
streamwise, 0.03575 ordinary spanwise, 0.00000 hairpin head and 0.41758 hairpin
limb. The test-oracle permutation equals the validation mapping, so mapping is
not the cause of the failure.

Across the six held-out `VortexId`s, equal-instance head/limb macro-F1 is
0.23749. The strict two-limb/one-head topology succeeds for proxy labels in
3/6 test hairpins but in 0/6 mapped KMeans hairpins; the mean soft score falls
from 0.67226 to 0.05206.

This is a clear failure of pure rotation-invariant, sorted-neighbour FMT for
coordinate-defined channel vortex types. It does not predict the outcome of a
supervised Raw+FMT model, because Raw geometry retains directional information.

The test ordinary classes are fixed stratified samples of 1,000 cubes each;
these metrics are not whole-field prevalence-weighted segmentation scores.

## Evidence

- `config/Other_Task4B_FMTFourClassClustering_1.2.yaml`
- `outputs/Other_Task4B_FMTFourClassClustering_1.2/channel_GTs/summary.json`
- `outputs/Other_Task4B_FMTFourClassClustering_1.2/channel_GTs/task4b_fmt_four_class_kmeans_result.npz`
- `outputs/Other_Task4B_FMTFourClassClustering_1.2/channel_GTs/task4b_fmt_four_class_kmeans_summary.{png,pdf,svg,tiff}`
