# Task4-b channel proxy-label protocol 1.1

> **Superseded after independent audit.** This version randomly split complete
> hairpin `VortexId`s while ordinary cubes used x slabs. Cross-split primitives
> therefore read overlapping spatial regions. Its label artifact is retained as
> historical evidence, but it is not an eligible formal Task4-b evaluation.
> Use `Task4B_channel_proxy_label_protocol_1.2.md`.

## Frozen scope and taxonomy

This protocol applies only to the supplied steady channel-flow volume. Axes are
fixed as x=streamwise, y=spanwise, and z=wall-normal, with the lower wall at
`zmin`.

The first Task4-b experiment freezes a **vortex-only four-class** target:

| ID | Class | Definition |
|---:|---|---|
| 0 | ordinary streamwise | channel-profile-deviation candidate outside every positive `VortexId`, with absolute velocity–curl angle at most 45 degrees |
| 1 | ordinary spanwise | the same support, with angle greater than 45 degrees |
| 2 | hairpin head | positive `VortexId`, angle greater than 45 degrees, `omega_y_prime>0`, and `abs(omega_y_prime)>=max(abs(omega_x),abs(omega_z))` |
| 3 | hairpin limb | every remaining positive-`VortexId` cube; includes legs, necks and transition cubes |

Non-vortex cubes are assigned `ignore=-1` and do not enter KMeans evaluation or
the neural four-class loss. This resolves the initial ambiguity: adding
non-vortex while retaining ordinary streamwise, ordinary spanwise, head and
limb would be a five-class task. A future five-class experiment must receive a
new version and cannot be mixed with 1.1.

These targets are deterministic **proxy labels**, not manual anatomical truth.
`VortexIds>0` gives manual hairpin support and instance identity only.

## Vortex-candidate gate

Two definitions were compared using only the 44 training `VortexId` instances:

1. standard whole-domain IVD, `norm(omega-mean_xyz(omega))`;
2. channel-profile vorticity deviation,
   `norm(omega-mean_xy(omega)(z))`.

The second quantity is not standard IVD and must always retain its full name.
It removes the channel's z-dependent mean shear. Thresholds were percentiles of
the independent 131x98x155 target cube grid. Selection minimized candidate
volume subject to both training voxel-weighted hairpin recall and
VortexId-equal recall being at least 0.95. Cubes in the manual hairpin mask were
then restored by a hierarchy override, enforcing `hairpin subset vortex`.

The frozen selection is channel-profile deviation percentile 81.0,
threshold 5.477661. Before the override it retains 378,079/1,989,890 cubes
(19.00%). Recall is:

| Split | VortexIds | Micro recall | VortexId-equal recall | Minimum ID recall |
|---|---:|---:|---:|---:|
| train | 44 | 0.9527 | 0.9533 | 0.7843 |
| validation | 15 | 0.9505 | 0.9502 | 0.8776 |
| test | 14 | 0.9652 | 0.9671 | 0.9306 |

Standard whole-domain IVD is rejected for this label source: its flow-grid p95
threshold covers only about 1.27% of annotated hairpin cubes, while a 0.95
recall gate admits most of the target volume. This does not revise the frozen
IVD-p95 Task1/Task2/Task3 protocol; it is a separate Task4-b proxy experiment.

## Anatomical physics checks

The stored `oyf` quantity agrees numerically with
`omega_y-mean_xy(omega_y)(z)` and supplies the requested `omega_y_prime`.
The head hard rule labels 4,275 cubes, while limb labels 12,436; 72/73 manual
instances contain at least one head cube. The complete class counts are:

- ordinary streamwise: 117,483;
- ordinary spanwise: 244,651;
- hairpin head: 4,275;
- hairpin limb: 12,436.

The requested x/z position order and left/right vorticity signs are reported
per `VortexId` as quality checks, not used to tune or overwrite labels. This is
intentional: forcing the full position order would manufacture anatomy from a
rule that many voxelized instances do not satisfy. Limb quality checks split
temporary anchors by `abs(omega_x)>=abs(omega_z)` (leg) versus the complement
(neck); they do not create additional training classes.

## Split and evaluation boundary

- Hairpin cubes are split by complete `VortexId`.
- Ordinary cubes use non-overlapping buffered x intervals. The maximum frozen
  streamline half-length is smaller than each specified buffer.
- Feature normalization, KMeans, cluster-to-class mapping, neural checkpoint
  selection and any class weighting use train/validation only.
- Test labels never select a cluster permutation, epoch or hyperparameter.
- One steady volume supports only within-volume spatial/instance holdout. It
  cannot establish temporal or cross-flow generalization.

Primary multiclass metrics are macro-F1, per-class F1 and balanced accuracy.
Adjusted Rand Index and Normalized Mutual Information are additionally reported
for KMeans. Integer-label mean squared error is prohibited because the four
class IDs have no ordinal distance.

## Reproducible implementation

- `FMT_Utils/Task4B_ProxyLabels_3D.py`
- `experiments/Build_Task4B_ProxyLabels_1_1.py`
- `config/Verify_Task4B_ProxyLabels_1.1.yaml`
- `outputs/Verify_Task4B_ProxyLabels_1.1/channel_GTs/summary.json`
