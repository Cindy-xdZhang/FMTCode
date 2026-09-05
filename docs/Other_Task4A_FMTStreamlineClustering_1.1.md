# Other_Task4A_FMTStreamlineClustering_1.1

## Status and scope

This is an exploratory, label-free Task4-a attempt. `channel_GTs.vtk` supplies
positive vortex-instance identifiers (`VortexIds`) and a non-vortex background,
but it does not supply streamwise/spanwise class labels. The experiment therefore
does **not** estimate supervised classification accuracy and does not replace the
formal Task4 class-label protocol required by `docs/research_tasks_and_protocol.md`.

## Frozen data and class semantics

- Vortex support: voxel centers whose cell-associated `VortexIds > 0`.
- Background: `VortexIds <= 0`; excluded from streamline clustering and kept as 0
  in the output segmentation.
- Instance identity: the positive integer `VortexIds`; it is never passed to FMT
  or KMeans as a feature.
- Channel axes, verified from `channel.vtk`: x is streamwise, y is spanwise, z is
  wall-normal. The x component has mean absolute velocity 0.8593, compared with
  0.0381 for y and 0.0310 for z.
- Predicted class 1 is named streamwise only after clustering: it is the KMeans
  cluster with the larger median absolute local-vorticity alignment with x.
  Predicted class 2 is the other cluster. Vorticity is the curl of velocity and
  supplies the physically relevant vortex-axis direction. This naming diagnostic
  does not alter the KMeans partition and is not included in the FMT features.

An earlier implementation draft used center-streamline endpoint direction for
this post-hoc name. That measures advection direction: both channel clusters were
more than 0.996 aligned with x and therefore could not define streamwise versus
spanwise vortex axes. Version 1.1 corrects the naming variable to vorticity before
the result is accepted; endpoint alignment remains an exported failure diagnostic.

## Frozen primitive and FMT recipe

Every positive cube center seeds a seven-line cross primitive: center and x/y/z
plus/minus offsets. The steady streamline geometry is integrated in both directions
with fourth-order Runge-Kutta integration of unit velocity. x and y are periodic;
z remains bounded by the channel walls. The primitive contains 16 backward samples,
the seed, and 16 forward samples (33 positions per line). Step and neighbor offset
are deterministic functions of the independent GT voxel size and are recorded in
the run summary in physical units.

The FMT recipe is copied from the frozen Task1 3D protocol: six frequencies, Gram
rotation invariants with chirality, sorted neighbor pooling, and neighbor weight
0.5 applied after `StandardScaler`. Only these FMT features enter two-cluster
KMeans. FMT is rotation invariant, so success is not guaranteed: a failure may
show that absolute channel orientation is required rather than that integration
or visualization is incorrect.

To prevent large hairpins from dominating KMeans, at most 128 valid seeds per
`VortexId` enter `fit`; every valid seed is then predicted. Invalid primitives are
not used for scaling or clustering. For the requested complete cube image, an
invalid seed receives the nearest valid prediction from the same `VortexId`; the
number of such filled voxels is reported.

## Label-free topology proxy

For each `VortexId`, each predicted class is decomposed with 26-neighbor voxel
connectivity. A component is dominant when it contains at least
`max(3, ceil(0.05 * vortex_voxels))` cubes.

- Hard local success: exactly two dominant streamwise components and exactly one
  dominant spanwise component. The mean is reported across all instances.
- Single-input-component success: the same quantity restricted to instances whose
  input GT mask is itself one 26-connected component. This separates voxelization
  discontinuities from predicted splitting errors.
- Soft topology score:

  `dominant_mass_fraction * exp(-(|n_streamwise-2| + |n_spanwise-1|))`

  where `dominant_mass_fraction` is the fraction of all cubes covered by the two
  largest streamwise components and the largest spanwise component. The score is
  in [0, 1]. It is an evaluation/validation proxy, not a differentiable training
  loss and not a substitute for manual class labels.

No threshold or feature choice in version 1.1 is selected from this proxy. Any
future optimization must create a new experiment version and split complete
`VortexId` instances between model selection and confirmation.

## Frozen 1.1 result

The real channel run retained all 73 positive instance IDs and integrated all
16,711 seven-line primitives successfully. Global FMT KMeans produced 13,999
voxels in raw cluster 0 and 2,712 in raw cluster 1. Post-hoc vorticity naming
mapped raw cluster 1 to streamwise and raw cluster 0 to spanwise.

- Strict volume-connectivity success: 4/73 (5.48%).
- Strict success on the 62 input-single-component instances: 3/62 (4.84%).
- Mean/median soft topology score: 0.2662 / 0.1353.
- Failure distribution: 33/73 instances have no streamwise component reaching
  the frozen 5% dominance threshold; the most common topology is therefore
  `(0 streamwise, 1 spanwise)`.
- Vorticity naming separation is weak: the streamwise raw cluster has median
  absolute x-vorticity alignment 0.592, versus 0.544 for the spanwise raw
  cluster. The corresponding y alignments are 0.449 and 0.528.

This is evidence against the proposed **global pure-FMT KMeans** recipe, not
evidence that the GT lacks hairpins. The global clusters separate local states
and/or instances more strongly than they recover a stable head-versus-legs split.

A physical sanity check that directly labels every voxel by whether `|omega_x|`
or `|omega_y|` is larger reaches 13/73 strict successes and a mean soft score of
0.3576. Its most common topologies include `(2,2)`, `(1,1)`, and `(1,2)`. Thus the
strict volume `2+1` rule is itself sensitive to neck transitions and fragmented
voxel classes. It should remain a strict diagnostic alongside a centerline- or
skeleton-based anatomical score, not serve as the sole model-selection metric.

An exploratory per-instance KMeans calculation (not accepted into 1.1) reaches
10/73 strict successes and mean soft score 0.3570. Because this choice was made
after inspecting 1.1, it requires a new version if pursued.

## Figure contract

- Intended conclusion: inspect whether pure Task1 FMT features recover a spatial
  two-class partition compatible with the expected hairpin 2+1 topology.
- Evidence: a two-color 3D cube segmentation, cluster-axis naming diagnostics,
  and per-instance hard/soft topology results.
- Archetype: image plate plus quantitative diagnostics.
- Source integrity: the input instance mask and run summary must retain all 73
  positive IDs at the frozen preview resolution.
- Failure boundary: visual plausibility and the topology proxy cannot establish
  physical class correctness without streamwise/spanwise annotations.
