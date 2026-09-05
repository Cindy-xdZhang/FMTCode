# Task4-b channel proxy-label protocol 1.2

## Frozen taxonomy

The channel axes are x=streamwise, y=spanwise and z=wall-normal. The task is a
vortex-only four-class target: 0 ordinary streamwise, 1 ordinary spanwise,
2 hairpin head and 3 hairpin limb. Non-vortex cubes use `ignore=-1` and do not
enter KMeans or neural classification. `VortexIds>0` supplies manual hairpin
support and instance identity only; head/limb remain deterministic proxy labels.

Ordinary cubes are divided by the absolute velocity–curl angle at 45 degrees.
Inside a positive `VortexId`, head additionally requires angle>45 degrees,
`omega_y_prime>0`, and y-prime vorticity magnitude at least as large as the x
and z components. Every remaining hairpin cube is limb, including legs, necks
and transition cubes.

## Candidate gate

Standard whole-domain IVD and channel-profile vorticity deviation were compared
using training instances only. The selected quantity is explicitly the latter,
`norm(omega-mean_xy(omega)(z))`, not standard IVD. The train-constrained winner
is target-grid percentile 80.5, threshold 5.388844. It retains 388,029 of
1,989,890 cubes (19.50%) before restoring 721 annotated hairpin cubes.

The training micro/VortexId-equal recall is 0.9500/0.9532. Validation is
0.9709/0.9718 and test is 0.9361/0.9375; validation/test were not constraints.
As a traceable rejection diagnostic, standard whole-domain flow-grid IVD p95 is
37.8991 and recalls only 0.01275 of annotated hairpin cubes.

The final counts are 120,996 ordinary streamwise, 251,043 ordinary spanwise,
4,275 hairpin head and 12,436 hairpin limb cubes.

## Leakage-free spatial split

All sample types use the same periodic-x slabs: train `[0.05,0.50)`, validation
`[0.57,0.73)` and test `[0.80,0.97)`. A hairpin enters a split only if the full
voxel support of its `VortexId` lies inside that slab. This gives 33/8/6 complete
instances; 26 buffer-crossing instances are excluded from downstream learning.

For the frozen primitive, the minimum cyclic x gap is 0.219911 and the support
radius upper bound is 0.107221, so the gap exceeds two radii (0.214442). The
cache records this executable non-overlap certificate.

## Physics quality checks

Head exists in 72/73 annotated instances and every evaluable head has positive
`omega_y_prime`. The joint x/z position order succeeds in 24/72. The requested
sign checks succeed for left leg `omega_x<0` in 58/64, right leg `omega_x>0` in
47/64, left neck `omega_z<0` in 62/64 and right neck `omega_z>0` in 59/64; all
four hold together in 39/64. These are QA metrics, not hard label constraints.

## Evidence

- `config/Verify_Task4B_ProxyLabels_1.2.yaml`
- `outputs/Verify_Task4B_ProxyLabels_1.2/channel_GTs/summary.json`
- `outputs/Verify_Task4B_ProxyLabels_1.2/channel_GTs/proxy_label_manifest.json`
- `outputs/Verify_Task4B_ProxyLabels_1.2/channel_GTs/task4b_proxy_labels.npz`
