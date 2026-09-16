# Other_Task4C_Saliency_1.1

User request (2026-09-16): visualize many predicted-hairpin bundles and identify influential curve portions, before the proposed ablation studies. This is an interpretation feature; no new classifier or model selection.

## Frozen classifier and replay

Use BottomDensity 1.2 p35/n0_k06, 141 features and 76,738 parameters, seed 96611. The published three-seed F1 is 0.888404 ± 0.011497; this seed's existing test F1 is 0.878796480272495. They must not be conflated. Data remains 193,000/3,000/10,000, source scientific commit a7643890d22c7faf63c1f716bb6222ba5271c5ff. The source config hash is fixed in the new config.

Weights were intentionally not saved. Replay the original training function's code object and initialization, permutations, optimizer and validation schedule on V100 through the already selected epoch 56 (the original stopped at 106). Epoch 56 is read from the frozen result before replay, not selected using new outputs. Capture the model in memory with a factory wrapper. Compare all original train/validation/test probabilities, labels and decisions; do not export saliency if reproduction fails. Never save or download weights.

## Geometry maps

- Score: hairpin logit minus non-hairpin logit. This avoids interpreting saturation of the final probability as absence of influence.
- Point sensitivity: spatial norm of the exact score gradient with respect to the original normalized geometry, through whole-bundle normalization, fixed Fourier encoder, training-only feature standardization and trained classifier. This is sensitivity, not signed support or an additive causal allocation.
- SmoothGrad: average 16 gradient vectors under independent point noise with standard deviation 0.002 in input radius units, then spatial norm. Same frozen neighbor graph; invalid padding never perturbed. Keep raw gradient available too. No further smoothing of exported values.
- Local shape intervention: on one line at a time replace each specified 7/8-edge segment with its endpoint chord, retaining endpoints, 32 points and all other lines. Recompute all continuous encoder operations including normalization; preserve seed-neighbor identities. Export original minus changed logit margin and probability. Positive means the removed curvature supported hairpin under this particular intervention, negative means it suppressed it. The chord is a diagnostic geometry, not a newly integrated physical vortex line. Changes include arc length/tangent and possible global normalization effects; do not call this an isolated curvature derivative.
- For point display only, average changes of overlapping windows over their modified interior points. Unmodified endpoints show neutral color. Keep the exact window scores and allow inspecting the strongest window, with a chord overlay. Overlapping values are not additive explanations.

## Examples and viewer

Select 200 examples per flow among reference test predictions >=0.5 by frozen spatial farthest-point display ordering, seed 91641 plus flow index. Do not use GT or saliency to select; retain both true and false positives. User can filter by flow, GT agreement, inspect individual bundles or many bundles, switch attribution modes, choose common or per-bundle color range, and inspect exact sample IDs, probabilities and shape-intervention scores. Physical geometry is recovered with the original centroid and radius. A gallery may place normalized shapes on a display grid, explicitly distinct from physical positions. Scalar fields and source metadata are downloadable; no accuracy claims from the display subset.

## Validation

Check differentiable forward against original encoder, cached features and original selected predictions; padding gradients zero; input gradients against finite differences on nondegenerate geometry; straightening endpoints and other lines unchanged. Check that zero classifier weights give zero sensitivity and shape changes. Inspect model randomization as a diagnostic rather than assuming all attractive maps are valid. Record all replay and export processes in the Ibex registry, including failures. Browser interaction and real-data export must complete before declaring delivery complete.

References: [SmoothGrad](https://arxiv.org/abs/1706.03825), [Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). Method-level conclusions belong only in `docs/experiment_log.md`.
