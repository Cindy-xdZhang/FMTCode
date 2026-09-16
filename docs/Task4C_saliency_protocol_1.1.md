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

## Execution record

- Scientific preparation: 70a9afce. CPU and V100 gradient/forward/intervention checks passed.
- 51961968: V100 preflight completed; 51961969: stopped before training because the old identity uses `git_commit`, not `commit`. Evidence retained.
- Adapter correction: df325fbf; source paths are checked against their local repository equivalents. All 23 frozen source hashes match. 51962084 V100 preflight completed; 51962085 replay and attribution export completed on Tesla V100-SXM2-32GB / gpu609-09 in 14m21s.
- Viewer implementation began at 1b054e53; final source is `experiments/Build_Task4C_Saliency_1_1.py` and `experiments/templates/task4c_saliency.html`. Shared/per-bundle scales, individual patch scores, original/chord overlay, label-agreement filters, paging and both flows were checked in the browser with real exported data. Default gallery has eight bundles per page, with up to 64 selectable. Physical position and isolated bundle modes preserve physical proportions.

## Completed artifacts

`outputs/Other_Task4C_Saliency_1.1/independent_audit.json` independently verifies all 56 replay epochs and all 193,000/3,000/10,000 prediction arrays as exactly equal to the frozen seed. All 400 normalized geometries, physical coordinates and display selections match original data. There are 7,244 valid lines, 231,808 points and 50,708 local interventions. Channel has 149 label-agreeing and 51 label-disagreeing predictions; TBL has 135 and 65. This spatial display subset is not a new accuracy estimate.

The differentiable geometry forward and cached classifier probabilities differ by at most 2.861023e-6 because model batch sizes differ. Historical predictions are unchanged. Raw source arrays live in `package/`, while `viewer/index.html` loads 400 per-bundle data chunks, with corresponding JSON files. The viewer works offline when the folder is kept intact, or at `http://127.0.0.1:8766/viewer/index.html` while the loopback-only local server is running. No model weights were saved or downloaded.

Archive SHA256: `188b97220981b85f80add2133cd3de7e3a5b39c0fd513269e0df2251a8a860bc`. Scientific manifest SHA256: `c730cfbb14d2b7c1cd85e9c8e3c5f176f319a9d3241414bf63230c62441eea31`. Auditor: `experiments/Audit_Task4C_Saliency_1_1.py`. Method interpretation is recorded only in `docs/experiment_log.md`.
