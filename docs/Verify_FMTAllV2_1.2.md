# Verify_FMTAllV2_1.2 — core-encoder replacement control

Registered 2026-09-07 after Task1 and part of Task2/Task5 in1.1 were evaluated. This is a methodological control on an already used benchmark, not an unseen confirmation or a model-selection experiment. No hyperparameter search is performed.

The1.1 comparison used the entire old task recipe versus the231-dimensional objective encoder alone. That valid standalone comparison also removed the old auxiliary neighbour blocks. It cannot isolate replacing `fmt_all` within the established task pipelines. Preserve1.1 and add this separate comparison:

| Task | Old arm | New arm | Capacity and reuse |
|---|---|---|---|
| Task1 | fmt_all+kin4 (189) | fmt_all_v2+kin4 (259) | same PCA8 and KMeans; reuse exactly the1.1 old-arm predictions |
| Task2 | fmt_all+kin4 (189) | fmt_all_v2+kin4 (259) | same frozen VAE settings; reuse1.1 Raw/old predictions, retrain new arm only |
| Task5 | fmt_all+gram2+kin6 (268) | fmt_all_v2+gram2+kin6 (338) | pad old to338, retrain both residual arms with identical338-wide architecture and shared frozen Raw backbone |

All datasets, seeds, data, labels and splits remain exactly those in1.1. Retain all1.1 outcomes; no result-based candidate or epoch changes. Task5's larger matched width is necessary to retain the old auxiliary blocks and equalize capacity; do not compare its new arm directly against the smaller1.1 old network as the primary contrast. Original source dependencies are taken from the frozen1.1 snapshot. Only the runner's declarative new-feature dispatch and this control config change.

The `fmt_all_v2` core remains231-dimensional and objective as tested. Retained `kin4/kin6` and the Raw branch are not covered by that exact numerical invariance certificate. Thus this control isolates pipeline performance under core replacement; it does not establish strict objectivity of the complete augmented feature/network. The two comparisons answer different questions and must be reported separately.

Reuse is explicit: aggregated Task1/Task2 old/Raw rows name their1.1 source; verify input file identities and labels. Task5 native outputs alone supply its matched-width comparison. No checkpoint download or long-term storage; delete new residual checkpoints after predictions and metrics are durable.

Task1/Task2 old-arm reuse preserves exact data and seeds but Task2 retraining may be allocated a different GPU model from its1.1 source run. Device records are retained for both; this is a seed/data paired comparison, not a claim of identical floating-point hardware for reused Task2 pairs. Task5's two retrained arms share the same allocated GPU and exactly matched parameter count.
