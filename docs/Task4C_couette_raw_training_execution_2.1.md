# Couette raw-curl incremental dataset and training 2.1

2026-09-22: user confirmed that unnormalized curl agrees with their C++ visualization and must be used for this hairpin work. Channel/TBL must remain unchanged for now. The suggestion that their curl magnitude may be near one is a user hypothesis, not a verified result. User authorized dataset construction, git commit/push, and Ibex training.

## Frozen construction

Couette threshold -0.00125; all eight lambda2 vertices pass and mean stored oyf>0, plus pointwise candidate check. Reuse the original physically uniform 2,000,000 candidate positions and unchanged native field cache. Redo shortest-curve classification, quota Poisson selection of 200,000 centers, and all integration with raw curl. Existing Couette1.1 remains an arc-length historical version.

Integrator: classical RK4, dx/dtau=curl(v), fixed parameter step .0002. Each direction performs 35/50/65 actual updates, with seed additional. The products .007/.01/.013 are parameter intervals, NOT prescribed physical arc lengths. Boundary or out-of-bounds RK stage permits a shortened half only when completed updates >=ceil(.25*requested), explicitly confirmed by user. Other termination must complete the target. Three scales are prefixes of the maximum trace, each merged bidirectionally and resampled by actual arc length to 32 points. All three must be valid; no boundary wrapping. This reproduces the approved preview's integration semantics, not an audited C++ binary/derivative implementation.

Labels: shortest saved32 has >=17 GT points, and longest merged raw integration points have >=1 GT head and >=1 GT leg. Head is raw omega_y>0 and acute velocity-vorticity angle>45 degrees; all remaining GT is leg. No neighbor label vote or seed-head requirement. Labels were recomputed, not inherited.

Proposal short positives19,275; Poisson uses18,889 positive +181,111 negative centers (98% of available positives under inherited shortage rule), common radius .002705329613473178. Of200,000 candidates,23,941 fail integration validity. Final **176,059 seeds /528,177 curves**, **16,029 positive /160,030 negative**. All six GT instances have strict candidate-intersection seeds:0=2182,1=1656,2=4021,3=1061,4=2765,5=4597.

Independent audit: all shortest GT queried again;2,068,527 longest raw points of17,025 short-positive valid curves reclassified, all labels match;64 seeds three scales reintegrated;32 full-neighbor query/FPS replays; all strict candidates pass. Independent approved-preview loop exactly agrees for original seed184465; three synthetic tests check raw magnitude, boundary stopping,25% count rule. Dataset audit SHA256: `6ade347de505c8cf3371e7a330743100e1cf2d7e9da5e9ad19dfd5576d4cb4a4`.

## Loading and training

Complete pool retained. Balanced supervision indices select approx1:2 (here exact) without duplicating points or using predictions, while covering each split's GT-instance candidate intersection. Neighbor pool is the whole new Couette pool, initial3h with h=.00818015638904223, expand to kth neighbor if needed and use FPS inside the ball. Training uses6 neighbors + center.

| Fold | Couette train seeds | Couette test seeds | Test instances |
|---|---:|---:|---|
|0|12000 (4000 positive)|3000 (1000 positive)|0,4,5|
|1|12000 (4000 positive)|3000 (1000 positive)|2|
|2|12000 (4000 positive)|3000 (1000 positive)|1|
|3|7044 (2348 positive)|1761 (587 positive)|3|

Same original overlap grouping algorithm; new population balancing swaps singleton fold IDs1/2 relative to Couette1.1. Channel/TBL original five folds are unchanged. Couette has four independent groups; fold4 has no contribution and its original two-flow results are reused without duplicate training.

Append to frozen StrictCandidateTraining1.1 Channel/TBL rows (24000train/6000test per fold), preserving all old row order/labels/IDs/features/voxels. Combined counts: folds0–2 train36000/test9000; fold3 train31044/test7761. Instance isolation holds within each flow, no validation split. Source/data hashes must pass on Ibex before submission.

Models: original p35/h0 FMT, tangent jitter sigma.1/max.25 and dropout.35; original Conv3D8^3 dropout.15. Frozen architecture/normalization/feature and voxel implementations reused; training statistics recomputed on combined training rows. Single seed96611, fresh initialization, AdamW lr.001/weight decay.0001, batch128, clip5,50epochs. Training batches randomly shuffled every epoch. Existing target-risk class weighting to1/3 positives is retained on combined training labels. Per-seed evaluation uses majority of three length predictions, threshold.5; report aggregate and each flow. Final epoch is primary; maximum-test-F1 epoch is labeled test-selected, not an independent final evaluation. Four folds x two models; no added seeds or hyperparameter search. CPU/GPU encoding parity, independent voxel checks and GPU gradient pilots precede fits; all per-epoch test predictions audited after training.

## Execution state

Local dataset and independent audit complete. Git push, remote transfer verification, scheduler IDs, and actual GPU/training status will be recorded below once performed; their preparation is not completion.

Dataset: `outputs/mainExp_Task4C_CouetteDataset_2.1`. Training: `outputs/mainExp_Task4C_CouetteTraining_2.1`. Deploy: `experiments/Deploy_Task4C_CouetteTraining_2_1.py`. Gallery uses `Start_Task4C_BundleGallery_1_6.ps1` after export, showing actual new curves/labels rather than old-label preview.
