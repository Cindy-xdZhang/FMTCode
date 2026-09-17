# Task4-c FPS augmentation search 1.2: extend the shortlist to 20

On 2026-09-17 the user requested testing the top20 after the 264-candidate screening and partial top12 refinement. This extension promotes ranks13–20 from the **same completed screening ranking**. It does not repeat screening, modify the data or alter model/augmentation recipes. The operative interpretation is top20 three-seed validation refinement, then the selected recipe and original FPS control receive final testing. The optional clarification about testing every candidate is recorded separately; no broader final-test sweep is assumed.

The source is `Ablation_Task4C_FPSAugmentSearch_1.1`, scientific commit `9d05abc571bc7c45e5da8a8f9f3ee55db367d467`, config SHA256 `4f5b741fc131a28ee27b7dd6588426d7202579e0a527fd7b8d86d32adda10675`. Its source code, outputs and original top12 shortlist remain frozen. Source directory: `/ibex/user/zhanx0o/FMT_Task4C_FPSAugmentSearch_1p1_20260916/outputs/Ablation_Task4C_FPSAugmentSearch_1.1`.

## Unchanged scientific settings

Reuse all193,000 training,3,000 validation and10,000 test bundles, their labels and split membership. All original Fourier encoding, six-frequency input, FPS neighbors,32/48-point resampling, curvature weighting, training-only augmentation, normalization, optimizer, batch128, deterministic V100, early stopping and validation selection remain as specified in the1.1 protocol. The Python adapter calls the original training function bytecode and only changes output identity, dispatch and cross-version result verification. No checkpoint is retained.

The first12 candidates'36 refinement fits continue in their original checkout. The new version references these results with their **original** commit/config/seed identity. Only eight additional candidates receive new fits: ranks13–20 are `c146,c164,c198,c192,c187,c253,c200,c163`. Together with the original12, the full20 candidate order is frozen in the1.2 config.

All20 use refinement seeds96612,96613,96711, maximum500 epochs, patience50 and the same training budget. Select one by three-seed mean validation F1, then mean Average Precision, then candidate ID; report sample standard deviation and all seeds. Test is not loaded during refinement. Then run the selected candidate and the original FPS p35 control with seeds96721–96723, exactly as1.1 planned. No test-driven reranking.

## Scheduling and verification

The original pending selection51974957 is held while the extension is prepared. After the replacement chain is submitted successfully, cancel the original pending selection, six final fits51974958 and merge51974959; preserve their cancellation records. Original refinement51974956 remains active. The new selection depends on successful completion of **both** the original36 and additional24 fits. No completed/running fit is cancelled or overwritten.

This adds24 refinement fits; total refinement is60. The replacement chain has prepare1 + refine24 + select1 + final6 + merge1 =33 new Slurm processes,30 new fits (the six final fits replace the old pending six). Across both versions, the planned unique completed fits are264+60+6=330. Keep all cancelled processes in the registry.

The original refinement array is capped at12 future concurrent GPUs. The additional array receives at most12 concurrent GPUs, reduced further if more than12 old jobs are still running when submitted; old jobs are not preempted. Combined active capacity stays at most24 V100 GPUs. This scheduling cap does not alter fitting budgets.

Preparation independently checks all264 source validation predictions, exact complete ranking and top20 membership, old source-code hashes, original data file hashes and source identity before constructing references. Result verification accepts old identity only for the36 explicitly reused refinement fits, and current identity for new fits. New selection/merge verify every required prediction, source label and instance ID, full row coverage, metrics and selected epoch.

Implementation: `experiments/Task4C_FPSAugmentSearch_1_2.py`, `config/Ablation_Task4C_FPSAugmentSearch_1.2.json`, `ibex_bash/task4c_fps_augment_search_1p2.sh`. No1.1 implementation/config edit. Experimental conclusions remain in `docs/experiment_log.md`; all job IDs and events go to `docs/ibex_run_registry.md`.


<!-- task4c-fpsaug-final-20260917 -->
## Completion and configuration identity

Completed2026-09-17 11:47:29Saudi. Source1.1 scientific commit9d05abc571bc7c45e5da8a8f9f3ee55db367d467; new1.2 scientific commit21cf1fb75b556741ae7a6da8bd0f08104c1a0db5. The actual Git/Ibex LF config SHA256 is14a849f54553e51b16b5aec5e33fe2f274f2fe9e456b254854e32a7834883c78. Earlier deployment prose used the local Windows CRLF hash e908650742e5eedaa7f08d17ffe035c67407d7e661c43bb2604c53d947024974; canonical LF bytes and parsed JSON are exactly identical. This is a recorded hash-format correction, with no scientific revision. Independent final audit verifies330 fits,336 prediction files and337 executed processes;8 superseded unstarted processes remain cancelled records. Results are in experiment_log and paper_tables_tasks_3d; no further search follows automatically.
