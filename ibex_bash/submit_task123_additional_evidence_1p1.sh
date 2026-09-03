#!/bin/bash
# Submit the three independent Task1--Task3 evidence chains.
# The launcher records every Slurm job ID before returning and refuses an
# accidental second submission into the same output roots.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
cd "$PROJECT_ROOT"

STRONG_ROOT="outputs/Verify_Task123_StrongBaselines_1.1"
ABLATION_ROOT="outputs/Ablation_Task123_FMTComponents_1.1"
NOISE_ROOT="outputs/Verify_Task123_NoiseRobustness_1.1"
MANIFEST="$PROJECT_ROOT/outputs/task123_additional_evidence_1p1_submission.tsv"

if [[ -e "$MANIFEST" ]]; then
  echo "refusing duplicate submission: $MANIFEST already exists" >&2
  exit 2
fi
mkdir -p "$STRONG_ROOT/logs" "$ABLATION_ROOT/logs" "$NOISE_ROOT/logs"
printf '%s\t%s\t%s\n' experiment stage job_id > "$MANIFEST"

job_id() {
  local raw="$1"
  printf '%s' "${raw%%;*}"
}

submit_cpu() {
  local name="$1" root="$2" dependency="$3" action="$4" array="${5:-}"
  local args=(--parsable --job-name="$name"
    --output="$root/logs/%x_%A_%a.out" --error="$root/logs/%x_%A_%a.err")
  [[ -n "$dependency" ]] && args+=(--dependency="$dependency")
  [[ -n "$array" ]] && args+=(--array="$array")
  job_id "$(sbatch "${args[@]}" ibex_bash/task123_additional_evidence_1p1_cpu.sh "$action")"
}

submit_gpu() {
  local name="$1" root="$2" dependency="$3" action="$4" array="$5"
  local args=(--parsable --job-name="$name" --array="$array"
    --output="$root/logs/%x_%A_%a.out" --error="$root/logs/%x_%A_%a.err")
  [[ -n "$dependency" ]] && args+=(--dependency="$dependency")
  job_id "$(sbatch "${args[@]}" ibex_bash/task123_additional_evidence_1p1_gpu.sh "$action")"
}

record() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$MANIFEST"
}

S1="$(submit_cpu SBase_T1Select "$STRONG_ROOT" "" strong_task1_select '0-9%10')"
record Verify_Task123_StrongBaselines_1.1 task1_selection_array "$S1"
S2="$(submit_cpu SBase_T1Freeze "$STRONG_ROOT" "afterok:$S1" strong_task1_freeze)"
record Verify_Task123_StrongBaselines_1.1 task1_freeze "$S2"
S3="$(submit_cpu SBase_T1Confirm "$STRONG_ROOT" "afterok:$S2" strong_task1_run '0-9%10')"
record Verify_Task123_StrongBaselines_1.1 task1_confirmation_array "$S3"
S4="$(submit_cpu SBase_T1Merge "$STRONG_ROOT" "afterok:$S3" strong_task1_merge)"
record Verify_Task123_StrongBaselines_1.1 task1_merge "$S4"
S5="$(submit_cpu SBase_T2Freeze "$STRONG_ROOT" "" strong_task2_freeze)"
record Verify_Task123_StrongBaselines_1.1 task2_freeze "$S5"
S6="$(submit_gpu SBase_T2 "$STRONG_ROOT" "afterok:$S5" strong_task2 '0-39%24')"
record Verify_Task123_StrongBaselines_1.1 task2_array "$S6"
S7="$(submit_gpu SBase_T3 "$STRONG_ROOT" "" strong_task3 '0-9%10')"
record Verify_Task123_StrongBaselines_1.1 task3_array "$S7"
S8="$(submit_cpu SBase_Summary "$STRONG_ROOT" "afterok:$S4:$S6:$S7" strong_summarize)"
record Verify_Task123_StrongBaselines_1.1 summary "$S8"
S9="$(submit_cpu SBase_Audit "$STRONG_ROOT" "afterok:$S8" strong_audit)"
record Verify_Task123_StrongBaselines_1.1 independent_audit "$S9"

A1="$(submit_gpu Abl_FMT_T1 "$ABLATION_ROOT" "" ablation_task1 '0-9%10')"
record Ablation_Task123_FMTComponents_1.1 task1_array "$A1"
A2="$(submit_cpu Abl_FMT_T1Merge "$ABLATION_ROOT" "afterok:$A1" ablation_task1_merge)"
record Ablation_Task123_FMTComponents_1.1 task1_merge "$A2"
A3="$(submit_gpu Abl_FMT_T2 "$ABLATION_ROOT" "" ablation_task2 '0-69%24')"
record Ablation_Task123_FMTComponents_1.1 task2_array "$A3"
A4="$(submit_gpu Abl_FMT_T3 "$ABLATION_ROOT" "" ablation_task3 '0-29%24')"
record Ablation_Task123_FMTComponents_1.1 task3_array "$A4"
A5="$(submit_cpu Abl_FMT_Summary "$ABLATION_ROOT" "afterok:$A2:$A3:$A4" ablation_summarize)"
record Ablation_Task123_FMTComponents_1.1 summary "$A5"
A6="$(submit_cpu Abl_FMT_Audit "$ABLATION_ROOT" "afterok:$A5" ablation_audit)"
record Ablation_Task123_FMTComponents_1.1 independent_audit "$A6"
A7="$(submit_cpu Abl_FMT_Cleanup "$ABLATION_ROOT" "afterok:$A6" ablation_cleanup)"
record Ablation_Task123_FMTComponents_1.1 checkpoint_cleanup "$A7"

N1="$(submit_gpu Noise_T1 "$NOISE_ROOT" "" noise_task1 '0-9%10')"
record Verify_Task123_NoiseRobustness_1.1 task1_array "$N1"
N2="$(submit_cpu Noise_T1Merge "$NOISE_ROOT" "afterok:$N1" noise_task1_merge)"
record Verify_Task123_NoiseRobustness_1.1 task1_merge "$N2"
N3="$(submit_gpu Noise_T2 "$NOISE_ROOT" "" noise_task2 '0-49%24')"
record Verify_Task123_NoiseRobustness_1.1 task2_array "$N3"
N4="$(submit_gpu Noise_T3 "$NOISE_ROOT" "" noise_task3 '0-49%24')"
record Verify_Task123_NoiseRobustness_1.1 task3_array "$N4"
N5="$(submit_cpu Noise_Summary "$NOISE_ROOT" "afterok:$N2:$N3:$N4" noise_summarize)"
record Verify_Task123_NoiseRobustness_1.1 summary "$N5"
N6="$(submit_cpu Noise_Audit "$NOISE_ROOT" "afterok:$N5" noise_audit)"
record Verify_Task123_NoiseRobustness_1.1 independent_audit "$N6"
N7="$(submit_cpu Noise_Cleanup "$NOISE_ROOT" "afterok:$N6" noise_cleanup)"
record Verify_Task123_NoiseRobustness_1.1 checkpoint_cleanup "$N7"

cat "$MANIFEST"
