#!/bin/bash
# Submit the preregistered noise-type sweep:
#   A = Verify_Task123_NoiseTypes_1.1        (FMT vs original Raw arms; 1.1 runner)
#   B = Verify_Task123_NoiseTypesStrong_1.1  (stronger baselines; 1.2 runner)
# B's summary merges A's audited rows, so it waits for A's independent audit.
# Every job ID is recorded before returning; duplicate submission is refused.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
cd "$PROJECT_ROOT"

ROOT_A="outputs/Verify_Task123_NoiseTypes_1.1"
ROOT_B="outputs/Verify_Task123_NoiseTypesStrong_1.1"
MANIFEST="$PROJECT_ROOT/outputs/task123_noisetypes_1p1_submission.tsv"
CPU=ibex_bash/task123_noisetypes_1p1_cpu.sh
GPU=ibex_bash/task123_noisetypes_1p1_gpu.sh

if [[ -e "$MANIFEST" ]]; then
  echo "refusing duplicate submission: $MANIFEST already exists" >&2
  exit 2
fi
for root in "$ROOT_A" "$ROOT_B"; do
  if [[ -e "$root" ]]; then
    echo "refusing to submit into existing output root: $root" >&2
    exit 2
  fi
done
for required in outputs/Verify_Task123_StrongBaselines_1.1/task1/frozen_baselines.json \
                outputs/Verify_Task123_StrongBaselines_1.1/task2/frozen_baselines.json \
                outputs/Verify_Task123_StrongBaselines_1.2/task1/frozen_baselines.json \
                outputs/Verify_Task3_UniformFMT_9.1/global_selection.json; do
  [[ -e "$required" ]] || { echo "missing prerequisite: $required" >&2; exit 2; }
done
mkdir -p "$ROOT_A/logs" "$ROOT_B/logs"
printf '%s\t%s\t%s\n' experiment stage job_id > "$MANIFEST"

job_id() {
  local raw="$1"
  printf '%s' "${raw%%;*}"
}

submit() {
  local script="$1" root="$2" name="$3" dependency="$4" action="$5" array="${6:-}"
  local args=(--parsable --job-name="$name"
    --output="$root/logs/%x_%A_%a.out" --error="$root/logs/%x_%A_%a.err")
  [[ -n "$dependency" ]] && args+=(--dependency="$dependency")
  [[ -n "$array" ]] && args+=(--array="$array")
  job_id "$(sbatch "${args[@]}" "$script" "$action")"
}

record() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$MANIFEST"
}

A1="$(submit "$GPU" "$ROOT_A" NT_A_T1 "" a_task1 '0-9%10')"
record Verify_Task123_NoiseTypes_1.1 task1_array "$A1"
A2="$(submit "$CPU" "$ROOT_A" NT_A_T1Merge "afterok:$A1" a_task1_merge)"
record Verify_Task123_NoiseTypes_1.1 task1_merge "$A2"
A3="$(submit "$GPU" "$ROOT_A" NT_A_T2 "" a_task2 '0-49%24')"
record Verify_Task123_NoiseTypes_1.1 task2_array "$A3"
A4="$(submit "$GPU" "$ROOT_A" NT_A_T3 "" a_task3 '0-49%24')"
record Verify_Task123_NoiseTypes_1.1 task3_array "$A4"
A5="$(submit "$CPU" "$ROOT_A" NT_A_Summary "afterok:$A2:$A3:$A4" a_summarize)"
record Verify_Task123_NoiseTypes_1.1 summary "$A5"
A6="$(submit "$CPU" "$ROOT_A" NT_A_Audit "afterok:$A5" a_audit)"
record Verify_Task123_NoiseTypes_1.1 independent_audit "$A6"
A7="$(submit "$CPU" "$ROOT_A" NT_A_Cleanup "afterok:$A6" a_cleanup)"
record Verify_Task123_NoiseTypes_1.1 checkpoint_cleanup "$A7"

B1="$(submit "$CPU" "$ROOT_B" NT_B_T1 "" b_task1 '0-9%10')"
record Verify_Task123_NoiseTypesStrong_1.1 task1_array "$B1"
B2="$(submit "$CPU" "$ROOT_B" NT_B_T1Merge "afterok:$B1" b_task1_merge)"
record Verify_Task123_NoiseTypesStrong_1.1 task1_merge "$B2"
B3="$(submit "$GPU" "$ROOT_B" NT_B_T2 "" b_task2 '0-49%24')"
record Verify_Task123_NoiseTypesStrong_1.1 task2_array "$B3"
B4="$(submit "$GPU" "$ROOT_B" NT_B_T3 "" b_task3 '0-9%10')"
record Verify_Task123_NoiseTypesStrong_1.1 task3_array "$B4"
B5="$(submit "$CPU" "$ROOT_B" NT_B_Summary "afterok:$B2:$B3:$B4:$A6" b_summarize)"
record Verify_Task123_NoiseTypesStrong_1.1 summary "$B5"
B6="$(submit "$CPU" "$ROOT_B" NT_B_Audit "afterok:$B5" b_audit)"
record Verify_Task123_NoiseTypesStrong_1.1 independent_audit "$B6"

cat "$MANIFEST"
