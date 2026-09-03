#!/bin/bash
# Submit Verify_Task123_NoiseRobustness_1.2: corruption robustness of the
# stronger non-FMT baselines on the same corrupted realizations as 1.1.
# Task1 (KMeans, numpy features) runs on CPU; Task2 (VAE training) and Task3
# (frozen-model inference) on GPU. No checkpoints are written, so there is no
# cleanup stage. Every job ID is recorded before returning.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
cd "$PROJECT_ROOT"

ROOT="outputs/Verify_Task123_NoiseRobustness_1.2"
MANIFEST="$PROJECT_ROOT/outputs/task123_noise_1p2_submission.tsv"
CPU=ibex_bash/task123_noise_1p2_cpu.sh
GPU=ibex_bash/task123_noise_1p2_gpu.sh

if [[ -e "$MANIFEST" ]]; then
  echo "refusing duplicate submission: $MANIFEST already exists" >&2
  exit 2
fi
if [[ -e "$ROOT" ]]; then
  echo "refusing to submit into existing output root: $ROOT" >&2
  exit 2
fi
for required in outputs/Verify_Task123_NoiseRobustness_1.1/independent_audit.json \
                outputs/Verify_Task123_StrongBaselines_1.1/task1/frozen_baselines.json \
                outputs/Verify_Task123_StrongBaselines_1.1/task2/frozen_baselines.json \
                outputs/Verify_Task123_StrongBaselines_1.2/task1/frozen_baselines.json; do
  [[ -e "$required" ]] || { echo "missing prerequisite: $required" >&2; exit 2; }
done
mkdir -p "$ROOT/logs"
printf '%s\t%s\t%s\n' experiment stage job_id > "$MANIFEST"

job_id() {
  local raw="$1"
  printf '%s' "${raw%%;*}"
}

submit() {
  local script="$1" name="$2" dependency="$3" action="$4" array="${5:-}"
  local args=(--parsable --job-name="$name"
    --output="$ROOT/logs/%x_%A_%a.out" --error="$ROOT/logs/%x_%A_%a.err")
  [[ -n "$dependency" ]] && args+=(--dependency="$dependency")
  [[ -n "$array" ]] && args+=(--array="$array")
  job_id "$(sbatch "${args[@]}" "$script" "$action")"
}

record() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$MANIFEST"
}

N1="$(submit "$CPU" NS12_T1 "" noise_task1 '0-9%10')"
record Verify_Task123_NoiseRobustness_1.2 task1_array "$N1"
N2="$(submit "$CPU" NS12_T1Merge "afterok:$N1" noise_task1_merge)"
record Verify_Task123_NoiseRobustness_1.2 task1_merge "$N2"
N3="$(submit "$GPU" NS12_T2 "" noise_task2 '0-49%24')"
record Verify_Task123_NoiseRobustness_1.2 task2_array "$N3"
N4="$(submit "$GPU" NS12_T3 "" noise_task3 '0-9%10')"
record Verify_Task123_NoiseRobustness_1.2 task3_array "$N4"
N5="$(submit "$CPU" NS12_Summary "afterok:$N2:$N3:$N4" noise_summarize)"
record Verify_Task123_NoiseRobustness_1.2 summary "$N5"
N6="$(submit "$CPU" NS12_Audit "afterok:$N5" noise_audit)"
record Verify_Task123_NoiseRobustness_1.2 independent_audit "$N6"

cat "$MANIFEST"
