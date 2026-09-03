#!/bin/bash
# Repair launcher for the Task1 sub-chain of Verify_Task123_StrongBaselines_1.2.
#
# The first selection array (51277929) failed on tangaroa, f22raptor and
# boeing747 because the new geometric-statistics baseline produced NaN
# curvature on fully stagnant primitives (obstacle interiors); the dependent
# freeze/confirmation/merge/summary/audit jobs could therefore never start and
# were cancelled.  After the feature fix, ALL ten selection shards are
# recomputed so every dataset uses the same code; the seven shards produced by
# the pre-fix code are moved aside (never deleted) for the audit trail.
# Task2 (51277934) and Task3 (51277935) arrays are untouched and the new
# summary depends on them exactly as the original launcher did.
# History: the first run of this launcher (2026-09-03T15:25) submitted the
# selection/freeze/confirmation/merge jobs but its summary/audit sbatch calls
# failed with "Job dependency problem" because the Task3 array had already
# completed; those two jobs were then submitted by hand with the same actions
# and the manifest rows were filled in.  The loop below prevents a repeat.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
cd "$PROJECT_ROOT"

STRONG_ROOT="outputs/Verify_Task123_StrongBaselines_1.2"
MANIFEST="$PROJECT_ROOT/outputs/task123_replication_1p2_submission.tsv"
REPAIR_MARK="$STRONG_ROOT/task1/repair_submitted.txt"
CPU=ibex_bash/task123_replication_1p2_cpu.sh
TASK2_ARRAY="${TASK2_ARRAY:?original Task2 array job id required}"
TASK3_ARRAY="${TASK3_ARRAY:?original Task3 array job id required}"

if [[ ! -e "$MANIFEST" ]]; then
  echo "original submission manifest missing: $MANIFEST" >&2
  exit 2
fi
if [[ -e "$REPAIR_MARK" ]]; then
  echo "refusing duplicate repair submission: $REPAIR_MARK exists" >&2
  exit 2
fi
if [[ -e "$STRONG_ROOT/task1/frozen_baselines.json" ]]; then
  echo "refusing repair: Task1 baselines already frozen" >&2
  exit 2
fi

SHARDS="$STRONG_ROOT/task1/development_selection_shards"
if [[ -d "$SHARDS" ]]; then
  ASIDE="$STRONG_ROOT/task1/superseded_selection_shards_prefix_$(date +%Y%m%dT%H%M%S)"
  mv "$SHARDS" "$ASIDE"
  echo "moved pre-fix shards to $ASIDE"
fi

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
  job_id "$(sbatch "${args[@]}" "$CPU" "$action")"
}

record() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$MANIFEST"
}

R1="$(submit_cpu SB12r_T1Select "$STRONG_ROOT" "" strong_task1_select '0-9%10')"
record Verify_Task123_StrongBaselines_1.2 task1_selection_array_repair "$R1"
R2="$(submit_cpu SB12r_T1Freeze "$STRONG_ROOT" "afterok:$R1" strong_task1_freeze)"
record Verify_Task123_StrongBaselines_1.2 task1_freeze_repair "$R2"
R3="$(submit_cpu SB12r_T1Confirm "$STRONG_ROOT" "afterok:$R2" strong_task1_run '0-9%10')"
record Verify_Task123_StrongBaselines_1.2 task1_confirmation_array_repair "$R3"
R4="$(submit_cpu SB12r_T1Merge "$STRONG_ROOT" "afterok:$R3" strong_task1_merge)"
record Verify_Task123_StrongBaselines_1.2 task1_merge_repair "$R4"
# Slurm rejects afterok dependencies on jobs that already left the queue, so
# finished arrays are verified COMPLETED and dropped from the dependency list.
SUMMARY_DEP="afterok:$R4"
for upstream in "$TASK2_ARRAY" "$TASK3_ARRAY"; do
  if squeue -j "$upstream" -h -o %i 2>/dev/null | grep -q .; then
    SUMMARY_DEP="$SUMMARY_DEP:$upstream"
  else
    states="$(sacct -j "$upstream" -X -n -o State | tr -d ' ' | sort -u | paste -sd, -)"
    if [[ "$states" != "COMPLETED" ]]; then
      echo "upstream array $upstream is not uniformly COMPLETED: $states" >&2
      exit 3
    fi
  fi
done
R5="$(submit_cpu SB12r_Summary "$STRONG_ROOT" "$SUMMARY_DEP" strong_summarize)"
record Verify_Task123_StrongBaselines_1.2 summary_repair "$R5"
R6="$(submit_cpu SB12r_Audit "$STRONG_ROOT" "afterok:$R5" strong_audit)"
record Verify_Task123_StrongBaselines_1.2 independent_audit_repair "$R6"

date -Is > "$REPAIR_MARK"
tail -n 6 "$MANIFEST"
