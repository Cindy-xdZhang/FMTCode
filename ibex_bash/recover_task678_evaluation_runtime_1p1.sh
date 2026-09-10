#!/bin/bash
set -euo pipefail
CHECKOUT="$1"
CONFIG="$2"
OUTPUT_ROOT="$3"
INDEX="$4"
EXPECTED_COMMIT="$5"
cd "$CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
ARGS=(--config "$CONFIG" --root "$OUTPUT_ROOT" --index "$INDEX")
python -m experiments.Recover_Task678_EvaluationRuntime_1_1 "${ARGS[@]}" --event RUNNING --exit-code 0
trap 'code=$?; if [ "$code" -ne 0 ]; then python -m experiments.Recover_Task678_EvaluationRuntime_1_1 "${ARGS[@]}" --event FAILED --exit-code "$code"; fi' EXIT
python -u -m experiments.Recover_Task678_EvaluationRuntime_1_1 "${ARGS[@]}"
python -m experiments.Recover_Task678_EvaluationRuntime_1_1 "${ARGS[@]}" --event COMPLETED --exit-code 0
