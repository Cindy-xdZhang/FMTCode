#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
phase="$1"
config="${2:-config/Ablation_Task4C_OriginalCenter_1.1.json}"
cat /proc/self/limits
finish_run() {
    code=$?
    trap - EXIT
    "$PY" -m experiments.Task4C_OriginalCenter_1_1 runtime --config "$config" --runtime-phase "$phase" --state ENDED --exit-code "$code" || true
    exit "$code"
}
"$PY" -m experiments.Task4C_OriginalCenter_1_1 runtime --config "$config" --runtime-phase "$phase" --state STARTED
trap finish_run EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$PY" -u -m experiments.Task4C_OriginalCenter_1_1 "$phase" --config "$config" --index "${SLURM_ARRAY_TASK_ID:-0}"
