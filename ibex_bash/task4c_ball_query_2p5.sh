#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD" PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
driver="$1"
phase="$2"
case "$driver" in
    fmt) module=experiments.Task4C_BallQuery_2_5 ;;
    baselines) module=experiments.Task4C_BallQueryBaselines_2_5 ;;
    *) exit 2 ;;
esac
finish_run() {
    code=$?
    trap - EXIT
    "$PY" -m "$module" runtime --index "${SLURM_ARRAY_TASK_ID:-0}" --runtime-phase "$phase" --state ENDED --exit-code "$code" || true
    exit "$code"
}
"$PY" -m "$module" runtime --index "${SLURM_ARRAY_TASK_ID:-0}" --runtime-phase "$phase" --state STARTED
trap finish_run EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$PY" -u -m "$module" "$phase" --index "${SLURM_ARRAY_TASK_ID:-0}"
