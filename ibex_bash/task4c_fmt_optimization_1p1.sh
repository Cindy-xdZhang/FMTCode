#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD" PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
phase="$1"
index="${SLURM_ARRAY_TASK_ID:-0}"
printf 'phase=%s index=%s job=%s start=%s\n' "$phase" "$index" "$SLURM_JOB_ID" "$(date --iso-8601=seconds)"
if [[ "$phase" == "verify-source" ]]; then
    "$PY" -m unittest tests.test_task4c_fmt_optimization_1_1 tests.test_task4c_v2_newlabel_training_1_1
fi
"$PY" -u -m experiments.Task4C_FMTOptimization_1_1 "$phase" --index "$index"
printf 'phase=%s index=%s completed=%s\n' "$phase" "$index" "$(date --iso-8601=seconds)"
