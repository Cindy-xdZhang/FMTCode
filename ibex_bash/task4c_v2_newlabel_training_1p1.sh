#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD" PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
phase="$1"
index="${SLURM_ARRAY_TASK_ID:-0}"
finish_run() {
    code=$?
    trap - EXIT
    "$PY" -m experiments.Task4C_V2NewLabelTraining_1_1 runtime --runtime-phase "$phase" --state ENDED --exit-code "$code" --index "$index" || true
    exit "$code"
}
"$PY" -m experiments.Task4C_V2NewLabelTraining_1_1 runtime --runtime-phase "$phase" --state STARTED --index "$index"
trap finish_run EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$phase" == "verify-source" ]]; then
    "$PY" -m unittest tests.test_task4c_v2_newlabel_training_1_1 tests.test_task4c_ball_query_3_1 tests.test_task4c_v3_training_3_1
fi
"$PY" -u -m experiments.Task4C_V2NewLabelTraining_1_1 "$phase" --index "$index"
