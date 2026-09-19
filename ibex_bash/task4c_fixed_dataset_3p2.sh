#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
phase="$1"
config="${2:-config/mainExp_Task4C_FixedDataset_3.2.json}"
finish_run() {
    code=$?
    trap - EXIT
    "$PY" -m experiments.Task4C_FixedDataset_3_2 runtime --config "$config" --runtime-phase "$phase" --state ENDED --exit-code "$code" || true
    exit "$code"
}
"$PY" -m experiments.Task4C_FixedDataset_3_2 runtime --config "$config" --runtime-phase "$phase" --state STARTED
trap finish_run EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$phase" == "verify-inputs" ]]; then
    "$PY" -m unittest tests.test_task4c_fixed_dataset_3_2 tests.test_task4c_fixed_dataset_3_1 tests.test_task4c_fixed_dataset_2_1
fi
"$PY" -u -m experiments.Task4C_FixedDataset_3_2 "$phase" --config "$config" --index "${SLURM_ARRAY_TASK_ID:-0}"
