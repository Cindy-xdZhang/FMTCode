#!/usr/bin/env bash
set -euo pipefail
phase="$1"
config="$2"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
python Submit_Task6_PNNTuning_1_2.py runtime --config "$config" --phase "$phase" --state STARTED
finish() {
    code=$?
    trap - EXIT
    python Submit_Task6_PNNTuning_1_2.py runtime --config "$config" --phase "$phase" --state ENDED --exit-code "$code" || true
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$phase" == "preflight" ]]; then
    python -m unittest tests.test_task6_pnn_trans_1_1 tests.test_task6_pnn_tuning_1_2 tests.test_task6_pnn_tuning_pipeline_1_2
elif [[ "$phase" == "advance" ]]; then
    python Submit_Task6_PNNTuning_1_2.py final --config "$config"
else
    python -m experiments.Task6_PNNTuning_1_2 "$phase" --config "$config" --index "${SLURM_ARRAY_TASK_ID:-0}"
fi
