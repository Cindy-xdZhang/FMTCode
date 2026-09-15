#!/usr/bin/env bash
set -euo pipefail
adapter="$1"
config="$2"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.FMTv8_NormFrequency_3_1 runtime --config "$config" --runtime-phase task4_single --state STARTED
finish() {
    code=$?
    trap - EXIT
    python -m experiments.FMTv8_NormFrequency_3_1 runtime --config "$config" --runtime-phase task4_single --state ENDED --exit-code "$code" || true
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python "$adapter" --config "$config" --index "${SLURM_ARRAY_TASK_ID:?}"
