#!/usr/bin/env bash
set -euo pipefail
phase="$1"
config="$2"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.Task4C_FPSAugmentSearch_1_1 runtime --config "$config" --runtime-phase "$phase" --state STARTED
finish() {
    code=$?
    trap - EXIT
    python -m experiments.Task4C_FPSAugmentSearch_1_1 runtime --config "$config" --runtime-phase "$phase" --state ENDED --exit-code "$code" || true
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -m experiments.Task4C_FPSAugmentSearch_1_1 "$phase" --config "$config" --index "${SLURM_ARRAY_TASK_ID:-0}"
