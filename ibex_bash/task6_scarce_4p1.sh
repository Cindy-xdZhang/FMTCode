#!/usr/bin/env bash
set -euo pipefail
phase="$1"
config="$2"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
echo "EVENT START $(date -Is) node=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} phase=$phase"
trap 'code=$?; echo "EVENT EXIT $(date -Is) code=$code"; exit "$code"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$phase" == "preflight" ]]; then
    python -m unittest tests.test_task6_scarce_4_1 tests.test_task6_recovery_3d
else
    if [[ "$phase" == "search" || "$phase" == "final" ]]; then nvidia-smi --query-gpu=name,uuid --format=csv,noheader; fi
    python -m experiments.Task6_Scarce_4_1 "$phase" --config "$config" --index "${SLURM_ARRAY_TASK_ID:-0}"
fi
