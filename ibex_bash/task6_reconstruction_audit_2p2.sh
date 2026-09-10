#!/usr/bin/env bash
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
echo "EVENT START $(date -Is) node=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} device=CPU"
trap 'code=$?; echo "EVENT EXIT $(date -Is) node=$(hostname) code=$code"; exit "$code"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -m unittest tests.test_task6_recovery_3d
python -m experiments.Verify_Task6_ReconstructionAudit_2_2 --config "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
