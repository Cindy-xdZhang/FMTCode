#!/usr/bin/env bash
set -euo pipefail
phase="$1"
config="$2"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
echo "EVENT START $(date -Is) node=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} phase=$phase"
if [[ "$phase" == "train" ]]; then nvidia-smi --query-gpu=name,uuid --format=csv,noheader; fi
trap 'code=$?; echo "EVENT EXIT $(date -Is) node=$(hostname) code=$code"; exit "$code"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$phase" == "preflight" ]]; then
    python -m unittest tests.test_task6_recovery_3d tests.test_task6_primitive_vae_2_1
fi
if [[ "$phase" == "train" || "$phase" == "audit-results" ]]; then
    python -m experiments.Task6_Reconstruction_3_1 "$phase" --config "$config" --index "${SLURM_ARRAY_TASK_ID:-0}"
else
    python -m experiments.Task6_PrimitiveVAE_2_1 "$phase" --config "$config" --index "${SLURM_ARRAY_TASK_ID:-0}"
fi
