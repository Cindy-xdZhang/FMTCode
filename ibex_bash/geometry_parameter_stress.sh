#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
mode=${1:?mode required}
cd "${FMT_STRESS_ROOT:?set isolated deployment root}"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
hostname
date -Iseconds
if [[ "$mode" == preflight ]]; then
  python tests/test_geometric_controls_3d.py
  python -m experiments.Run_GeometryParameterStress --mode preflight
elif [[ "$mode" == run ]]; then
  python -m experiments.Run_GeometryParameterStress --mode run --job-index "$SLURM_ARRAY_TASK_ID"
elif [[ "$mode" == summarize ]]; then
  python -m experiments.Run_GeometryParameterStress --mode summarize
elif [[ "$mode" == audit ]]; then
  python -m experiments.Audit_GeometryParameterStress
else
  exit 2
fi
