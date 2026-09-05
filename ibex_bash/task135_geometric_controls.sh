#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail

mode=${1:?mode required}
repo_root=${FMT_GEOMETRY_ROOT:?set the isolated deployment directory}
cd "$repo_root"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
hostname
date -Iseconds
if [[ "$mode" == preflight ]]; then
  python tests/test_geometric_controls_3d.py
  python -m experiments.Run_Task135_GeometricControls --mode preflight
elif [[ "$mode" == run ]]; then
  python -m experiments.Run_Task135_GeometricControls --mode run --job-index "$SLURM_ARRAY_TASK_ID"
elif [[ "$mode" == summarize ]]; then
  python -m experiments.Run_Task135_GeometricControls --mode summarize
elif [[ "$mode" == audit ]]; then
  python -m experiments.Audit_Task135_GeometricControls
else
  exit 2
fi
