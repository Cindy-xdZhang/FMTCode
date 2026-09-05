#!/bin/bash -l
#SBATCH --array=0-69%24
#SBATCH --job-name=FMTp123t1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o outputs/Verify_Task123_FixedPathline_1.1/logs/%x.%A_%a.out
#SBATCH -e outputs/Verify_Task123_FixedPathline_1.1/logs/%x.%A_%a.err

set -euo pipefail
repo_root=${FMT_FIXED_PATHLINE_REPO_ROOT:-/ibex/scratch/zhanx0o/FMT_Task123_Pathline_20260902/repo}
cd "$repo_root"
mkdir -p outputs/Verify_Task123_FixedPathline_1.1/logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
hostname
python -m experiments.Run_Task123_FixedPathline \
  --config config/Verify_Task123_FixedPathline_1.1.yaml \
  --mode task1 --job-index "$SLURM_ARRAY_TASK_ID"
