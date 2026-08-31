#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-3%4
#SBATCH -J FMTT3raw81
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH -p gpu,gpu1,gpu4,gpu24,gpu72
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

set -euo pipefail
cd "${TASK81_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ExtendedTuned_8_1}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
python Repair_Task3_RawDependencies_8_1.py \
  --config config/Verify_Task3_RawDependencyRebuild_8.1.yaml \
  --mode train --job-index "$SLURM_ARRAY_TASK_ID"
