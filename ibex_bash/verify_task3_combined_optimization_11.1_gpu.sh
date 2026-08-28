#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-159%24
#SBATCH -J FMTT3m111
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task3_LossOptimization_7_1}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
python Search_Task3_LossOptimization_7_1.py \
  --config config/Verify_Task3_CombinedOptimization_11.1.yaml \
  --mode candidate --job-index "$SLURM_ARRAY_TASK_ID"
