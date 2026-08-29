#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT3c122c
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task3_CombinedConfirmation_12_2}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export TASK3_OPTIMIZATION_REPO_ROOT="${TASK3_OPTIMIZATION_REPO_ROOT:-/home/zhanx0o/FMT_Task3_LossOptimization_7_1}"
export TASK3_CONFIRMATION_SOURCE_MANIFEST="${TASK3_CONFIRMATION_SOURCE_MANIFEST:-/ibex/scratch/zhanx0o/FMT_Task3_Confirmation_SourcePacks_12_2/source_staging_manifest.json}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
python Confirm_Task3_CombinedOptimization_12_1.py \
  --config config/Confirm_Task3_CombinedOptimization_12.2.yaml \
  --mode cache --job-index "$SLURM_ARRAY_TASK_ID"
