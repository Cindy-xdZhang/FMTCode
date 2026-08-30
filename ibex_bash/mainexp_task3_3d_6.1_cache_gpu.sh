#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT3m61c
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

set -euo pipefail
cd "${TASK61_REPO_ROOT:-/home/zhanx0o/FMT_Task3_AnchoredFeature_6_1}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export TASK46_SOURCE_MODEL_ROOT="${TASK46_SOURCE_MODEL_ROOT:-/home/zhanx0o/FMT_Task3_AnchoredFeatureDecomposition_22_1}"
export TASK3_ANCHORED6_SOURCE_MANIFEST="${TASK3_ANCHORED6_SOURCE_MANIFEST:-/ibex/scratch/zhanx0o/FMT_Task3_AnchoredFeature_6_1/source_staging_manifest.json}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
python Confirm_Task3_AnchoredFeature_6_1.py \
  --config config/mainExp_Task3_3D_6.1.yaml \
  --mode cache --job-index "$SLURM_ARRAY_TASK_ID"
