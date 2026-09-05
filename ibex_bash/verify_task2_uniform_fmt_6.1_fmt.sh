#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-559%24
#SBATCH -J FMTT2u61f
#SBATCH -o outputs/Verify_Task2_UniformFMT_6.1/logs/%x.%A_%a.out
#SBATCH -e outputs/Verify_Task2_UniformFMT_6.1/logs/%x.%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G

set -euo pipefail
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p outputs/Verify_Task2_UniformFMT_6.1/logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python -m experiments.Search_Task2_FMTVAE_3D \
  --config config/Verify_Task2_UniformFMT_6.1.yaml \
  --mode fmt --job-index "$SLURM_ARRAY_TASK_ID"
