#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-559%10
#SBATCH -J FMTT2f41fmt
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH -p gpu4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=48G

set -euo pipefail
repo_root=${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Search_Task2_FMTVAE_3D.py \
  --config config/Verify_Task2_FMTVAEFamilySearch_4.1.yaml \
  --mode fmt --job-index "$SLURM_ARRAY_TASK_ID"

