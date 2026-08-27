#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT2ivd12
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint="a100|v100|rtx2080ti"
#SBATCH --mem=8G

set -euo pipefail
repo_root=${FMT_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
datasets=(channel cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa \
          deltaWing_resampled deltaWing_LBM f22raptor boeing747 smokeBuoyancy)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Run_Task2_IVDPercentile_Frozen_4_1.py \
  --config config/Ablation_Task23IVDPercentile_1.2.yaml \
  --mode dataset --dataset "${datasets[$SLURM_ARRAY_TASK_ID]}" --resume
