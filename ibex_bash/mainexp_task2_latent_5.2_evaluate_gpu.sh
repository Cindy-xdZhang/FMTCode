#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT2l52r
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G

set -euo pipefail
cd "${TASK2L52_REPO_ROOT:-/home/zhanx0o/FMT_Task2_LatentBottleneck_5_2}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export TASK2_LATENT51_ROOT=/home/zhanx0o/FMT_Task2_LatentBottleneck_5_1
export TASK2_SOURCE_ROOT=/home/zhanx0o/FMT_Task12_3D_20260823
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
datasets=(channel cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa deltaWing_resampled deltaWing_LBM f22raptor boeing747 smokeBuoyancy)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
python Confirm_Task2_LatentBottleneck_5_2.py \
  --config config/mainExp_Task2_3D_5.2.yaml --mode dataset \
  --dataset "${datasets[$SLURM_ARRAY_TASK_ID]}"
