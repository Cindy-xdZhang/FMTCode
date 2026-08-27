#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT3m52
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
datasets=(channel cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa deltaWing_resampled deltaWing_LBM f22raptor boeing747 smokeBuoyancy)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
python Run_Task3_FMTResidual_Frozen_5_2.py \
  --config config/mainExp_Task3_3D_5.2.yaml --mode dataset \
  --dataset "${datasets[$SLURM_ARRAY_TASK_ID]}"
