#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT2f41out
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=6:00:00
#SBATCH -p gpu4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=48G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
mkdir -p slurm_logs
DATASETS=(channel cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa deltaWing_resampled deltaWing_LBM f22raptor boeing747 smokeBuoyancy)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
python Search_Task2_FMTVAE_Stage2_3D.py \
  --config config/Verify_Task2_FMTVAEFamilySearch_4.1.yaml \
  --mode outer --dataset "$dataset"

