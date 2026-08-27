#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-1%2
#SBATCH -J FMTT23c41n
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH -p gpu4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
mkdir -p slurm_logs
DATASETS=(boeing747 smokeBuoyancy)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
python Build_Task23_FamilySearch_Confirmation.py \
  --group new2 --stage cache --dataset "$dataset"

