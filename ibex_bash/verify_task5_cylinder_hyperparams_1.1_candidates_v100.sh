#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-59%12
#SBATCH -J FMTT5c11v
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
candidate_index=$((SLURM_ARRAY_TASK_ID / 2))
dataset_index=$((SLURM_ARRAY_TASK_ID % 2))
DATASETS=(cylinder3d halfcylinderRe640)
dataset=${DATASETS[$dataset_index]}
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Search_Task5_CylinderHyperparams.py \
  --config config/Verify_Task5_CylinderHyperparams_1.1.yaml \
  --mode candidate --candidate-index "$candidate_index" \
  --dataset "$dataset" --seed 61
