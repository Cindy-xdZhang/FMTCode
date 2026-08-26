#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-4%5
#SBATCH -J FMTT5r160cand
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
seeds=(70 71 72 73 74)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Verify_Task5_Re160FreshTimes.py \
  --config config/Verify_Task5_Re160FreshTimes_1.1.yaml \
  --mode train-candidate --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"

