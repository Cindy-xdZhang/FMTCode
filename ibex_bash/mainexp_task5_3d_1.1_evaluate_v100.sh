#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT5m11eval
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=1:30:00
#SBATCH -p debug
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=v100
#SBATCH --mem=48G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Evaluate_Task5_Multiscale.py \
  --config config/mainExp_Task5_3D_1.1_evaluate.yaml
