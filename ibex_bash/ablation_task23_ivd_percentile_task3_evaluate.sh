#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-4%5
#SBATCH -J FMTT3ivde
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint="a100|v100|rtx2080ti"
#SBATCH --mem=48G

set -euo pipefail
repo_root=${FMT_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
tags=(p80 p85 p87p5 p90 p92p5)
tag=${tags[$SLURM_ARRAY_TASK_ID]}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Evaluate_Task3_MainTable.py \
  --config "outputs/Ablation_Task23IVDPercentile_1.1/generated_configs/task3_${tag}_evaluate.yaml"
