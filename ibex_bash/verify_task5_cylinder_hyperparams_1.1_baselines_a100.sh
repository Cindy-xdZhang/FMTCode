#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-1%2
#SBATCH -J FMTT5c11base
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=a100
#SBATCH --mem=48G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
CONFIGS=(
  config/Verify_Task5_CylinderHyperparams_1.1_baseline_re160.yaml
  config/Verify_Task5_CylinderHyperparams_1.1_baseline_re640.yaml
)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Verify_Task3_FMTClassifier.py --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
