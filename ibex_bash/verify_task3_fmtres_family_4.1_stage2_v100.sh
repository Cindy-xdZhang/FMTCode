#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-299%10
#SBATCH -J FMTT3f41s2
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
python Search_Task3_FMTResidual_Stage2_3D.py \
  --config config/Verify_Task3_FMTResidualFamilySearch_4.1.yaml \
  --mode candidate --job-index "$SLURM_ARRAY_TASK_ID"
