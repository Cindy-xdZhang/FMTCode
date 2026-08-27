#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3f22out
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
python Search_Task3_FMTResidual_Stage2_3D.py \
  --config outputs/Verify_Task3_F22Hyperparams_1.1/focused_config.yaml \
  --mode outer
