#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2lb51s
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK2LB51_REPO_ROOT:-/home/zhanx0o/FMT_Task2_LatentBottleneck_5_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export TASK2_SOURCE_ROOT=/home/zhanx0o/FMT_Task12_3D_20260823
python Search_Task2_LatentBottleneck_5_1.py \
  --config config/Verify_Task2_LatentBottleneck_5.1.yaml \
  --mode select
