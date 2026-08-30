#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2l52p
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
cd "${TASK2L52_REPO_ROOT:-/home/zhanx0o/FMT_Task2_LatentBottleneck_5_2}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export TASK2_LATENT51_ROOT=/home/zhanx0o/FMT_Task2_LatentBottleneck_5_1
export TASK2_SOURCE_ROOT=/home/zhanx0o/FMT_Task12_3D_20260823
python -m pytest tests/test_mainexp_task2_3d_5_2.py \
  tests/test_task2_latent_bottleneck_5_1.py -q
python Confirm_Task2_LatentBottleneck_5_2.py \
  --config config/mainExp_Task2_3D_5.2.yaml --mode static-preflight
