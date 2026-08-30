#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2lb51p
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
cd "${TASK2LB51_REPO_ROOT:-/home/zhanx0o/FMT_Task2_LatentBottleneck_5_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export TASK2_SOURCE_ROOT=/home/zhanx0o/FMT_Task12_3D_20260823
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python Search_Task2_LatentBottleneck_5_1.py \
  --config config/Verify_Task2_LatentBottleneck_5.1.yaml \
  --mode preflight
