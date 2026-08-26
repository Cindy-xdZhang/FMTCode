#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT5m11merge
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH -p debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Merge_Task3_ResidualShards.py \
  --config config/mainExp_Task5_3D_1.1_evaluate.yaml \
  --root outputs/mainExp_Task5_3D_1.1
