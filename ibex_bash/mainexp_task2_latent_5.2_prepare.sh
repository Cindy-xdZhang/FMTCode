#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2l52m
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK2L52_REPO_ROOT:-/home/zhanx0o/FMT_Task2_LatentBottleneck_5_2}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
python Prepare_Task2_LatentConfirmation_SourceManifest_5_2.py \
  --config config/mainExp_Task2_3D_5.2.yaml
