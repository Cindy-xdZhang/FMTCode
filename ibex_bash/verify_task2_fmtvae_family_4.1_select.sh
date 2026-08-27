#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2f41sel
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
repo_root=${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Search_Task2_FMTVAE_3D.py \
  --config config/Verify_Task2_FMTVAEFamilySearch_4.1.yaml --mode select

