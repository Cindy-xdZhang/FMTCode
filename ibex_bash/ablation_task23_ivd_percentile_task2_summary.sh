#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2ivdps
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
repo_root=${FMT_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Run_Task2_IVDPercentile_Sweep.py \
  --config config/Ablation_Task23IVDPercentile_1.1.yaml --summarize
