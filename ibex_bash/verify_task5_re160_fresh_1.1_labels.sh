#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT5r160lab
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Build_Task3_GlobalIVD_Labels.py \
  --config config/Verify_Task5_Re160FreshTimes_1.1_labels.yaml
