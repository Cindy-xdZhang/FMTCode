#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3pf491
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK491_REPO_ROOT:-/home/zhanx0o/FMT_Task3_FinalPortfolio_49_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Select_Task3_FinalPortfolio_49_1.py \
  --config config/Verify_Task3_FinalPortfolio_49.1.yaml \
  --mode select
