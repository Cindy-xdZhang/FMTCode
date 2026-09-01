#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3esdp821s
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK382_REPO_ROOT:-/home/zhanx0o/FMT_Task3_EarlyStoppingMinDeltaPortfolio_82_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Select_Task3_EarlyStoppingMinDeltaPortfolio_82_1.py \
  --config config/Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1.yaml \
  --mode select
