#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3m72v
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${TASK72_REPO_ROOT:-/home/zhanx0o/FMT_Task3_AdaptiveTuned_7_2}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export TASK72_SOURCE_MODEL_ROOT="${TASK72_SOURCE_MODEL_ROOT:-/home/zhanx0o/FMT_Task3_AdaptivePortfolio_52_1}"
export TASK3_TUNED72_SOURCE_MANIFEST="${TASK3_TUNED72_SOURCE_MANIFEST:-/ibex/scratch/zhanx0o/FMT_Task3_AdaptiveTuned_7_2/source_staging_manifest.json}"
python Confirm_Task3_AdaptiveTuned_7_2.py \
  --config config/mainExp_Task3_3D_7.2.yaml \
  --mode source-preflight
