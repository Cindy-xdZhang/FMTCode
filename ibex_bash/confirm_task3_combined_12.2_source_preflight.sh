#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3c122v
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task3_CombinedConfirmation_12_2}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export TASK3_OPTIMIZATION_REPO_ROOT="${TASK3_OPTIMIZATION_REPO_ROOT:-/home/zhanx0o/FMT_Task3_LossOptimization_7_1}"
export TASK3_CONFIRMATION_SOURCE_MANIFEST="${TASK3_CONFIRMATION_SOURCE_MANIFEST:-/ibex/scratch/zhanx0o/FMT_Task3_Confirmation_SourcePacks_12_2/source_staging_manifest.json}"
python Confirm_Task3_CombinedOptimization_12_1.py \
  --config config/Confirm_Task3_CombinedOptimization_12.2.yaml \
  --mode source-preflight
