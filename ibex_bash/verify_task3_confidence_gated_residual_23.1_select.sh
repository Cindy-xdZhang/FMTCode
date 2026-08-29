#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3g231s
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ConfidenceGatedResidual_23_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Search_Task3_LossOptimization_7_1.py \
  --config config/Verify_Task3_ConfidenceGatedResidual_23.1.yaml \
  --mode select
