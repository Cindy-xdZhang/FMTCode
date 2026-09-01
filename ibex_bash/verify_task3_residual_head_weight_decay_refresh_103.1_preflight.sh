#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3rhwd1031p
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
cd "${TASK3103_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ResidualHeadWeightDecayRefresh_103_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python Search_Task3_LossOptimization_7_1.py \
  --config config/Verify_Task3_ResidualHeadWeightDecayRefresh_103.1.yaml \
  --mode preflight
