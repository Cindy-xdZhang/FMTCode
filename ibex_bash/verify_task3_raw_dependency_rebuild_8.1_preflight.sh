#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3raw81p
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd "${TASK81_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ExtendedTuned_8_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Repair_Task3_RawDependencies_8_1.py \
  --config config/Verify_Task3_RawDependencyRebuild_8.1.yaml \
  --mode preflight
