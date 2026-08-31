#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3m81d
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd "${TASK81_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ExtendedTuned_8_1}"
mkdir -p slurm_logs
python Freeze_Task3_RawDependencyClosure_8_1.py \
  --config config/mainExp_Task3_3D_8.1.yaml \
  --mode freeze
python Freeze_Task3_RawDependencyClosure_8_1.py \
  --config config/mainExp_Task3_3D_8.1.yaml \
  --mode verify
