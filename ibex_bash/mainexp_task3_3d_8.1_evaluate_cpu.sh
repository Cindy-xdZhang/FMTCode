#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT3m81r
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G

set -euo pipefail
cd "${TASK81_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ExtendedTuned_8_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export TASK81_SOURCE_MODEL_ROOT="${TASK81_SOURCE_MODEL_ROOT:-/home/zhanx0o/FMT_Task3_ExtendedPortfolio_54_1}"
export TASK3_TUNED81_SOURCE_MANIFEST="${TASK3_TUNED81_SOURCE_MANIFEST:-/ibex/scratch/zhanx0o/FMT_Task3_ExtendedTuned_8_1/source_staging_manifest.json}"
export TASK3_FROZEN_RAW_DEPENDENCY_ROOT="$PWD/outputs/mainExp_Task3_3D_8.1/frozen_raw_dependencies"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python Freeze_Task3_RawDependencyClosure_8_1.py \
  --config config/mainExp_Task3_3D_8.1.yaml \
  --mode verify
python Confirm_Task3_ExtendedTuned_8_1.py \
  --config config/mainExp_Task3_3D_8.1.yaml \
  --mode dataset --job-index "$SLURM_ARRAY_TASK_ID"
