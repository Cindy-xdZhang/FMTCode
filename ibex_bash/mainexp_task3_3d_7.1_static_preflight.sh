#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3m71p
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK71_REPO_ROOT:-/home/zhanx0o/FMT_Task3_FinalTuned_7_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export TASK71_SOURCE_MODEL_ROOT="${TASK71_SOURCE_MODEL_ROOT:-/home/zhanx0o/FMT_Task3_FinalPortfolio_49_1}"
export TASK3_TUNED7_SOURCE_MANIFEST="${TASK3_TUNED7_SOURCE_MANIFEST:-/ibex/scratch/zhanx0o/FMT_Task3_FinalTuned_7_1/source_staging_manifest.json}"
python Prepare_Task3_FinalTuned_SourceManifest_7_1.py \
  --config config/mainExp_Task3_3D_7.1.yaml
python Confirm_Task3_FinalTuned_7_1.py \
  --config config/mainExp_Task3_3D_7.1.yaml \
  --mode static-preflight
