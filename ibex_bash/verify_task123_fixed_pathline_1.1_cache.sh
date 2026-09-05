#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-13%14
#SBATCH -J FMTp123cache
#SBATCH -o outputs/Verify_Task123_FixedPathline_1.1/logs/%x.%A_%a.out
#SBATCH -e outputs/Verify_Task123_FixedPathline_1.1/logs/%x.%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

set -euo pipefail
repo_root=${FMT_FIXED_PATHLINE_REPO_ROOT:-/ibex/scratch/zhanx0o/FMT_Task123_Pathline_20260902/repo}
cd "$repo_root"
mkdir -p outputs/Verify_Task123_FixedPathline_1.1/logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python -m experiments.Prepare_Task123_FixedPathline \
  --config config/Verify_Task123_FixedPathline_1.1.yaml \
  --mode build-cache --job-index "$SLURM_ARRAY_TASK_ID" \
  --boeing-pack /ibex/scratch/zhanx0o/FMT_Task123_Pathline_20260902/source_packs/boeing747_fixed_pathline_windows.nc \
  --smoke-path /home/zhanx0o/DeepVortex/FLowDataFolder/SmokeBuoyancy80_239.nc \
  --staging-manifest tmp/Verify_Task123_FixedPathline_1.1/windows/source_staging_manifest.json
