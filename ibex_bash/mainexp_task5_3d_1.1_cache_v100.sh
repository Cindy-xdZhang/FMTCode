#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT5m11cache
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH -p gpu4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# These five source fields already exist on Ibex.  The remaining five caches
# are built from the user's local originals and copied before label jobs start.
DATASETS=(cylinder3d halfcylinderRe640 tangaroa deltaWing_resampled smokeBuoyancy)
base_index=$((SLURM_ARRAY_TASK_ID % 5))
dataset=${DATASETS[$base_index]}
if (( SLURM_ARRAY_TASK_ID < 5 )); then
  phase=development
else
  phase=confirmation
fi
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Build_Task5_Multiscale_Cache.py \
  --config config/mainExp_Task5_3D_1.1.yaml \
  --phase "$phase" --dataset "$dataset"
