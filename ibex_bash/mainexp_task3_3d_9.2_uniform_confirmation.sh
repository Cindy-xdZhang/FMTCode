#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-19%20
#SBATCH -J FMTT3u92
#SBATCH -o outputs/mainExp_Task3_3D_9.2_uniform_confirmation/logs/%x.%A_%a.out
#SBATCH -e outputs/mainExp_Task3_3D_9.2_uniform_confirmation/logs/%x.%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

set -euo pipefail
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p outputs/mainExp_Task3_3D_9.2_uniform_confirmation/logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python -m experiments.Run_UniformFMT_Confirmation_3D \
  --config config/mainExp_Task3_3D_9.2_uniform_confirmation.yaml \
  --mode run --job-index "$SLURM_ARRAY_TASK_ID"

