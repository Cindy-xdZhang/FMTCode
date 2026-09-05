#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT1u41e
#SBATCH -o outputs/mainExp_Task1_3D_4.1/logs/%x.%j.out
#SBATCH -e outputs/mainExp_Task1_3D_4.1/logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G

set -euo pipefail
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p outputs/mainExp_Task1_3D_4.1/logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python -m experiments.Run_Task1_3D_Uniform \
  --config config/mainExp_Task1_3D_4.1_uniform.yaml --mode evaluate
