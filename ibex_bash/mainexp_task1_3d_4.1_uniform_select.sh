#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT1u41s
#SBATCH -o outputs/mainExp_Task1_3D_4.1/logs/%x.%j.out
#SBATCH -e outputs/mainExp_Task1_3D_4.1/logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p outputs/mainExp_Task1_3D_4.1/logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
hostname
python -m experiments.Run_Task1_3D_Uniform \
  --config config/mainExp_Task1_3D_4.1_uniform.yaml --mode select
