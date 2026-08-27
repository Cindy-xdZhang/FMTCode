#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3m41s
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Run_Task3_FMTResidual_Frozen_4_1.py \
  --config config/mainExp_Task3_3D_4.1.yaml --mode summary

