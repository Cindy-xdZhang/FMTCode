#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2nfs24
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
python -m experiments.Run_Task2_3D_Main \
  --config config/mainExp_Task2_3D_2.4_newflows.yaml --summarize
