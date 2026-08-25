#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2m33s
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Run_Task2_3D_Main.py \
  --config config/mainExp_Task2_3D_3.3.yaml --summarize
