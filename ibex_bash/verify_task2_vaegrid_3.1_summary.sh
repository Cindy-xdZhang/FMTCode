#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2vg31s
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Sweep_Task2_VAE_3D.py \
  --config config/Verify_Task2_VAEGrid_3D_3.1.yaml --summarize
