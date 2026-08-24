#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3m31merge
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Merge_Task3_ResidualShards.py \
  --config config/mainExp_Task3_3D_3.1_evaluate.yaml
