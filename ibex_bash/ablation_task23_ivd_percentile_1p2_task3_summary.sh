#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-4%5
#SBATCH -J FMTT3ivd12s
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
repo_root=${FMT_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
tags=(p80 p85 p87p5 p90 p92p5)
tag=${tags[$SLURM_ARRAY_TASK_ID]}
python Run_Task3_FMTResidual_Frozen_4_1.py \
  --config "outputs/Ablation_Task23IVDPercentile_1.2/generated_configs/task3_${tag}_final.yaml" \
  --mode summary
