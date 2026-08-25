#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2h33
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Summarize_Task2_3D_Hierarchy.py \
  --task2 outputs/mainExp_Task2_3D_3.3/summary.json \
  --task1 \
    outputs/mainExp_Task1_3D_3.3_reference_old8/paper_table.csv \
    outputs/mainExp_Task1_3D_3.3_reference_new2/paper_table.csv \
  --output outputs/mainExp_Task2_3D_3.3/hierarchy.json
