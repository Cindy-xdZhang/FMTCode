#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -J T4B51_view
#SBATCH -o outputs/mainExp_Task4B_PatchSegmentation_5.1/logs/render.%j.out
#SBATCH -e outputs/mainExp_Task4B_PatchSegmentation_5.1/logs/render.%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
date -Is
hostname
python -m experiments.Render_Task4B_PatchSegmentation_5_1
