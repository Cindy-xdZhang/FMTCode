#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -J T4B_GT098
#SBATCH -o outputs/Other_Task4B_ProxyGTThreshold_4.2/logs/render.%j.out
#SBATCH -e outputs/Other_Task4B_ProxyGTThreshold_4.2/logs/render.%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
date -Is
hostname
python -m experiments.Visualize_Task4B_ProxyGroundTruth_3D "$@"
