#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -J T4B41_3D
#SBATCH -o outputs/Verify_Task4B_VelocityCurlMemorization_4.1/logs/render3d.%j.out
#SBATCH -e outputs/Verify_Task4B_VelocityCurlMemorization_4.1/logs/render3d.%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
date -Is
hostname
python -m experiments.Visualize_Task4B_VelocityCurl_3D
