#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -J T4B51_build
#SBATCH -o outputs/mainExp_Task4B_PatchSegmentation_5.1/logs/build.%j.out
#SBATCH -e outputs/mainExp_Task4B_PatchSegmentation_5.1/logs/build.%j.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
sha256sum -c DEPLOYMENT_MANIFEST_5p1.sha256
date -Is
hostname
python -m unittest discover -s tests -p test_task4b_patch_segmentation_5_1.py
python -m experiments.Task4B_PatchSegmentation_5_1 build
