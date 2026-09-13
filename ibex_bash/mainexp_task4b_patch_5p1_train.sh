#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100|p100
#SBATCH --time=04:00:00
#SBATCH -J T4B51_train
#SBATCH -o outputs/mainExp_Task4B_PatchSegmentation_5.1/logs/train.%j.out
#SBATCH -e outputs/mainExp_Task4B_PatchSegmentation_5.1/logs/train.%j.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
sha256sum -c DEPLOYMENT_MANIFEST_5p1.sha256
date -Is
hostname
nvidia-smi --query-gpu=name,uuid --format=csv
python -m experiments.Task4B_PatchSegmentation_5_1 train
python -m experiments.Task4B_PatchSegmentation_5_1 audit
python -m experiments.Task4B_PatchSegmentation_5_1 render
