#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100|p100
#SBATCH --time=04:00:00
#SBATCH -J T4B41_fit
#SBATCH -o outputs/Verify_Task4B_VelocityCurlMemorization_4.1/logs/fit.%j.out
#SBATCH -e outputs/Verify_Task4B_VelocityCurlMemorization_4.1/logs/fit.%j.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
sha256sum -c DEPLOYMENT_MANIFEST_4p1.sha256 > outputs/Verify_Task4B_VelocityCurlMemorization_4.1/deployment_check_fit.txt
date -Is
hostname
nvidia-smi --query-gpu=name,uuid --format=csv
python -m experiments.Verify_Task4B_VelocityCurlMemorization train
python -m experiments.Verify_Task4B_VelocityCurlMemorization audit
