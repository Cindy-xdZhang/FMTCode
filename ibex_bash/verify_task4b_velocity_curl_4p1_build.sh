#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J T4B41_build
#SBATCH -o outputs/Verify_Task4B_VelocityCurlMemorization_4.1/logs/build.%j.out
#SBATCH -e outputs/Verify_Task4B_VelocityCurlMemorization_4.1/logs/build.%j.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
sha256sum -c DEPLOYMENT_MANIFEST_4p1.sha256 > outputs/Verify_Task4B_VelocityCurlMemorization_4.1/deployment_check_build.txt
date -Is
hostname
python -m unittest discover -s tests -p test_task4b_velocity_curl_labels_3d.py
python -m experiments.Verify_Task4B_VelocityCurlMemorization build --input-root inputs
