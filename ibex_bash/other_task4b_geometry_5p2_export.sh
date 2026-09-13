#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J T4B52_export
#SBATCH -o outputs/Other_Task4B_GeometrySearch_5.2/logs/export.%j.out
#SBATCH -e outputs/Other_Task4B_GeometrySearch_5.2/logs/export.%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
sha256sum -c DEPLOYMENT_EXPORT_5p2.sha256
date -Is
hostname
python -m experiments.Export_Task4B_GeometrySearch_5_2
