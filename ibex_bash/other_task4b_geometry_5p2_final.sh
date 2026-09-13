#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100|p100
#SBATCH --time=04:00:00
#SBATCH -J T4B52_final
#SBATCH -o outputs/Other_Task4B_GeometrySearch_5.2/logs/final.%j.out
#SBATCH -e outputs/Other_Task4B_GeometrySearch_5.2/logs/final.%j.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
sha256sum -c DEPLOYMENT_MANIFEST_5p2.sha256
date -Is
hostname
nvidia-smi --query-gpu=name,uuid --format=csv
python -m experiments.Task4B_GeometrySearch_5_2 select
python -m experiments.Task4B_GeometrySearch_5_2 final
python -m experiments.Task4B_GeometrySearch_5_2 audit
