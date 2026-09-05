#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3nfe23
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=a100
#SBATCH --mem=32G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python -m experiments.Evaluate_Task3_FrozenConfirmation \
  --config config/mainExp_Task3NewFlows_2.3_evaluate.yaml
