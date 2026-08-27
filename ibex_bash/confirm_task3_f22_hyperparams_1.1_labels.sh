#!/bin/bash
#SBATCH -N 1
#SBATCH -p debug
#SBATCH -J FMTT3f22lab
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Build_Task3_GlobalIVD_Labels.py \
  --config config/Confirm_Task3_F22Hyperparams_1.1_labels.yaml
