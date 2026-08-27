#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3f22sel
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${TASK23_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Search_Task3_FMTResidual_Stage2_3D.py \
  --config outputs/Verify_Task3_F22Hyperparams_1.1/focused_config.yaml \
  --mode select
