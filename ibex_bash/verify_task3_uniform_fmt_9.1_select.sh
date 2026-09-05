#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3u91s
#SBATCH -o outputs/Verify_Task3_UniformFMT_9.1/logs/%x.%j.out
#SBATCH -e outputs/Verify_Task3_UniformFMT_9.1/logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p outputs/Verify_Task3_UniformFMT_9.1/logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python -m experiments.Search_Task3_FMTResidual_3D \
  --config config/Verify_Task3_UniformFMT_9.1.yaml --mode select
