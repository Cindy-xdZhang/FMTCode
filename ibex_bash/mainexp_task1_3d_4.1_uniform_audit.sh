#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT1u41a
#SBATCH -o outputs/mainExp_Task1_3D_4.1/logs/%x.%j.out
#SBATCH -e outputs/mainExp_Task1_3D_4.1/logs/%x.%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p outputs/mainExp_Task1_3D_4.1/logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python -m experiments.Audit_Task1_Uniform_4_1
