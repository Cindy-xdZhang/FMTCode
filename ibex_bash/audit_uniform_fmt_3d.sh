#!/bin/bash -l
#SBATCH --job-name=FMTuniformAudit
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err

set -euo pipefail

CONFIG=${1:?usage: audit_uniform_fmt_3d.sh CONFIG}
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python -u -m experiments.Audit_UniformFMT_3D --config "$CONFIG"
