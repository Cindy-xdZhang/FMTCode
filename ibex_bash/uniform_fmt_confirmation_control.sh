#!/bin/bash -l
#SBATCH --job-name=FMTuConfirmCtl
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o outputs/uniform_confirmation_control/logs/%x.%j.out
#SBATCH -e outputs/uniform_confirmation_control/logs/%x.%j.err

set -euo pipefail
mode=${1:?usage: uniform_fmt_confirmation_control.sh MODE CONFIG}
config=${2:?usage: uniform_fmt_confirmation_control.sh MODE CONFIG}
repo_root=${FMT_UNIFORM_REPO_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}
cd "$repo_root"
mkdir -p outputs/uniform_confirmation_control/logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
if [[ "$mode" == "audit" ]]; then
  python -u -m experiments.Audit_UniformFMT_3D \
    --confirmation-config "$config"
else
  python -u -m experiments.Run_UniformFMT_Confirmation_3D \
    --config "$config" --mode "$mode"
fi
