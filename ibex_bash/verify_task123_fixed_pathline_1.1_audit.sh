#!/bin/bash -l
#SBATCH --job-name=FMTp123audit
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o outputs/Verify_Task123_FixedPathline_1.1/logs/%x.%j.out
#SBATCH -e outputs/Verify_Task123_FixedPathline_1.1/logs/%x.%j.err

set -euo pipefail
repo_root=${FMT_FIXED_PATHLINE_REPO_ROOT:-/ibex/scratch/zhanx0o/FMT_Task123_Pathline_20260902/repo}
cd "$repo_root"
mkdir -p outputs/Verify_Task123_FixedPathline_1.1/logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python -u -m experiments.Audit_Task123_FixedPathline \
  --config config/Verify_Task123_FixedPathline_1.1.yaml
