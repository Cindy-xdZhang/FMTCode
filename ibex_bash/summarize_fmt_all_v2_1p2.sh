#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/fmt_all_v2_1p2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
date -Is
hostname
python -u experiments/Audit_FMTAllV2_3D.py --config config/Verify_FMTAllV2_1.2.json
python -u experiments/Summarize_FMTAllV2_3D.py --config config/Verify_FMTAllV2_1.2.json
date -Is
