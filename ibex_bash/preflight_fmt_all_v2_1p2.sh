#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/fmt_all_v2_1p2
export PYTHONPATH="$PWD:/home/zhanx0o/FMT_Uniform_3D_20260901"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --quiet -c SOURCE_MANIFEST.sha256
date -Is
hostname
python -m unittest discover -s tests -p test_fmt_all_v2_3d.py
python -u -m experiments.Preflight_FMTAllV2_Control_3D
date -Is
