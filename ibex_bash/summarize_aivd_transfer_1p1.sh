#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/aivd_transfer_1p1
export PYTHONPATH="$PWD:/home/zhanx0o/FMT_Uniform_3D_20260901"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
sha256sum --quiet -c SOURCE_MANIFEST.sha256
sha256sum --quiet -c REPORT_MANIFEST.sha256
date -Is
hostname
python -u -m experiments.Audit_AIVDTransfer_3D
python -u -m experiments.Summarize_AIVDTransfer_3D
date -Is
