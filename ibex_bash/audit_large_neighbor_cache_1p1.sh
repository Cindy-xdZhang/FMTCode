#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/large_neighbor_1p1
export PYTHONPATH="$PWD:$PWD/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --quiet -c SOURCE_MANIFEST.sha256
sha256sum --quiet -c CACHE_AUDIT_MANIFEST.sha256
python -u -m experiments.Audit_LargeNeighbor_Cache
