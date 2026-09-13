#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/large_neighbor_1p1
export PYTHONPATH="$PWD:$PWD/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
sha256sum --quiet -c SOURCE_MANIFEST.sha256
date -Is
hostname
case "${MODE:-run}" in
  smoke)
    python -m unittest discover -s tests -p test_large_neighbor_3d.py
    python -u -m experiments.Build_LargeNeighbor_3D --dataset cylinder3d --smoke ;;
  cache)
    python -u -m experiments.Build_LargeNeighbor_3D --index "$SLURM_ARRAY_TASK_ID" ;;
  preflight)
    python -u -m experiments.Run_LargeNeighbor_3D --preflight fixed ;;
  run)
    python -u -m experiments.Run_LargeNeighbor_3D --task "$TASK" --index "$SLURM_ARRAY_TASK_ID" ;;
  audit)
    python -u -m experiments.Audit_LargeNeighbor_3D ;;
esac
date -Is
