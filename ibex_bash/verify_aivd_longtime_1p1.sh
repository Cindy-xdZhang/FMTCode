#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/aivd_longtime_1p1
export PYTHONPATH="$PWD:$PWD/experiments"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --quiet -c SOURCE_MANIFEST.sha256
date -Is
hostname
case "${MODE:-run}" in
  preflight_fixed)
    python -m unittest tests.test_aivd_transfer_3d tests.test_aivd_longtime_3d
    python -u -m experiments.Run_Task1235_AIVDLongtime_1_1 --preflight fixed ;;
  preflight_task5)
    python -u -m experiments.Run_Task1235_AIVDLongtime_1_1 --preflight task5 ;;
  cache)
    DATASETS=(cylinder3d halfcylinderRe640 halfcylinderRe6400)
    for PHASE in development confirmation; do
      python -u -m experiments.Build_Task5_Multiscale_Cache --config config/Verify_AIVDLongtime_1.1_task5_cache.yaml --dataset "${DATASETS[$SLURM_ARRAY_TASK_ID]}" --phase "$PHASE"
    done ;;
  audit)
    python -u -m experiments.Audit_Task1235_AIVDLongtime_1_1 ;;
  run)
    python -u -m experiments.Run_Task1235_AIVDLongtime_1_1 --task "$TASK" --index "$SLURM_ARRAY_TASK_ID" ;;
esac
date -Is
