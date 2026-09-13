#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/aivd_transfer_1p1
export PYTHONPATH="$PWD:/home/zhanx0o/FMT_Uniform_3D_20260901"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --quiet -c SOURCE_MANIFEST.sha256
date -Is
hostname
if [[ "${MODE:-run}" == "preflight" ]]; then
  python -m unittest discover -s tests -p test_aivd_transfer_3d.py
  python -u -m experiments.Preflight_AIVDTransfer_3D
else
  python -u -m experiments.Verify_AIVDTransfer_3D --task "$TASK" --index "$SLURM_ARRAY_TASK_ID"
fi
date -Is
