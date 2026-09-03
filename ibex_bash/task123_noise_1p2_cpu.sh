#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
ACTION="${1:?missing action}"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-}}"
CONFIG=config/Verify_Task123_NoiseRobustness_1.2.yaml

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

case "$ACTION" in
  noise_task1)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG" \
      --mode task1 --job-index "$INDEX"
    ;;
  noise_task1_merge)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG" --mode task1-merge
    ;;
  noise_summarize)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG" --mode summarize
    ;;
  noise_audit)
    python experiments/Audit_Task123_AdditionalEvidence_1_1.py --kind noise-strong \
      --config "$CONFIG"
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
