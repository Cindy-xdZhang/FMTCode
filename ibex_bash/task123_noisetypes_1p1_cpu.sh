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
CONFIG_A=config/Verify_Task123_NoiseTypes_1.1.yaml
CONFIG_B=config/Verify_Task123_NoiseTypesStrong_1.1.yaml

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

case "$ACTION" in
  a_task1_merge)
    python experiments/Run_Task123_NoiseRobustness_1_1.py --config "$CONFIG_A" --mode task1-merge
    ;;
  a_summarize)
    python experiments/Run_Task123_NoiseRobustness_1_1.py --config "$CONFIG_A" --mode summarize
    ;;
  a_audit)
    python experiments/Audit_Task123_AdditionalEvidence_1_1.py --kind noise --config "$CONFIG_A"
    ;;
  a_cleanup)
    python experiments/Run_Task123_NoiseRobustness_1_1.py --config "$CONFIG_A" --mode cleanup
    ;;
  b_task1)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG_B" \
      --mode task1 --job-index "$INDEX"
    ;;
  b_task1_merge)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG_B" --mode task1-merge
    ;;
  b_summarize)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG_B" --mode summarize
    ;;
  b_audit)
    python experiments/Audit_Task123_AdditionalEvidence_1_1.py --kind noise-strong \
      --config "$CONFIG_B"
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
