#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
ACTION="${1:?missing action}"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-}}"

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

case "$ACTION" in
  strong_task1_select)
    python -m experiments.Run_Task123_StrongBaselines_1_1 \
      --config config/Verify_Task123_StrongBaselines_1.1.yaml \
      --mode task1-select --job-index "$INDEX"
    ;;
  strong_task1_freeze)
    python -m experiments.Run_Task123_StrongBaselines_1_1 \
      --config config/Verify_Task123_StrongBaselines_1.1.yaml --mode task1-freeze
    ;;
  strong_task1_run)
    python -m experiments.Run_Task123_StrongBaselines_1_1 \
      --config config/Verify_Task123_StrongBaselines_1.1.yaml \
      --mode task1-run --job-index "$INDEX"
    ;;
  strong_task1_merge)
    python -m experiments.Run_Task123_StrongBaselines_1_1 \
      --config config/Verify_Task123_StrongBaselines_1.1.yaml --mode task1-merge
    ;;
  strong_task2_freeze)
    python -m experiments.Run_Task123_StrongBaselines_1_1 \
      --config config/Verify_Task123_StrongBaselines_1.1.yaml --mode task2-freeze
    ;;
  strong_summarize)
    python -m experiments.Run_Task123_StrongBaselines_1_1 \
      --config config/Verify_Task123_StrongBaselines_1.1.yaml --mode summarize
    ;;
  strong_audit)
    python -m experiments.Audit_Task123_AdditionalEvidence_1_1 --kind strong \
      --config config/Verify_Task123_StrongBaselines_1.1.yaml
    ;;
  ablation_task1_merge)
    python -m experiments.Run_Task123_FMTComponentAblation_1_1 \
      --config config/Ablation_Task123_FMTComponents_1.1.yaml --mode task1-merge
    ;;
  ablation_summarize)
    python -m experiments.Run_Task123_FMTComponentAblation_1_1 \
      --config config/Ablation_Task123_FMTComponents_1.1.yaml --mode summarize
    ;;
  ablation_audit)
    python -m experiments.Audit_Task123_AdditionalEvidence_1_1 --kind ablation \
      --config config/Ablation_Task123_FMTComponents_1.1.yaml
    ;;
  ablation_cleanup)
    python -m experiments.Run_Task123_FMTComponentAblation_1_1 \
      --config config/Ablation_Task123_FMTComponents_1.1.yaml --mode cleanup
    ;;
  noise_task1_merge)
    python -m experiments.Run_Task123_NoiseRobustness_1_1 \
      --config config/Verify_Task123_NoiseRobustness_1.1.yaml --mode task1-merge
    ;;
  noise_summarize)
    python -m experiments.Run_Task123_NoiseRobustness_1_1 \
      --config config/Verify_Task123_NoiseRobustness_1.1.yaml --mode summarize
    ;;
  noise_audit)
    python -m experiments.Audit_Task123_AdditionalEvidence_1_1 --kind noise \
      --config config/Verify_Task123_NoiseRobustness_1.1.yaml
    ;;
  noise_cleanup)
    python -m experiments.Run_Task123_NoiseRobustness_1_1 \
      --config config/Verify_Task123_NoiseRobustness_1.1.yaml --mode cleanup
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
