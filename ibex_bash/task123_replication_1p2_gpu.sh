#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
ACTION="${1:?missing action}"
INDEX="${SLURM_ARRAY_TASK_ID:?GPU actions require an array index}"
STRONG_CONFIG=config/Verify_Task123_StrongBaselines_1.2.yaml
ABLATION_CONFIG=config/Ablation_Task123_FMTComponents_1.2.yaml

module load cuda/11.8 || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "$ACTION" in
  strong_task2)
    python experiments/Run_Task123_StrongBaselines_1_1.py --config "$STRONG_CONFIG" \
      --mode task2-run --job-index "$INDEX"
    ;;
  strong_task3)
    python experiments/Run_Task123_StrongBaselines_1_1.py --config "$STRONG_CONFIG" \
      --mode task3-run --job-index "$INDEX"
    ;;
  ablation_task1)
    python experiments/Run_Task123_FMTComponentAblation_1_1.py --config "$ABLATION_CONFIG" \
      --mode task1 --job-index "$INDEX"
    ;;
  ablation_task2)
    python experiments/Run_Task123_FMTComponentAblation_1_1.py --config "$ABLATION_CONFIG" \
      --mode task2 --job-index "$INDEX"
    ;;
  ablation_task3)
    python experiments/Run_Task123_FMTComponentAblation_1_1.py --config "$ABLATION_CONFIG" \
      --mode task3 --job-index "$INDEX"
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
