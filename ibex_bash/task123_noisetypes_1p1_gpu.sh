#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/home/zhanx0o/FMT_Uniform_3D_20260901}"
ACTION="${1:?missing action}"
INDEX="${SLURM_ARRAY_TASK_ID:?GPU actions require an array index}"
CONFIG_A=config/Verify_Task123_NoiseTypes_1.1.yaml
CONFIG_B=config/Verify_Task123_NoiseTypesStrong_1.1.yaml

module load cuda/11.8 || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "$ACTION" in
  a_task1)
    python experiments/Run_Task123_NoiseRobustness_1_1.py --config "$CONFIG_A" \
      --mode task1 --job-index "$INDEX"
    ;;
  a_task2)
    python experiments/Run_Task123_NoiseRobustness_1_1.py --config "$CONFIG_A" \
      --mode task2 --job-index "$INDEX"
    ;;
  a_task3)
    python experiments/Run_Task123_NoiseRobustness_1_1.py --config "$CONFIG_A" \
      --mode task3 --job-index "$INDEX"
    ;;
  b_task2)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG_B" \
      --mode task2 --job-index "$INDEX"
    ;;
  b_task3)
    python experiments/Run_Task123_NoiseRobustness_1_2.py --config "$CONFIG_B" \
      --mode task3 --job-index "$INDEX"
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
