#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J T4A_Repro11
#SBATCH -o outputs/Other_Task4A_FMTStreamlineClustering_1.1/ibex_repro_20260903/logs/%x.%j.out
#SBATCH -e outputs/Other_Task4A_FMTStreamlineClustering_1.1/ibex_repro_20260903/logs/%x.%j.err

# Reproduction of Other_Task4A_FMTStreamlineClustering_1.1 on Ibex (CPU).
# Frozen config; only input paths and output directory are overridden.
# Images are skipped here (--no-images) because the summary-figure step depends
# on a local figure-QA helper; views are rendered locally from the saved NPZ.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903}"
DATA_DIR="${FMT_CHANNEL_DATA_DIR:-/ibex/user/zhanx0o/FLowDataFolder/channel_flow}"
OUT="outputs/Other_Task4A_FMTStreamlineClustering_1.1/ibex_repro_20260903"

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

hostname
date -Is
printf 'deployment_base_commit='; cat DEPLOYMENT_BASE_COMMIT.txt
sha256sum -c DEPLOYMENT_MANIFEST.sha256
printf '%s  %s\n' c1e3c18d0e67b6eb33ea41cd49c3a4c55216838a0f4a42f778ef00aaba3d4564 "$DATA_DIR/channel.vtk" | sha256sum -c
printf '%s  %s\n' b9fdd93a1d2861781f824f28c80e46e06ebd3e5dcb58f0529010d7514e8fac09 "$DATA_DIR/channel_GTs.vtk" | sha256sum -c
python -m py_compile \
  experiments/Run_Task4A_FMTStreamlineClustering_1_1.py \
  FMT_Utils/Task4A_StreamlineClustering_3D.py \
  FMT_Utils/VoxelSegmentation_3D.py \
  FMT_Utils/DFT_FMT_3D.py

mkdir -p "$OUT/logs"
python -m experiments.Run_Task4A_FMTStreamlineClustering_1_1 \
  --config config/Other_Task4A_FMTStreamlineClustering_1.1.yaml \
  --gt "$DATA_DIR/channel_GTs.vtk" \
  --flow "$DATA_DIR/channel.vtk" \
  --output-dir "$OUT" \
  --no-images

sha256sum "$OUT/summary.json" "$OUT/per_vortex_topology_metrics.csv" \
  "$OUT/task4a_clustering_result.npz" > "$OUT/evidence_sha256.txt"
cat "$OUT/evidence_sha256.txt"
date -Is
