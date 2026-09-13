#!/bin/bash
#SBATCH --job-name=aivd_camera_1p3
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --output=/home/zhanx0o/FMT_Uniform_3D_20260901/outputs/Verify_AIVDTranslationObservers_1.3/logs/%j.out
#SBATCH --error=/home/zhanx0o/FMT_Uniform_3D_20260901/outputs/Verify_AIVDTranslationObservers_1.3/logs/%j.err
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901/aivd_camera_1p3
export PYTHONPATH="$PWD:$PWD/experiments"
export NATURE_FIGURE_SKILL_ROOT=/home/zhanx0o/FMT_Uniform_3D_20260901/aivd_observers_1p1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --check SOURCE_MANIFEST.sha256
python -u experiments/Render_AIVDTranslationVelocityArrows_3D.py --config config/Verify_AIVDTranslationObservers_1.3.json
