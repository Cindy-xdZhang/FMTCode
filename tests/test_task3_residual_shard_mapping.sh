#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
script="$repo_root/ibex_bash/mainexp_task3_3d_3.1_residual_shards_v100.sh"
expected_datasets=(
  cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa
  deltaWing_resampled deltaWing_LBM f22raptor channel
  boeing747 smokeBuoyancy
)

for task_id in $(seq 0 19); do
  output=$(TASK3_SHARD_DRY_RUN=1 TASK3_REPO_ROOT="$repo_root" \
    SLURM_ARRAY_TASK_ID=$task_id bash "$script")
  base_index=$((task_id % 10))
  dataset=${expected_datasets[$base_index]}
  if (( base_index < 8 )); then
    data_group=old8
  else
    data_group=new2
  fi
  if (( task_id < 10 )); then
    mode=fmt
  else
    mode=raw_pca
  fi
  [[ "$output" == *"task=$task_id dataset=$dataset group=$data_group mode=$mode"* ]]
  [[ "$output" == *"config/mainExp_Task3_3D_3.1_${mode}_${data_group}.yaml"* ]]
done

echo "TASK3 RESIDUAL SHARD MAPPING TEST PASSED"
