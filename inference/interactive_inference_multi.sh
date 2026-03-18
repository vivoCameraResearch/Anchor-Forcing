#!/usr/bin/env bash
set -euo pipefail

# 循环次数
N=10

for (( i=1; i<=N; i++ )); do
  echo "[$(date '+%F %T')] Run $i/$N start"
  SEED=$((RANDOM % 10000 + 100))

  MASTER_PORT_ARG="--master_port=$((RANDOM % 10000 + 20000))"

  torchrun \
    --nproc_per_node=8 \
    ${MASTER_PORT_ARG} \
    /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11179416/code/LongLive/z_inference/interactive_inference_multi.py \
    --config_path /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11179416/code/LongLive/z_inference/configs/longlive_interactive_inference_mem.yaml \
    --seed $SEED

  echo "[$(date '+%F %T')] Run $i/$N done"
done
