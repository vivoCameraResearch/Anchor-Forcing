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
    inference/interactive_inference_multi.py \
    --config_path configs/interactive_inference.yaml \
    --seed $SEED

  echo "[$(date '+%F %T')] Run $i/$N done"
done
