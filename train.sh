#!/bin/bash

# Project path and config
CONFIG=configs/train_long.yaml
echo "CONFIG="$CONFIG

torchrun \
  --nnodes 1 \
  --nproc_per_node=8 \
  train.py \
  --config_path $CONFIG \