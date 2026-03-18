torchrun \
  --nproc_per_node=1 \
  --master_port=$((RANDOM % 10000 + 10000)) \
  inference/interactive_inference.py \
  --config_path configs/interactive_inference.yaml